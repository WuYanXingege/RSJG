"""Stage-local complete-state resume at committed logical updates."""

import math
import os
import random
from pathlib import Path
import numpy as np
import torch
from bcep_formal.control import pause_requested
from bcep_formal.optimizations import finite_tensors
from bcep_joint.model import set_lr
from bcep_joint.common import write, save


RAW_METRICS = {
    "nll_final_sum": ("valid_agent_count", "train_nll_final_agent_micro"),
    "h4_nll_sum": ("valid_agent_count", "train_h4_nll_agent_micro"),
    "h8_nll_sum": ("valid_agent_count", "train_h8_nll_agent_micro"),
    "head_fde_sum": ("valid_agent_count", "train_head_minFDE_agent_micro"),
    "head_move_sum": ("valid_agent_count", "train_head_movement_agent_micro"),
    "head_loss_sum": ("valid_agent_count", "train_head_loss_agent_micro"),
    "clean_path_raw_sum": ("valid_scene_count", "train_clean_path_scene_macro"),
    "clean_path_signal_sum": ("valid_scene_count", "train_clean_path_alpha2_scene_macro"),
}
COUNTS = {count for count, _ in RAW_METRICS.values()}


def optimizer(model, lr, decay=0.0001):
    groups = [[], []]
    for name, param in model.named_parameters():
        if param.requires_grad:
            groups[int(param.ndim <= 1 or name.endswith("bias") or "norm" in name.lower())].append(param)
    return torch.optim.AdamW(
        [dict(params=groups[0], weight_decay=decay), dict(params=groups[1], weight_decay=0.0)], lr=lr
    )


def run_stage(folder, identity, recipe, model, train_ids, backward, validate, until_update, *, resume):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    last = folder / "last.pt"
    if last.exists() != resume:
        raise ValueError("existing stage requires resume; resume requires last.pt")
    if not train_ids or until_update < 0:
        raise ValueError("empty train split or negative session budget")
    opt = optimizer(model, recipe["lr"], recipe["weight_decay"])
    joint = identity["stage"] == "joint"
    plateau = (
        None
        if joint
        else torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, factor=0.5, patience=recipe["scheduler_patience"], min_lr=0.000001
        )
    )
    order_rng = torch.Generator().manual_seed(recipe["seed"])
    view_rng = torch.Generator().manual_seed(recipe["seed"] + 1)
    device = next(model.parameters()).device
    state = dict(
        schema="bcep-ebjd-integrated-state-v3",
        identity=identity,
        recipe=recipe,
        trainable=[n for n, p in model.named_parameters() if p.requires_grad],
        epoch=1,
        completed_epoch=0,
        update=0,
        cursor=0,
        order=None,
        history=[],
        epoch_sums={},
        epoch_raw_sums={},
        epoch_counts={},
        epoch_steps=0,
        stop_reason=None,
        max_updates=math.ceil(len(train_ids) / recipe["logical_scenes"]) * recipe["max_epochs"],
    )
    if resume:
        prior = torch.load(last, map_location="cpu", weights_only=False)
        for key in ("schema", "identity", "recipe", "trainable", "max_updates"):
            if prior[key] != state[key]:
                raise ValueError("resume mismatch: " + key)
        state = prior
        for key in ("epoch_sums", "epoch_steps", "epoch_raw_sums", "epoch_counts"):
            if key not in state:
                raise ValueError("missing partial epoch statistics: " + key)
        if state["status"] == "COMPLETE" or until_update <= state["update"]:
            return state
        model.load_state_dict(state["model"], strict=True)
        opt.load_state_dict(state["optimizer"])
        if plateau:
            plateau.load_state_dict(state["scheduler"])
        random.setstate(state["rng"]["python"])
        np.random.set_state(state["rng"]["numpy"])
        torch.set_rng_state(state["rng"]["cpu"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(state["rng"]["cuda"])
        order_rng.set_state(state["rng"]["order"])
        view_rng.set_state(state["rng"]["view"])

    def persist(status, completed=False):
        state.update(
            status=status,
            session_until_update=until_update,
            model=model.state_dict(),
            optimizer=opt.state_dict(),
            scheduler=plateau.state_dict() if plateau else state.get("scheduler"),
            rng=dict(
                cpu=torch.get_rng_state(),
                python=random.getstate(),
                numpy=np.random.get_state(),
                cuda=torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                order=order_rng.get_state(),
                view=view_rng.get_state(),
            ),
        )
        if completed:
            name = f"epoch_{state['completed_epoch']:03d}.pt"
            save(state, folder / name)
            temp = folder / "last.link"
            temp.unlink(missing_ok=True)
            os.link(folder / name, temp)
            temp.replace(last)
            ranking = sorted(state["history"], key=lambda row: (row["validation"], row["epoch"]))[:5]
            for alias, target in [("best.pt", ranking[0]["file"])]:
                temp = folder / (alias + ".link")
                temp.unlink(missing_ok=True)
                os.link(folder / target, temp)
                temp.replace(folder / alias)
            from bcep.contracts import digest

            write(
                folder / "TOP5.json", [dict(row, sha256=digest(folder / row["file"])) for row in ranking]
            )
            write(folder / "EPOCH_METRICS.json", state["history"])
        else:
            save(state, last)
        return state

    while True:
        if pause_requested() or state["update"] >= until_update:
            return persist("SESSION_BUDGET_PAUSED")
        if state["order"] is None:
            state["order"] = [
                train_ids[i] for i in torch.randperm(len(train_ids), generator=order_rng).tolist()
            ]
            if joint:
                state["scheduler"] = set_lr(opt, state["epoch"], recipe["scheduler_horizon"])
        group = state["order"][state["cursor"] : state["cursor"] + recipe["logical_scenes"]]
        model.train()
        opt.zero_grad(set_to_none=True)
        diagnostics = backward(group, order_rng, view_rng)
        torch.nn.utils.clip_grad_norm_(model.parameters(), recipe["clip"], error_if_nonfinite=True)
        opt.step()
        finite_tensors(model.parameters(), "post-update parameters")
        state["cursor"] += len(group)
        state["update"] += 1
        state["last_update_metrics"] = {key: float(value) for key, value in diagnostics.items()}
        state["epoch_steps"] += 1
        for key, value in state["last_update_metrics"].items():
            if key in COUNTS:
                if not value.is_integer() or value <= 0:
                    raise ValueError("invalid observation count: " + key)
                totals = state["epoch_counts"]
                value = int(value)
            elif key in RAW_METRICS:
                totals = state["epoch_raw_sums"]
            else:
                totals = state["epoch_sums"]
            totals[key] = totals.get(key, 0) + value
        if state["cursor"] != len(train_ids):
            continue
        model.eval()
        with torch.no_grad():
            value = float(validate())
        if not math.isfinite(value):
            raise FloatingPointError("nonfinite validation metric")
        if plateau:
            plateau.step(value)
        epoch = state["epoch"]
        state["history"].append(
            dict(
                epoch=epoch,
                update=state["update"],
                validation=value,
                metrics=getattr(validate, "metrics", {}),
                train_raw_sums=state["epoch_raw_sums"],
                train_counts=state["epoch_counts"],
                **{
                    RAW_METRICS[k][1]: v / state["epoch_counts"][RAW_METRICS[k][0]]
                    for k, v in state["epoch_raw_sums"].items()
                },
                train_mean_per_update={
                    key: value / state["epoch_steps"] for key, value in state["epoch_sums"].items()
                },
                file=f"epoch_{epoch:03d}.pt",
            )
        )
        best = min(state["history"], key=lambda row: row["validation"])
        stale = epoch - best["epoch"]
        state.update(
            epoch=epoch + 1,
            completed_epoch=epoch,
            cursor=0,
            order=None,
            stale=stale,
            epoch_sums={},
            epoch_raw_sums={},
            epoch_counts={},
            epoch_steps=0,
        )
        done = epoch >= recipe["max_epochs"] or (epoch >= recipe["minimum"] and stale >= recipe["patience"])
        state["stop_reason"] = (
            ("max_epochs" if epoch >= recipe["max_epochs"] else "early_stop") if done else None
        )
        persist("COMPLETE" if done else "EPOCH_COMPLETE", True)
        if done:
            return state
