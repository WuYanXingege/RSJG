#!/usr/bin/env python3
"""Train/evaluate the fold-generic single-stage unified coupling bridge."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from ebjd.config import build_model, load_config
from ebjd.data import augment_batch_with_points
from ebjd.metrics import MetricAccumulator
from ebjd.objectives import (
    diffusion_loss, gated_geometry_mean, map_loss, relative_motion_loss,
)
from ebjd.proposals import NativeProposalBank, file_sha256
from ebjd.representation import cosine_vp
from ebjd.sampling import differentiable_sample
from ebjd.train import fit_training_scales
from ebjd.unified import (
    FrozenNativeGoalProvider, UnifiedCouplingBridge, gather_candidates,
    joint_pair_loss,
)
from run_zara1_native_prior_pilot import loader


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_bank(path: str, checkpoint_sha: str, manifest_sha: str, split: str):
    return NativeProposalBank(
        path, expected_checkpoint_sha256=checkpoint_sha,
        expected_manifest_sha256=manifest_sha, expected_split=split)


def optimizer_for(model, config):
    decay, no_decay, unet_decay, unet_no_decay = [], [], [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        is_unet = "bridge.encoder.map_unet" in name
        no_wd = parameter.ndim <= 1 or name.endswith("bias") or "norm" in name.lower()
        target = (unet_no_decay if no_wd else unet_decay) if is_unet else (
            no_decay if no_wd else decay)
        target.append(parameter)
    training = config["training"]
    groups = [
        {"params": decay, "lr": 1e-4, "weight_decay": float(training["weight_decay"])},
        {"params": no_decay, "lr": 1e-4, "weight_decay": 0.0},
        {"params": unet_decay, "lr": 1e-5, "weight_decay": float(training["weight_decay"])},
        {"params": unet_no_decay, "lr": 1e-5, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        groups, betas=tuple(training["optimizer_betas"]),
        eps=float(training["optimizer_eps"]))


def set_epoch_lr(optimizer, epoch: int) -> tuple[float, float]:
    if epoch <= 5:
        factor = epoch / 5
    else:
        progress = (epoch - 5) / 25
        factor = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
    for index, group in enumerate(optimizer.param_groups):
        group["lr"] = (1e-5 if index >= 2 else 1e-4) * factor
    return 1e-4 * factor, 1e-5 * factor


def bridge_and_pair_loss(
    model, batch, proposals, pair_weight=1.0, *, time_draw=None, noise=None,
):
    """One shared context and one combined objective graph."""
    context = model.encode_context(batch.observed, batch.semantic_maps, batch.valid)
    coupled, _, _ = model.coupling.assign(
        context, proposals, model.bridge.representation.goal_scale,
        straight_through=True)
    pair = joint_pair_loss(
        coupled, batch.future[..., -1, :], batch.valid,
        model.bridge.representation.goal_scale, temperature=0.1)
    endpoint = batch.future[..., -1, :].detach()
    clean = model.bridge.representation.encode_target_conditioned(
        batch.future.detach(), context.baseline, endpoint, eta=0.0)[:, None]
    if time_draw is None:
        time_draw = torch.randint(
            1, 101, (batch.observed.shape[0],), device=batch.observed.device)
        time_draw = time_draw.float() / 100
    if noise is None:
        noise = torch.randn_like(clean, dtype=torch.float32)
    alpha, sigma = cosine_vp(time_draw)
    alpha = alpha.to(clean).view(-1, 1, 1, 1, 1)
    sigma = sigma.to(clean).view(-1, 1, 1, 1, 1)
    noisy = alpha * clean.float() + sigma * noise
    target = alpha * noise - sigma * clean.float()
    final, coarse, coarse_path, _ = model.bridge.denoiser(
        noisy, time_draw, context, proposal_goals=endpoint[:, None], proposal_eta=0.0)
    final_clean = alpha * noisy - sigma * final.float()
    final_path, _ = model.bridge.representation.decode_conditioned(
        final_clean, context.baseline, endpoint[:, None], eta=0.0)
    options = model.training_options
    loss_diff = diffusion_loss(final, coarse, target, batch.valid, options["coarse"])
    per_scene_geo = 0.5 * (
        relative_motion_loss(coarse_path, batch.future, context, 6.0, 2.0, 4.8)
        + relative_motion_loss(final_path, batch.future, context, 6.0, 2.0, 4.8))
    loss_geo = gated_geometry_mean(per_scene_geo, alpha)
    loss_map = map_loss(batch.future, context, 0.5)
    agent_part = loss_diff + options["map"] * loss_map
    scene_part = options["geometry"] * loss_geo + pair_weight * pair
    bridge = agent_part + options["geometry"] * loss_geo
    return bridge + pair_weight * pair, bridge, pair, agent_part, scene_part


def bound_reference_metrics(path: str, seed: int, manifest_sha256: str) -> dict:
    """Resolve a comparator by identity, never by JSON list ordering."""
    reference = json.loads(Path(path).read_text(encoding="utf-8"))
    if reference.get("manifest_sha256") != manifest_sha256:
        raise ValueError("matched-GDTS top-level manifest SHA256 mismatch")
    matches = [row for row in reference.get("runs", [])
               if int(row.get("seed", -1)) == int(seed)]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one matched-GDTS seed {seed} run")
    run = matches[0]
    if run.get("manifest_sha256") != manifest_sha256:
        raise ValueError("matched-GDTS run manifest SHA256 mismatch")
    if int(run.get("worlds", -1)) != 20 or int(run.get("ddim_steps", -1)) != 20:
        raise ValueError("matched-GDTS comparator is not the bound 20/20 protocol")
    return run["metrics"]


@torch.no_grad()
def evaluate(model, config, bank, device, seed: int) -> dict:
    model.eval()
    g_acc, h_acc = MetricAccumulator(), MetricAccumulator()
    generator = torch.Generator(device=device).manual_seed(seed)
    started = time.perf_counter()
    for cpu_batch in loader(config, "validation", torch.Generator().manual_seed(99173)):
        proposals = bank.batch(cpu_batch).to(device)
        batch = cpu_batch.to(device)
        context = model.encode_context(batch.observed, batch.semantic_maps, batch.valid)
        h_goals, permutation, _ = model.coupling.assign(
            context, proposals, model.bridge.representation.goal_scale,
            straight_through=False)
        noise = torch.randn(
            batch.observed.shape[0], 20, batch.observed.shape[1], 12, 2,
            device=device, generator=generator)
        g_path, g_goal, _ = differentiable_sample(
            model.bridge, context, noise, 20, proposal_goals=proposals,
            proposal_eta=0.0)
        h_path, h_goal, _ = differentiable_sample(
            model.bridge, context, gather_candidates(noise, permutation), 20,
            proposal_goals=h_goals, proposal_eta=0.0)
        if not torch.equal(g_goal.sort(1).values, h_goal.sort(1).values):
            raise RuntimeError("G/H endpoint multisets changed")
        g_acc.update(g_path, batch.future, batch.valid)
        h_acc.update(h_path, batch.future, batch.valid)
    return {"seed": seed, "G": g_acc.as_dict(), "H": h_acc.as_dict(),
            "elapsed_seconds": time.perf_counter() - started}


def save_checkpoint(path, model, optimizer, epoch, update, loader_generator, selection, identity):
    payload = {
        "schema": "ebjd-unified-coupling-bridge-v1", "epoch": epoch,
        "update_index": update, "model": model.state_dict(),
        "optimizer": optimizer.state_dict(), "selection_state": selection,
        "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all(),
                "loader": loader_generator.get_state()},
        "run_identity": identity,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary); os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-bank", required=True)
    parser.add_argument("--validation-bank-pattern", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--fold", required=True)
    parser.add_argument("--compact-scenes", action="store_true")
    parser.add_argument("--max-gpu-seconds", type=float, default=43200)
    parser.add_argument("--final-reserve-seconds", type=float, default=5400)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output, device = Path(args.output).resolve(), torch.device(args.device)
    output.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    print(
        f"Starting unified {args.fold} run | workers="
        f"{config['training'].get('data_loader_workers', 0)} | "
        f"output={output}", flush=True)
    checkpoint_sha = config["paths"]["matched_gdts_validation_checkpoint_sha256"]
    train_manifest_sha = file_sha256(config["paths"]["train_manifest"])
    validation_sha = config["evaluation"]["matched_gdts_validation"]["manifest_sha256"]
    train_bank = make_bank(args.train_bank, checkpoint_sha, train_manifest_sha, "train")
    validation_banks = {seed: make_bank(
        args.validation_bank_pattern.format(seed=seed), checkpoint_sha,
        validation_sha, "validation") for seed in range(2035, 2040)}
    config_sha256 = file_sha256(args.config)
    seed_all(3101)
    bridge = build_model(config, proposal_conditioned=True)
    print("Fitting train-fold representation scales...", flush=True)
    scale_loader = loader(config, "train", torch.Generator().manual_seed(0))
    scales = fit_training_scales(bridge, scale_loader)
    if scales is None:
        raise RuntimeError("endpoint-bridge representation scales were not fitted")
    goal_scale, bridge_scale = scales
    print(
        f"Representation scales ready | goal={goal_scale:.8g} "
        f"bridge={bridge_scale:.8g}", flush=True)
    bridge = bridge.to(device)
    identity = {
        "schema": "ebjd-unified-coupling-bridge-run-v1", "fold": args.fold,
        "training_seed": 3101, "source_commit": args.source_commit,
        "config": str(Path(args.config).resolve()),
        "config_sha256": config_sha256,
        "representation_scales": {
            "goal": goal_scale, "bridge": bridge_scale,
            "source": "training_fold_only_existing_rms_rule"},
        "train_bank_sha256": train_bank.sha256,
        "validation_bank_sha256": {str(k): v.sha256 for k, v in validation_banks.items()},
        "rollout": False, "actual_step_projection": False,
        "worlds": 20, "ddim_steps": 20, "temperature": 1.0,
        "sinkhorn_iterations": 20, "pair_weight": 1.0,
        "compact_scenes": bool(args.compact_scenes),
    }
    atomic_json(output / "RUN_IDENTITY.json", identity)
    model = UnifiedCouplingBridge(bridge).to(device)
    model.training_options = {"coarse": 0.25, "geometry": 0.05, "map": 0.10}
    optimizer = optimizer_for(model, config)
    train_provider = FrozenNativeGoalProvider(train_bank)
    loader_generator = torch.Generator().manual_seed(3101)
    update = 0
    selection = {"best_jade": None, "best_eligible": None}
    start_epoch = 1
    last_checkpoint = output / "last.pt"
    if args.resume and last_checkpoint.exists():
        payload = torch.load(last_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        update = int(payload["update_index"])
        selection = payload["selection_state"]
        start_epoch = int(payload["epoch"]) + 1
        random.setstate(payload["rng"]["python"])
        np.random.set_state(payload["rng"]["numpy"])
        torch.set_rng_state(payload["rng"]["torch_cpu"])
        torch.cuda.set_rng_state_all(payload["rng"]["torch_cuda"])
        loader_generator.set_state(payload["rng"]["loader"])
    gdts = bound_reference_metrics(
        config["paths"]["matched_gdts_validation_result"], 2035,
        validation_sha)
    started = time.perf_counter()
    index_path = output / "checkpoint_index.csv"
    if not index_path.exists():
        with index_path.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(["epoch", "update", "eligible", "JADE", "checkpoint", "sha256"])
    completed_normally = True
    for epoch in range(start_epoch, 31):
        if time.perf_counter() - started >= args.max_gpu_seconds - args.final_reserve_seconds:
            atomic_json(output / "STATUS.json", {"status": "BUDGET_PARTIAL", "epoch": epoch - 1, "update": update})
            completed_normally = False
            break
        main_lr, unet_lr = set_epoch_lr(optimizer, epoch)
        model.train(); train_provider.eval()
        optimizer.zero_grad(set_to_none=True)
        losses, pairs, bridges = [], [], []
        train_started = time.perf_counter(); pending = 0
        for cpu_batch in loader(config, "train", loader_generator):
            raw_proposals = train_provider(cpu_batch)
            cpu_batch, proposals = augment_batch_with_points(
                cpu_batch, raw_proposals, loader_generator)
            logical_batch = int(cpu_batch.observed.shape[0])
            total_agents = int(cpu_batch.valid.sum())
            logical_time = torch.randint(
                1, 101, (logical_batch,), device=device).float() / 100
            logical_noise = torch.randn(
                logical_batch, 1, cpu_batch.observed.shape[1], 12, 2,
                dtype=torch.float32, device=device)
            logical_total = logical_pair = logical_bridge = 0.0
            groups = (range(logical_batch) if args.compact_scenes else [None])
            for scene in groups:
                if scene is None:
                    batch = cpu_batch.to(device)
                    physical_proposals = proposals.to(device)
                    time_draw, noise = logical_time, logical_noise
                    agent_weight = scene_weight = 1.0
                else:
                    positions = cpu_batch.valid[scene].nonzero(
                        as_tuple=False).flatten()
                    batch = cpu_batch.compact_scenes(scene, scene + 1).to(device)
                    physical_proposals = proposals[scene:scene + 1, :, positions].to(device)
                    time_draw = logical_time[scene:scene + 1]
                    noise = logical_noise[
                        scene:scene + 1, :, positions.to(device)]
                    agent_weight = float(positions.numel()) / total_agents
                    scene_weight = 1.0 / logical_batch
                with torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16,
                    enabled=device.type == "cuda" and torch.cuda.is_bf16_supported(),
                ):
                    _, bridge_loss, pair_loss, agent_part, scene_part = (
                        bridge_and_pair_loss(
                            model, batch, physical_proposals,
                            time_draw=time_draw, noise=noise))
                    total = agent_weight * agent_part + scene_weight * scene_part
                    scaled = total / int(config["training"]["gradient_accumulation"])
                scaled.backward()
                logical_total += float(total.detach())
                logical_pair += scene_weight * float(pair_loss.detach())
                logical_bridge += (
                    agent_weight * float(agent_part.detach())
                    + scene_weight * float(
                        (bridge_loss - agent_part).detach()))
            pending += 1
            losses.append(logical_total); pairs.append(logical_pair)
            bridges.append(logical_bridge)
            if pending == int(config["training"]["gradient_accumulation"]):
                coupling_grad = torch.sqrt(sum(
                    p.grad.detach().float().square().sum() for p in model.coupling.parameters()
                    if p.grad is not None))
                bridge_grad = torch.sqrt(sum(
                    p.grad.detach().float().square().sum() for p in model.bridge.parameters()
                    if p.grad is not None))
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step(); optimizer.zero_grad(set_to_none=True); update += 1; pending = 0
                record = {"epoch": epoch, "update": update, "loss": losses[-1],
                          "bridge_loss": bridges[-1], "pair_loss": pairs[-1],
                          "coupling_grad_norm": float(coupling_grad),
                          "bridge_grad_norm": float(bridge_grad), "lr": main_lr,
                          "unet_lr": unet_lr, "rollout_update": 0.0,
                          "projection_active": 0.0}
                with (output / "train.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                if update == 1:
                    atomic_json(output / "FIRST_UPDATE.json", record)
        if pending:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step(); optimizer.zero_grad(set_to_none=True); update += 1
        training_seconds = time.perf_counter() - train_started
        validation_started = time.perf_counter()
        validation = evaluate(model, config, validation_banks[2035], device, 2035)
        validation_seconds = time.perf_counter() - validation_started
        eligible = all(validation["H"][name] <= gdts[name] for name in gdts)
        epoch_path = output / f"epoch_{epoch:03d}.pt"
        state = {"epoch": epoch, "validation": validation, "eligible": eligible}
        # Selection is updated before the durable checkpoint is emitted so a
        # service restart cannot lose selector history.
        if selection["best_jade"] is None or validation["H"]["JADE"] < selection["best_jade"]["JADE"]:
            selection["best_jade"] = {"epoch": epoch, **validation["H"]}
        if eligible and (selection["best_eligible"] is None or validation["H"]["JADE"] < selection["best_eligible"]["JADE"]):
            selection["best_eligible"] = {"epoch": epoch, **validation["H"]}
        save_checkpoint(epoch_path, model, optimizer, epoch, update, loader_generator, selection, identity)
        shutil.copy2(epoch_path, output / "last.pt")
        last = output / "last.pt"; digest = file_sha256(epoch_path)
        if selection["best_jade"]["epoch"] == epoch:
            shutil.copy2(epoch_path, output / "best_jade.pt")
        if eligible and selection["best_eligible"]["epoch"] == epoch:
            shutil.copy2(epoch_path, output / "best_eligible.pt")
        with index_path.open("a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow([
                epoch, update, int(eligible), validation["H"]["JADE"],
                str(epoch_path), digest])
        epoch_record = {"epoch": epoch, "update": update,
                        "loss": statistics.fmean(losses),
                        "pair_loss": statistics.fmean(pairs),
                        "bridge_loss": statistics.fmean(bridges),
                        "G_minADE": validation["G"]["minADE"],
                        "G_minFDE": validation["G"]["minFDE"],
                        "G_JADE": validation["G"]["JADE"],
                        "G_JFDE": validation["G"]["JFDE"],
                        "H_minADE": validation["H"]["minADE"],
                        "H_minFDE": validation["H"]["minFDE"],
                        "H_JADE": validation["H"]["JADE"],
                        "H_JFDE": validation["H"]["JFDE"],
                        "eligible": eligible, "training_seconds": training_seconds,
                        "validation_seconds": validation_seconds,
                        "elapsed_seconds": time.perf_counter() - started}
        with (output / "epoch_metrics.csv").open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=epoch_record.keys())
            if handle.tell() == 0: writer.writeheader()
            writer.writerow(epoch_record)
        atomic_json(output / "STATUS.json", {"status": "TRAINING", **epoch_record,
                    "best": selection})
        elapsed = time.perf_counter() - started
        eta = elapsed / epoch * (30 - epoch)
        print(f"E{epoch:02d}/30 | loss={epoch_record['loss']:.6f} pair={epoch_record['pair_loss']:.6f} bridge={epoch_record['bridge_loss']:.6f} | G: ADE/FDE/JADE/JFDE={validation['G']['minADE']:.4f}/{validation['G']['minFDE']:.4f}/{validation['G']['JADE']:.4f}/{validation['G']['JFDE']:.4f} | H: ADE/FDE/JADE/JFDE={validation['H']['minADE']:.4f}/{validation['H']['minFDE']:.4f}/{validation['H']['JADE']:.4f}/{validation['H']['JFDE']:.4f} | train={training_seconds/60:.1f}m val={validation_seconds/60:.1f}m total={(training_seconds+validation_seconds)/60:.1f}m ETA={eta/3600:.2f}h", flush=True)
    if completed_normally and start_epoch <= 30:
        atomic_json(output / "STATUS.json", {"status": "TRAINING_COMPLETE_PENDING_FINAL_EVALUATION", "epoch": 30, "update": update, "best": selection})
    selected_name = "best_eligible.pt" if (output / "best_eligible.pt").exists() else "best_jade.pt"
    selected = output / selected_name
    if completed_normally and selected.exists() and time.perf_counter() - started < args.max_gpu_seconds:
        payload = torch.load(selected, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"], strict=True)
        selected_sha = file_sha256(selected)
        evaluation_dir = output / "evaluations"
        evaluation_dir.mkdir(exist_ok=True)
        final = []
        for seed in range(2035, 2040):
            seed_path = evaluation_dir / f"seed_{seed}.json"
            existing = None
            if seed_path.exists():
                candidate = json.loads(seed_path.read_text(encoding="utf-8"))
                if (candidate.get("selected_checkpoint_sha256") == selected_sha
                        and candidate.get("validation_bank_sha256")
                        == validation_banks[seed].sha256):
                    existing = candidate["result"]
            run = existing or evaluate(
                model, config, validation_banks[seed], device, seed)
            if existing is None:
                atomic_json(seed_path, {
                    "status": "complete", "seed": seed,
                    "selected_checkpoint_sha256": selected_sha,
                    "validation_bank_sha256": validation_banks[seed].sha256,
                    "manifest_sha256": validation_sha, "worlds": 20,
                    "ddim_steps": 20, "result": run})
            final.append(run)
        summary = {}
        for mode in ("G", "H"):
            summary[mode] = {}
            for metric in ("minADE", "minFDE", "JADE", "JFDE"):
                values = [run[mode][metric] for run in final]
                summary[mode][metric] = {
                    "mean": statistics.fmean(values),
                    "sample_standard_deviation": statistics.stdev(values)}
        result = {"status": "COMPLETE" if completed_normally else "BUDGET_PARTIAL",
                  "selected_checkpoint": str(selected),
                  "selected_checkpoint_sha256": selected_sha,
                  "selection_identity": selected_name, "runs": final,
                  "summary": summary, "optimizer_updates": update,
                  "gpu_wall_seconds": time.perf_counter() - started}
        atomic_json(output / "FINAL_EVALUATION.json", result)
        atomic_json(output / "STATUS.json", result)


if __name__ == "__main__":
    main()
