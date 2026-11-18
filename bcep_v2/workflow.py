"""Guarded one-stage CLI. Inspect is read-only and never constructs a training queue."""

import argparse
import fcntl
import json
from pathlib import Path
import torch
import yaml
from bcep.contracts import canonical_hash, digest, require
from bcep_joint.common import local_seed, read, write, seed_all
from bcep_formal.data import FormalData
from bcep_formal.control import install
from .provider import GEOMETRY, Provider
from .bank import Bank, ExportPaused, bank_identity, check_export_pause, export
from .joint import build_joint
from .stages import density_callbacks, head_callbacks, joint_callbacks
from .state import run_stage

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/bcep_ebjd_v2/RESOLVED_CONFIG.yaml"
SEEDS = range(2035, 2040)
METRIC = dict(density="dev_nll_final_agent_micro", head="dev_minfde20_agent_micro", joint="dev_H_JADE")


def load_contract(path, plan_path, fold):
    cfg = yaml.safe_load(Path(path).read_text())
    fixed = yaml.safe_load(CONFIG.read_text())
    require(
        {k: v for k, v in cfg.items() if k != "allow_real_training"}
        == {k: v for k, v in fixed.items() if k != "allow_real_training"},
        "unsupported contract",
    )
    plan = read(plan_path)
    require(digest(plan["split_lock"]) == cfg["split_lock_sha256"], "split lock mismatch")
    descriptor = plan["folds"][fold]["split"]
    req = cfg["folds"][fold]
    require(digest(descriptor["path"]) == req["split_sha256"], "split changed")
    split = read(descriptor["path"])
    require(
        (split["fold"], split["unit"], split["dt"], split["pred_length"]) == (fold, "m", 0.4, 12),
        "protocol mismatch",
    )
    require(
        split["scales"] == req["density_scales"] and split["bridge_scales"] == req["bridge_scales"],
        "scale mismatch",
    )
    sources = {}
    for package in (
        "bcep",
        "bcep_joint",
        "bcep_formal",
        "bcep_confirmation",
        "ebjd",
        "selector_control",
        "bcep_v2",
    ):
        for item in sorted((ROOT / package).rglob("*.py")):
            sources[str(item.relative_to(ROOT))] = digest(item)
    for name, sha in cfg["geometry_source"].items():
        require(digest(ROOT / name) == sha, "geometry source mismatch")
        sources[name] = sha
    geometry = dict(
        layout=GEOMETRY,
        split=req["split_sha256"],
        crop_sources=cfg["geometry_source"],
        records=canonical_hash([r["sha256"] for r in split["records"]]),
    )
    original = Path(plan["assets"]).resolve()
    folder = original.parent / "bcep_ebjd_v2_integrated" / fold
    require(
        original == Path(plan_path).resolve().parent and folder.resolve() == folder,
        "ambiguous output root",
    )
    ident = dict(
        schema=cfg["schema"],
        fold=fold,
        config=canonical_hash(cfg),
        sources=canonical_hash(sources),
        split=req["split_sha256"],
        members=canonical_hash(split["members"]),
        geometry=canonical_hash(geometry),
    )
    return cfg, split, folder, geometry, ident


def selected(folder, stage, identity, model):
    """Require a terminal run and recompute its selector from the full history."""
    last = torch.load(folder / "last.pt", map_location="cpu", weights_only=False)
    require(
        last["status"] == "COMPLETE" and last["identity"] == identity, "stage unfinished or incompatible"
    )
    row = min(last["history"], key=lambda r: (r["validation"], r["epoch"]))
    path = folder / row["file"]
    sha = digest(path)
    require(
        digest(folder / "best.pt") == sha and read(folder / "TOP5.json")[0]["sha256"] == sha,
        "selector mismatch",
    )
    weights = torch.load(path, map_location="cpu", weights_only=False)
    require(
        weights["identity"] == identity
        and weights["update"] == row["update"] > 0
        and weights["completed_epoch"] == row["epoch"],
        "selected state mismatch",
    )
    model.load_state_dict(weights["model"], strict=True)
    model.eval().requires_grad_(False)
    receipt = dict(
        stage=stage,
        trained=True,
        frozen=True,
        weights_sha256=sha,
        epoch=row["epoch"],
        updates=row["update"],
        selection=METRIC[stage],
        value=row["validation"],
        identity=canonical_hash(identity),
    )
    if stage == "head":
        receipt["base"] = identity["base"]
    if (folder / "FROZEN.json").exists():
        require(read(folder / "FROZEN.json") == receipt, "frozen selection changed")
    else:
        write(folder / "FROZEN.json", receipt)
    return receipt


def recipe(cfg, stage):
    if stage == "joint":
        c = cfg["joint"]
        return dict(
            seed=c["downstream_seed"],
            lr=c["base_lr"],
            weight_decay=c["weight_decay"],
            clip=c["clip_norm"],
            logical_scenes=c["batch_scenes"] * c["gradient_accumulation"],
            max_epochs=c["max_epochs"],
            minimum=c["min_completed_epochs"],
            patience=c["patience"],
            scheduler_horizon=c["scheduler_horizon"],
        )
    c = cfg["provider"]["density" if stage == "density" else "candidate_head"]
    return dict(
        seed=cfg["provider_seed" if stage == "density" else "head_seed"],
        lr=c["lr"],
        weight_decay=c["weight_decay"],
        clip=c["grad_clip_norm"],
        logical_scenes=c["logical_batch_scenes"],
        max_epochs=c["max_epochs"],
        minimum=c["min_completed_epochs"],
        patience=c["early_stop_patience"],
        scheduler_patience=c["scheduler"]["patience"],
    )


def execute(args, cfg, split, root, geometry, ident):
    stage, device = args.stage, torch.device(args.device)
    if device.type == "cuda":
        from .runtime import configure_cuda

        configure_cuda()
    for path, value in [(root / "GEOMETRY.json", geometry), (root / "CONFIG.json", cfg)]:
        if path.exists():
            require(read(path) == value, "existing v2 output has a different contract")
        else:
            write(path, value)
    train_stage = stage in METRIC
    if train_stage and (args.until_update is None or args.until_update < 1):
        raise ValueError("training requires an explicit absolute update budget")
    seed_all(recipe(cfg, stage if train_stage else "density")["seed"])
    provider = Provider().to(device)
    provider.base.scales.copy_(torch.tensor(split["scales"], device=device))
    provider.base.scales_fitted.fill_(True)
    data = FormalData(split, "training" if train_stage else "export")

    def train(name, model, identity, callbacks):
        state = run_stage(
            root / name,
            identity,
            recipe(cfg, name),
            model,
            split["train_indices"],
            *callbacks,
            args.until_update,
            resume=args.resume,
            record_progress=(root.parent / "FORMAL_RUN.json").is_file(),
        )
        if state["status"] == "COMPLETE":
            selected(root / name, name, identity, model)
        return dict(stage=name, status=state["status"], update=state["update"])

    density_id = dict(ident, stage="density")
    if stage == "density":
        return train(stage, provider.base, density_id, density_callbacks(provider, data, device))
    base = selected(root / "density", "density", density_id, provider.base)
    head = None

    def cache(purpose, seed, *, view=None, create=False, refined=False):
        parent = root / ("refined_banks" if refined else "base_cache")
        folder = parent / f"{purpose}_{seed}"
        check_export_pause(folder)
        identity = bank_identity(
            split, purpose, seed, geometry, ident["sources"], base, head if refined else None, view=view
        )
        if create:

            def predict(i):
                x = data.item(i, purpose, False)
                obs = x["observed"].to(device)
                return provider.candidates(
                    obs,
                    x["semantic_maps"].to(device),
                    torch.ones(len(obs), device=device, dtype=torch.bool),
                    x["scene_id"],
                    x["agent_ids"],
                    seed,
                    geometry,
                    refined=refined,
                )

            export(folder, identity, split, predict)
        check_export_pause(folder)
        bank = Bank(folder, identity, split, interruptible=True)
        check_export_pause(folder)
        return bank

    base_train = [
        cache(
            "train",
            local_seed("bcep-v2-train-view", cfg["provider_seed"], v),
            view=v,
            create=stage == "base-cache",
        )
        for v in range(2)
    ]
    base_dev = [cache("selection_dev", s, create=stage == "base-cache") for s in SEEDS]
    if stage == "base-cache":
        return dict(stage=stage, complete_banks=len(base_train) + len(base_dev))
    head_id = dict(ident, stage="head", base=base, caches=[b.hash for b in base_train + base_dev])
    if stage == "head":
        return train(
            stage,
            provider.head,
            head_id,
            head_callbacks(provider, data, device, geometry, base_train, base_dev),
        )
    head = selected(root / "head", "head", head_id, provider.head)
    training = cache("train", 2035, create=stage == "banks", refined=True)
    dev = cache("selection_dev", 2035, create=stage == "banks", refined=True)
    if stage == "banks":
        return dict(stage=stage, train=training.hash, dev=dev.hash)
    joint_cfg = dict(cfg["joint"], scales=split["bridge_scales"])
    seed_all(joint_cfg["downstream_seed"])
    model = build_joint(joint_cfg).to(device)
    joint_id = dict(
        ident, stage="joint", base=base, head=head, train_bank=training.hash, dev_bank=dev.hash
    )
    if stage == "joint":
        return train(
            stage, model, joint_id, joint_callbacks(model, data, device, joint_cfg, training, dev)
        )
    frozen = selected(root / "joint", "joint", joint_id, model)
    test = {seed: cache("heldout_confirmation", seed, create=True, refined=True) for seed in SEEDS}
    from .evaluation import evaluate

    return evaluate(split, model, frozen, test, root / "evaluation", device)


PAUSED_RESUMABLE_EXIT_CODE = 75


def run_command(operation, root, stage):
    """Translate a resumable export interruption to the process exit status."""
    try:
        result = operation()
    except ExportPaused as exc:
        result = dict(status="PAUSED_RESUMABLE", stage=stage, bank=exc.folder.name, next_index=exc.next_index)
        write(root / "EXPORT_PAUSED.json", result)
        print(json.dumps(result), flush=True)
        return PAUSED_RESUMABLE_EXIT_CODE
    print(json.dumps(result), flush=True)
    return 0 if result.get("status") != "SESSION_BUDGET_PAUSED" else PAUSED_RESUMABLE_EXIT_CODE


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "stage", choices=["inspect", "density", "base-cache", "head", "banks", "joint", "evaluate"]
    )
    p.add_argument("--plan", required=True)
    p.add_argument("--fold", choices=["zara2", "zara1", "hotel", "univ", "eth"], required=True)
    p.add_argument("--config", default=str(CONFIG))
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--until-update", type=int)
    args = p.parse_args(argv)
    cfg, split, root, geometry, identity = load_contract(args.config, args.plan, args.fold)
    if args.stage == "inspect" or not args.execute:
        print(
            json.dumps(
                dict(
                    dry_run=True,
                    stage=args.stage,
                    identity=identity,
                    allow_real_training=cfg["allow_real_training"],
                )
            )
        )
        return
    if cfg["allow_real_training"] is not True:
        raise SystemExit("v2 execution disabled by configuration")
    if (root.parent / "FORMAL_RUN.json").exists():
        import os
        from .formal_manifest import validate_worker_lease
        validate_worker_lease(root.parent, os.environ.get("BCEP_FORMAL_LOCK_FD"))
        registered = read(root.parent / "FORMAL_RUN.json")
        require(identity == registered["folds"][args.fold]["identity"]
                and digest(args.config) == registered["config_sha256"]
                and Path(args.plan).resolve() == Path(registered["input_plan"])
                and args.device == "cuda", "formal worker identity differs")
        if args.stage in METRIC:
            require(args.until_update == registered["folds"][args.fold]["caps"][args.stage],
                    "formal worker requires its registered absolute update cap")
        if args.stage == "evaluate":
            raise ValueError("formal test requires the global freeze and the read-only matrix entry")
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        install()
        return run_command(lambda: execute(args, cfg, split, root, geometry, identity), root, args.stage)


def cli():
    raise SystemExit(main())


if __name__ == "__main__":
    cli()
