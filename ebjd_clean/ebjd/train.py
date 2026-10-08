"""Command line training entry point: ``python -m ebjd.train``."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import build_model, config_hash, load_config, trainer_options
from .data import SceneManifestDataset, SyntheticSceneDataset, augment_batch, collate_scenes
from .metrics import MetricAccumulator
from .representation import EndpointBridgeRepresentation, cv_baseline
from .sampling import predict
from .selection import ConstrainedSelector
from .trainer import EBJDTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train independent EBJD")
    parser.add_argument("--config", required=True)
    parser.add_argument("--ablation", default="none")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", help="epoch-boundary checkpoint in the original run directory")
    parser.add_argument("--run-id", help="new unique run id; defaults to UTC-like local timestamp")
    parser.add_argument("--smoke", action="store_true", help="synthetic wiring only")
    return parser.parse_args()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset(config: dict, split: str, smoke: bool):
    if smoke:
        return SyntheticSceneDataset(scenes=2, agents=2, pixels=32)
    path = config["paths"].get(f"{split}_manifest")
    if not path:
        raise ValueError(f"paths.{split}_manifest is required outside --smoke")
    return SceneManifestDataset(path)


def _loader(config: dict, split: str, smoke: bool, generator: torch.Generator) -> DataLoader:
    batch_size = 1 if smoke else int(config["training"]["batch_scenes"])
    return DataLoader(
        _dataset(config, split, smoke), batch_size=batch_size,
        shuffle=split == "train", collate_fn=collate_scenes,
        num_workers=0, generator=generator)


@torch.no_grad()
def fit_training_scales(model, loader) -> tuple[float, float] | None:
    if not isinstance(model.representation, EndpointBridgeRepresentation):
        return None
    goal_sum = bridge_sum = 0.0
    goal_count = bridge_count = 0
    for batch in loader:
        baseline, _ = cv_baseline(batch.observed)
        unscaled = model.representation._encode_unscaled(
            batch.future.double(), baseline.double())
        goal_mask = batch.valid[..., None, None].expand_as(unscaled[..., 11:, :])
        bridge_mask = batch.valid[..., None, None].expand_as(unscaled[..., :11, :])
        goal_sum += float(unscaled[..., 11:, :][goal_mask].square().sum())
        bridge_sum += float(unscaled[..., :11, :][bridge_mask].square().sum())
        goal_count += int(goal_mask.sum())
        bridge_count += int(bridge_mask.sum())
    goal_scale = max((goal_sum / max(goal_count, 1)) ** 0.5, 0.5)
    bridge_scale = max((bridge_sum / max(bridge_count, 1)) ** 0.5, 0.1)
    model.representation.set_scales(goal_scale, bridge_scale)
    model.encoder.goal_scale.fill_(goal_scale)
    return goal_scale, bridge_scale


@torch.no_grad()
def validate(
    model, loader, device: str, worlds: int, steps: int, seed: int = 2035,
) -> dict:
    model.eval()
    accumulator = MetricAccumulator()
    generator = torch.Generator(device=device).manual_seed(seed)
    for batch in loader:
        batch = batch.to(device)
        trajectory, _ = predict(
            model, batch.observed, batch.semantic_maps, batch.valid,
            worlds=worlds, steps=steps, generator=generator)
        accumulator.update(trajectory, batch.future, batch.valid)
    return accumulator.as_dict()


def _identity(config: dict, ablation: str, run_id: str, smoke: bool) -> dict:
    paths = config["paths"]
    data_identity = {}
    for split in ("train", "validation", "test"):
        path = paths.get(f"{split}_manifest")
        data_identity[split] = (
            {"path": str(Path(path).resolve()), "sha256": file_sha256(path)}
            if path and Path(path).exists() and not smoke else {"synthetic": True})
    source = paths.get("accepted_fold_matched_gdts_unet")
    configured_source = config["experiment"].get("source_commit")
    git_root = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], check=True,
        capture_output=True, text=True).stdout.strip()
    git_head = subprocess.run(
        ["git", "-C", git_root, "rev-parse", "HEAD"], check=True, capture_output=True,
        text=True).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "-C", git_root, "status", "--porcelain", "--", "ebjd_clean"], check=True,
        capture_output=True, text=True).stdout.strip())
    if configured_source and configured_source != git_head:
        raise ValueError("experiment.source_commit does not match checked-out HEAD")
    if dirty and not smoke:
        raise ValueError("formal training refuses a dirty ebjd_clean source tree")
    return {
        "source_commit": git_head, "working_tree_dirty": dirty,
        "implementation_base": config["experiment"]["implementation_base"],
        "config_hash": config_hash(config), "fold": config["experiment"]["fold"],
        "seed": int(config["seed"]), "ablation": ablation, "run_id": run_id,
        "data": data_identity,
        "initialization": {
            "path": str(Path(source).resolve()) if source else None,
            "sha256": paths.get("accepted_fold_matched_gdts_unet_sha256"),
        },
        "validation_comparator": {
            "checkpoint_path": str(Path(
                paths["matched_gdts_validation_checkpoint"]).resolve()),
            "checkpoint_sha256": paths[
                "matched_gdts_validation_checkpoint_sha256"],
            "result_path": str(Path(
                paths["matched_gdts_validation_result"]).resolve()),
            "result_sha256": paths["matched_gdts_validation_result_sha256"],
        },
    }


def verify_formal_baseline_binding(config: dict) -> dict:
    """Recompute every external identity used by validation checkpoint selection."""
    baseline_binding = config["evaluation"]["matched_gdts_validation"]
    if baseline_binding.get("status") != "bound_to_manifest":
        raise ValueError(
            "formal training requires matched_gdts_validation.status=bound_to_manifest; "
            "the published fragment reference cannot select synchronized-scene checkpoints")
    validation_manifest = config["paths"]["validation_manifest"]
    manifest_payload = json.loads(Path(validation_manifest).read_text())
    manifest_statistics = manifest_payload["statistics"]
    comparator_checkpoint = config["paths"]["matched_gdts_validation_checkpoint"]
    comparator_result = config["paths"]["matched_gdts_validation_result"]
    binding_checks = {
        "manifest_sha256": file_sha256(validation_manifest),
        "checkpoint_sha256": file_sha256(comparator_checkpoint),
        "result_sha256": file_sha256(comparator_result),
        "ebjd_initialization_sha256": config["paths"][
            "accepted_fold_matched_gdts_unet_sha256"],
        "fold": config["experiment"]["fold"], "unit": "m",
        "worlds": int(config["evaluation"]["worlds"]),
        "steps": int(config["evaluation"]["steps"]),
        "evaluation_seed": int(config["evaluation"]["selection_seed"]),
        "scene_count": int(manifest_statistics["scenes"]),
        "agent_occurrence_count": int(manifest_statistics["agents"]),
    }
    path_hash_checks = {
        "checkpoint_path_sha256": (
            config["paths"]["matched_gdts_validation_checkpoint_sha256"],
            binding_checks["checkpoint_sha256"]),
        "result_path_sha256": (
            config["paths"]["matched_gdts_validation_result_sha256"],
            binding_checks["result_sha256"]),
    }
    bad_paths = {
        name: {"expected": expected, "actual": actual}
        for name, (expected, actual) in path_hash_checks.items()
        if expected != actual}
    if bad_paths:
        raise ValueError(f"matched GDTS source file SHA256 mismatch: {bad_paths}")
    mismatched = {
        key: {"expected": expected, "actual": baseline_binding.get(key)}
        for key, expected in binding_checks.items()
        if baseline_binding.get(key) != expected}
    if mismatched:
        raise ValueError(f"matched GDTS baseline binding mismatch: {mismatched}")
    return binding_checks


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if not args.smoke:
        verify_formal_baseline_binding(config)
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    loader_generator = torch.Generator().manual_seed(seed + 17)
    run_id = args.run_id or (
        Path(args.resume).resolve().parent.name if args.resume
        else datetime.now().strftime("%Y%m%d_%H%M%S"))
    identity = _identity(config, args.ablation, run_id, args.smoke)
    root = Path(config["paths"]["output_root"])
    output = root / identity["fold"] / str(seed) / args.ablation / run_id
    if args.resume:
        if output.resolve() != Path(args.resume).resolve().parent:
            raise ValueError("resume checkpoint must be inside its identity-derived run directory")
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError(f"new run directory already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "RESOLVED_CONFIG.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "RUN_IDENTITY.json").write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    device = torch.device(args.device)
    model = build_model(config, args.ablation).to(device)
    options = trainer_options(config, args.ablation)
    if args.smoke:
        options.epochs = 1
        options.accumulation = 1
        options.rollout_start_epoch = 999
    train_loader = _loader(config, "train", args.smoke, loader_generator)
    validation_loader = _loader(config, "validation", args.smoke, loader_generator)
    fitted_scales = fit_training_scales(model, train_loader) if not args.resume else None
    trainer = EBJDTrainer(model, options, config, identity)
    baseline = config["evaluation"]["matched_gdts_validation"]
    selector = ConstrainedSelector(
        float(baseline["minADE"]), float(baseline["minFDE"]),
        float(config["evaluation"].get("marginal_tolerance", 0)))
    start_epoch = 1
    if args.resume:
        payload = trainer.load_checkpoint(args.resume, loader_generator)
        selector = ConstrainedSelector.from_state_dict(payload["selection_state"])
        start_epoch = int(payload["epoch"]) + 1
    if fitted_scales:
        print(json.dumps({"fitted_goal_scale": fitted_scales[0],
                          "fitted_bridge_scale": fitted_scales[1]}), flush=True)
    log_path = output / "train.jsonl"
    for epoch in range(start_epoch, options.epochs + 1):
        model.train()
        pending = []
        for batch in train_loader:
            pending.append(augment_batch(batch, loader_generator).to(device))
            if len(pending) == options.accumulation:
                record = trainer.train_step(pending, epoch)
                record.update({"epoch": epoch, "update": trainer.update_index})
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                pending = []
        if pending:
            record = trainer.train_step(pending, epoch)
            record.update({"epoch": epoch, "update": trainer.update_index})
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        evaluation = config["evaluation"]
        worlds = 2 if args.smoke else int(evaluation["worlds"])
        steps = 2 if args.smoke else int(evaluation["steps"])
        metrics = validate(
            model, validation_loader, str(device), worlds, steps,
            seed=int(evaluation["selection_seed"]))
        candidate = selector.consider(epoch, metrics)
        state = selector.state_dict()
        trainer.save_checkpoint(
            output / "last.pt", epoch, state, loader_generator,
            {"validation": metrics, "candidate": candidate})
        if selector.best_eligible is candidate:
            trainer.save_checkpoint(
                output / "best.pt", epoch, state, loader_generator,
                {"validation": metrics, "candidate": candidate})
        if selector.least_violation is candidate:
            trainer.save_checkpoint(
                output / "least_violation.pt", epoch, state, loader_generator,
                {"validation": metrics, "candidate": candidate})
        (output / "SELECTION_STATUS.json").write_text(
            json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({
            "epoch": epoch, "validation": metrics,
            "simultaneous_improvement": state["simultaneous_improvement"],
            "candidate": candidate,
        }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
