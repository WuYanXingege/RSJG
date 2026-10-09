"""Command line training entry point: ``python -m ebjd.train``."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import (
    SEMANTIC_CONFIG_EXCLUSIONS, build_model, config_differences, config_hash,
    load_config, semantic_config_hash, trainer_options,
)
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
    resumes = parser.add_mutually_exclusive_group()
    resumes.add_argument(
        "--resume", help="epoch-boundary checkpoint in the original run directory")
    resumes.add_argument(
        "--resume-from",
        help="controlled migration from a bound checkpoint under a prior execution identity")
    parser.add_argument(
        "--resume-from-sha256",
        help="mandatory expected SHA256 for --resume-from")
    parser.add_argument(
        "--migration-only", action="store_true",
        help="validate --resume-from, write and reload the migrated boundary, then exit")
    parser.add_argument("--run-id", help="new unique run id; defaults to UTC-like local timestamp")
    parser.add_argument("--smoke", action="store_true", help="synthetic wiring only")
    return parser.parse_args()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _tree_equal(left, right) -> bool:
    if torch.is_tensor(left) and torch.is_tensor(right):
        return left.shape == right.shape and torch.equal(left.cpu(), right.cpu())
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return left.dtype == right.dtype and left.shape == right.shape and np.array_equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return list(left) == list(right) and all(
            _tree_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(
            _tree_equal(a, b) for a, b in zip(left, right, strict=True))
    return left == right


def _tensor_audit(payload: dict) -> dict[str, int]:
    result = {"tensor_count": 0, "element_count": 0, "nonfinite_count": 0}

    def visit(value) -> None:
        if torch.is_tensor(value):
            result["tensor_count"] += 1
            result["element_count"] += value.numel()
            if value.is_floating_point() or value.is_complex():
                result["nonfinite_count"] += int((~torch.isfinite(value)).sum())
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)

    visit(payload)
    return result


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
        "config_hash": config_hash(config),
        "semantic_config_hash": semantic_config_hash(config),
        "fold": config["experiment"]["fold"],
        "seed": int(config["seed"]), "ablation": ablation, "run_id": run_id,
        "execution": {
            "scene_forward_chunk": int(
                config["training"].get("scene_forward_chunk", 0)),
            "cpu_scene_staging": bool(
                config["training"].get("scene_forward_chunk", 0)),
            "pytorch_cuda_alloc_conf": os.environ.get(
                "PYTORCH_CUDA_ALLOC_CONF", ""),
        },
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


def _selected_origin_artifacts(checkpoint: Path) -> dict:
    artifacts = {}
    for name in ("least_violation.pt", "best.pt"):
        path = checkpoint.parent / name
        if path.is_file():
            artifacts[name] = {
                "absolute_path": str(path.resolve()),
                "sha256": file_sha256(path),
            }
    return artifacts


def validate_migration_payload(
    payload: dict,
    checkpoint: Path,
    expected_sha256: str,
    config: dict,
    execution_identity: dict,
    ablation: str,
) -> dict:
    """Validate the narrow execution-only migration contract."""
    actual_sha256 = file_sha256(checkpoint)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"resume-from SHA256 mismatch: expected {expected_sha256}, got {actual_sha256}")
    if payload.get("schema") != "ebjd-checkpoint-v2":
        raise ValueError("resume-from requires an ebjd-checkpoint-v2 checkpoint")
    if payload.get("epoch_boundary_only") is not True:
        raise ValueError("resume-from requires epoch_boundary_only=True")
    origin_config = payload.get("resolved_config")
    origin_identity = payload.get("run_identity")
    if not isinstance(origin_config, dict) or not isinstance(origin_identity, dict):
        raise ValueError("resume-from checkpoint lacks resolved config or run identity")
    if config_hash(origin_config) != origin_identity.get("config_hash"):
        raise ValueError("origin checkpoint config hash does not match its run identity")
    old_semantic_hash = semantic_config_hash(origin_config)
    new_semantic_hash = semantic_config_hash(config)
    if old_semantic_hash != new_semantic_hash:
        raise ValueError("semantic configuration differs from origin checkpoint")
    differences = config_differences(origin_config, config)
    allowed_paths = {".".join(path) for path in SEMANTIC_CONFIG_EXCLUSIONS}
    disallowed = sorted(set(differences) - allowed_paths)
    if disallowed:
        raise ValueError(f"resume-from has disallowed config changes: {disallowed}")
    if int(config["training"].get("scene_forward_chunk", 0)) != 1:
        raise ValueError("OOM recovery migration requires scene_forward_chunk=1")
    identity_fields = (
        "implementation_base", "fold", "seed", "data", "initialization",
        "validation_comparator",
    )
    mismatched_identity = [
        field for field in identity_fields
        if origin_identity.get(field) != execution_identity.get(field)]
    if origin_identity.get("ablation") != ablation:
        mismatched_identity.append("ablation")
    if mismatched_identity:
        raise ValueError(
            f"resume-from semantic identity differs: {sorted(mismatched_identity)}")
    audit = _tensor_audit({
        "model": payload.get("model"), "optimizer": payload.get("optimizer")})
    if audit["nonfinite_count"]:
        raise ValueError("origin model/optimizer state contains nonfinite tensors")
    selection = payload.get("selection_state")
    rng = payload.get("rng")
    required_rng = {"python", "numpy", "torch_cpu", "torch_cuda", "loader_generator"}
    if not isinstance(selection, dict) or not isinstance(rng, dict):
        raise ValueError("origin checkpoint lacks selection or RNG state")
    if set(rng) != required_rng or rng.get("loader_generator") is None:
        raise ValueError("origin checkpoint RNG state is incomplete")
    return {
        "schema": "ebjd-controlled-checkpoint-migration-v1",
        "origin_checkpoint_absolute_path": str(checkpoint.resolve()),
        "origin_checkpoint_sha256": actual_sha256,
        "origin_epoch": int(payload["epoch"]),
        "resume_start_epoch": int(payload["epoch"]) + 1,
        "origin_update_index": int(payload["update_index"]),
        "origin_source_commit": origin_identity.get("source_commit"),
        "origin_config_hash": origin_identity.get("config_hash"),
        "origin_run_identity": origin_identity,
        "execution_source_commit": execution_identity["source_commit"],
        "execution_config_hash": execution_identity["config_hash"],
        "semantic_config_hash": new_semantic_hash,
        "checked_semantic_diff": differences,
        "runtime_settings": execution_identity["execution"],
        "allowed_changes": sorted(allowed_paths),
        "data_binding_checks": execution_identity["data"],
        "baseline_binding_checks": execution_identity["validation_comparator"],
        "origin_selected_artifacts": _selected_origin_artifacts(checkpoint),
        "origin_tensor_audit": audit,
    }


def audit_restored_state(
    trainer: EBJDTrainer, payload: dict, loader_generator: torch.Generator,
) -> dict[str, bool]:
    model_exact = all(
        torch.equal(current.detach().cpu(), payload["model"][name].detach().cpu())
        for name, current in trainer.model.state_dict().items())
    current_optimizer = trainer.optimizer.state_dict()
    origin_optimizer = payload["optimizer"]
    optimizer_exact = (
        current_optimizer["lr"] == origin_optimizer["lr"]
        and current_optimizer["unet_lr"] == origin_optimizer["unet_lr"]
        and _tree_equal(current_optimizer["states"], origin_optimizer["states"])
    )
    rng_exact = _tree_equal(
        trainer._rng_state(loader_generator), payload["rng"])
    result = {
        "model_state_exact": model_exact,
        "optimizer_state_exact": optimizer_exact,
        "rng_state_exact": rng_exact,
        "update_index_exact": trainer.update_index == int(payload["update_index"]),
    }
    if not all(result.values()):
        raise ValueError(f"restored checkpoint state is not exact: {result}")
    return result


def main() -> None:
    args = parse_args()
    if bool(args.resume_from) != bool(args.resume_from_sha256):
        raise ValueError(
            "--resume-from and --resume-from-sha256 must be provided together")
    if args.migration_only and not args.resume_from:
        raise ValueError("--migration-only requires --resume-from")
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
    checkpoint_payload = None
    migration_receipt = None
    if args.resume or args.resume_from:
        checkpoint_path = Path(args.resume or args.resume_from).resolve()
        checkpoint_payload = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False)
        if args.resume:
            saved_identity = checkpoint_payload.get("run_identity", {})
            if "resume_origin" in saved_identity:
                identity["resume_origin"] = saved_identity["resume_origin"]
        else:
            migration_receipt = validate_migration_payload(
                checkpoint_payload, checkpoint_path,
                args.resume_from_sha256, config, identity, args.ablation)
            identity["resume_origin"] = {
                "checkpoint_absolute_path": str(checkpoint_path),
                "checkpoint_sha256": args.resume_from_sha256,
                "epoch": int(checkpoint_payload["epoch"]),
                "update_index": int(checkpoint_payload["update_index"]),
                "source_commit": checkpoint_payload["run_identity"].get(
                    "source_commit"),
                "run_id": checkpoint_payload["run_identity"].get("run_id"),
            }
    root = Path(config["paths"]["output_root"])
    output = root / identity["fold"] / str(seed) / args.ablation / run_id
    if args.resume:
        if output.resolve() != Path(args.resume).resolve().parent:
            raise ValueError("resume checkpoint must be inside its identity-derived run directory")
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError(f"new run directory already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(output / "RESOLVED_CONFIG.json", config)
    _atomic_json(output / "RUN_IDENTITY.json", identity)
    device = torch.device(args.device)
    model = build_model(config, args.ablation).to(device)
    options = trainer_options(config, args.ablation)
    if args.smoke:
        options.epochs = 1
        options.accumulation = 1
        options.rollout_start_epoch = 999
    train_loader = _loader(config, "train", args.smoke, loader_generator)
    validation_loader = _loader(config, "validation", args.smoke, loader_generator)
    fitted_scales = (
        fit_training_scales(model, train_loader)
        if not (args.resume or args.resume_from) else None)
    trainer = EBJDTrainer(model, options, config, identity)
    baseline = config["evaluation"]["matched_gdts_validation"]
    selector = ConstrainedSelector(
        float(baseline["minADE"]), float(baseline["minFDE"]),
        float(config["evaluation"].get("marginal_tolerance", 0)))
    start_epoch = 1
    if args.resume:
        if checkpoint_payload is None:
            raise AssertionError("resume payload was not loaded")
        trainer.restore_checkpoint_payload(
            checkpoint_payload, loader_generator, require_identity=True)
        selector = ConstrainedSelector.from_state_dict(
            checkpoint_payload["selection_state"])
        start_epoch = int(checkpoint_payload["epoch"]) + 1
    elif args.resume_from:
        if checkpoint_payload is None or migration_receipt is None:
            raise AssertionError("migration payload was not validated")
        trainer.restore_checkpoint_payload(
            checkpoint_payload, loader_generator, require_identity=False)
        restoration_checks = audit_restored_state(
            trainer, checkpoint_payload, loader_generator)
        selector = ConstrainedSelector.from_state_dict(
            checkpoint_payload["selection_state"])
        start_epoch = int(checkpoint_payload["epoch"]) + 1
        migration_receipt["restoration_checks"] = restoration_checks
        migration_receipt["execution_run_identity"] = identity
        migration_receipt["selection_state_summary"] = {
            "candidate_count": len(selector.candidates),
            "best_eligible": selector.best_eligible,
            "least_violation": selector.least_violation,
            "simultaneous_improvement": selector.best_eligible is not None,
        }
        _atomic_json(output / "MIGRATION_RECEIPT.json", migration_receipt)
        _atomic_json(output / "SELECTION_STATUS.json", selector.state_dict())
        trainer.save_checkpoint(
            output / "last.pt", int(checkpoint_payload["epoch"]),
            selector.state_dict(), loader_generator,
            {"migration_receipt": migration_receipt})
        # Exercise the same strict restore path used by an ordinary resume from
        # the newly written execution identity before any optimizer update.
        migrated_payload = torch.load(
            output / "last.pt", map_location="cpu", weights_only=False)
        trainer.restore_checkpoint_payload(
            migrated_payload, loader_generator, require_identity=True)
        migration_receipt["secondary_restore_checks"] = audit_restored_state(
            trainer, migrated_payload, loader_generator)
        _atomic_json(output / "MIGRATION_RECEIPT.json", migration_receipt)
        trainer.save_checkpoint(
            output / "last.pt", int(checkpoint_payload["epoch"]),
            selector.state_dict(), loader_generator,
            {"migration_receipt": migration_receipt})
        del migrated_payload
    if checkpoint_payload is not None:
        del checkpoint_payload
    if fitted_scales:
        print(json.dumps({"fitted_goal_scale": fitted_scales[0],
                          "fitted_bridge_scale": fitted_scales[1]}), flush=True)
    if args.migration_only:
        print(json.dumps({
            "migration_only": True,
            "output": str(output.resolve()),
            "epoch": start_epoch - 1,
            "update": trainer.update_index,
        }, sort_keys=True), flush=True)
        return
    log_path = output / "train.jsonl"
    for epoch in range(start_epoch, options.epochs + 1):
        model.train()
        pending = []
        for batch in train_loader:
            augmented = augment_batch(batch, loader_generator)
            pending.append(
                augmented if options.scene_forward_chunk else augmented.to(device))
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
