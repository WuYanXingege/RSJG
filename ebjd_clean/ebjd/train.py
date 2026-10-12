"""Command line training entry point: ``python -m ebjd.train``."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.profiler import record_function
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


CERTIFIED_EXECUTION_MIGRATIONS = {
    "compact_grouped_v1": {
        "scene_forward_chunk": 1,
        "scene_compaction": True,
        "physical_group_max_scenes": 4,
        "physical_group_max_agent_slots": 96,
        "physical_group_max_padding_ratio": 1.5,
        "rollout_block_checkpointing": False,
    },
    "small_group_uncheckpoint_v1": {
        "scene_forward_chunk": 1,
        "scene_compaction": True,
        "physical_group_max_scenes": 4,
        "physical_group_max_agent_slots": 96,
        "physical_group_max_padding_ratio": 1.5,
        "rollout_block_checkpointing": False,
        "rollout_uncheckpointed_max_agent_slots": 32,
    },
}


def validate_execution_migration_settings(config: dict) -> dict:
    """Accept only explicit, versioned execution-only migration profiles."""
    training = config["training"]
    strategy = training.get("execution_strategy", "legacy_or_oom_safe_v1")
    if strategy == "legacy_or_oom_safe_v1":
        if int(training.get("scene_forward_chunk", 0)) != 1:
            raise ValueError(
                "legacy OOM recovery migration requires scene_forward_chunk=1")
        return {"strategy": strategy, "scene_forward_chunk": 1}
    expected = CERTIFIED_EXECUTION_MIGRATIONS.get(strategy)
    if expected is None:
        raise ValueError(f"execution strategy is not migration-certified: {strategy!r}")
    actual = {name: training.get(name) for name in expected}
    if actual != expected:
        raise ValueError(
            f"execution strategy {strategy!r} requires exact settings "
            f"{expected}, got {actual}")
    return {"strategy": strategy, **actual}


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
    resumes.add_argument(
        "--objective-fork-from",
        help="new experiment identity restored exactly from a bound checkpoint")
    parser.add_argument(
        "--resume-from-sha256",
        help="mandatory expected SHA256 for --resume-from")
    parser.add_argument(
        "--objective-fork-from-sha256",
        help="mandatory expected SHA256 for --objective-fork-from")
    parser.add_argument(
        "--objective-fork-contract", choices=("goal_v_weight_2",),
        help="strictly allow one preregistered semantic change")
    parser.add_argument(
        "--max-epochs-this-run", type=int,
        help="execution bound; preserve the original epoch-indexed LR schedule")
    parser.add_argument(
        "--absolute-stop-epoch", type=int,
        help="persistent inclusive epoch boundary retained across interruption resumes")
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


def resolve_stop_epoch(
    start_epoch: int, schedule_epochs: int,
    max_epochs_this_run: int | None, absolute_stop_epoch: int | None,
) -> int:
    """Resolve a bounded run without ever extending an absolute contract."""
    if start_epoch < 1 or schedule_epochs < 1:
        raise ValueError("epoch indices and schedule length must be positive")
    stop = int(schedule_epochs)
    if absolute_stop_epoch is not None:
        if absolute_stop_epoch < 1 or absolute_stop_epoch > schedule_epochs:
            raise ValueError("absolute stop epoch must lie within the LR schedule")
        if start_epoch > absolute_stop_epoch + 1:
            raise ValueError("resume starts beyond the persistent absolute stop epoch")
        stop = min(stop, int(absolute_stop_epoch))
    if max_epochs_this_run is not None:
        if max_epochs_this_run < 1:
            raise ValueError("max epochs this run must be positive")
        stop = min(stop, start_epoch + int(max_epochs_this_run) - 1)
    return stop


def new_short_selection(start_epoch: int, stop_epoch: int) -> dict:
    return {
        "definition": {
            "best_joint_short": "minimum JADE+0.5*JFDE within epochs 46-50",
            "best_marginal_short": "minimum minADE+minFDE within epochs 46-50",
            "least_violation_short": (
                "minimum summed positive marginal excess over bound baseline "
                "within epochs 46-50"),
        },
        "start_epoch": int(start_epoch), "stop_epoch": int(stop_epoch),
        "best_joint_short": None, "best_marginal_short": None,
        "least_violation_short": None,
    }


def short_role_scores(metrics: dict, baseline: dict, tolerance: float) -> dict:
    return {
        "best_joint_short": float(metrics["JADE"] + 0.5 * metrics["JFDE"]),
        "best_marginal_short": float(metrics["minADE"] + metrics["minFDE"]),
        "least_violation_short": float(
            max(0.0, metrics["minADE"] - float(baseline["minADE"]) - tolerance)
            + max(0.0, metrics["minFDE"] - float(baseline["minFDE"]) - tolerance)),
    }


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
            "strategy": str(config["training"].get(
                "execution_strategy", "legacy_or_oom_safe_v1")),
            "scene_forward_chunk": int(
                config["training"].get("scene_forward_chunk", 0)),
            "scene_compaction": bool(
                config["training"].get("scene_compaction", False)),
            "physical_group_max_scenes": int(
                config["training"].get("physical_group_max_scenes", 1)),
            "physical_group_max_agent_slots": int(
                config["training"].get("physical_group_max_agent_slots", 0)),
            "physical_group_max_padding_ratio": float(
                config["training"].get("physical_group_max_padding_ratio", 1.0)),
            "rollout_block_checkpointing": bool(
                config["training"].get("rollout_block_checkpointing", True)),
            "rollout_uncheckpointed_max_agent_slots": int(
                config["training"].get(
                    "rollout_uncheckpointed_max_agent_slots", 0)),
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
    migration_profile = validate_execution_migration_settings(config)
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
        "validated_execution_migration_profile": migration_profile,
        "allowed_changes": sorted(allowed_paths),
        "data_binding_checks": execution_identity["data"],
        "baseline_binding_checks": execution_identity["validation_comparator"],
        "origin_selected_artifacts": _selected_origin_artifacts(checkpoint),
        "origin_tensor_audit": audit,
    }


def validate_objective_fork_payload(
    payload: dict,
    checkpoint: Path,
    expected_sha256: str,
    config: dict,
    execution_identity: dict,
    ablation: str,
    contract: str,
) -> dict:
    """Allow exactly one preregistered objective change with full-state restore."""
    actual_sha256 = file_sha256(checkpoint)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"objective fork SHA256 mismatch: expected {expected_sha256}, got {actual_sha256}")
    if payload.get("schema") != "ebjd-checkpoint-v2" or \
            payload.get("epoch_boundary_only") is not True:
        raise ValueError("objective fork requires an epoch-boundary ebjd-checkpoint-v2")
    origin_config = payload.get("resolved_config")
    origin_identity = payload.get("run_identity")
    selection, rng = payload.get("selection_state"), payload.get("rng")
    required_rng = {"python", "numpy", "torch_cpu", "torch_cuda", "loader_generator"}
    if not isinstance(origin_config, dict) or not isinstance(origin_identity, dict):
        raise ValueError("objective fork checkpoint lacks config or run identity")
    if config_hash(origin_config) != origin_identity.get("config_hash"):
        raise ValueError("objective fork origin config hash is inconsistent")
    if not isinstance(selection, dict) or not isinstance(rng, dict) or \
            set(rng) != required_rng or rng.get("loader_generator") is None:
        raise ValueError("objective fork requires complete selection and RNG state")
    audit = _tensor_audit({
        "model": payload.get("model"), "optimizer": payload.get("optimizer")})
    if audit["nonfinite_count"]:
        raise ValueError("objective fork model/optimizer state is nonfinite")
    contracts = {
        "goal_v_weight_2": {
            "objectives.goal_v_weight": {"old": 1.0, "new": 2.0},
        },
    }
    if contract not in contracts:
        raise ValueError(f"unsupported objective fork contract: {contract!r}")
    differences = config_differences(origin_config, config)
    identity_changes = {"experiment.source_commit", "paths.output_root"}
    execution_fields = {".".join(path) for path in SEMANTIC_CONFIG_EXCLUSIONS}
    execution_fields -= identity_changes
    semantic_changes = set(contracts[contract])
    actual = set(differences)
    allowed = identity_changes | execution_fields | semantic_changes
    if not identity_changes.issubset(actual) or actual - allowed:
        raise ValueError(
            "objective fork differences violate the categorized frozen contract: "
            f"required_identity={sorted(identity_changes)}, "
            f"allowed_execution={sorted(execution_fields)}, "
            f"allowed_semantic={sorted(semantic_changes)}, "
            f"actual={sorted(differences)}")
    if actual & semantic_changes != semantic_changes:
        raise ValueError("objective fork is missing its sole semantic goal-weight change")
    for name, expected in contracts[contract].items():
        if differences[name] != expected:
            raise ValueError(
                f"objective fork value mismatch for {name}: {differences[name]}")
    identity_fields = (
        "implementation_base", "fold", "seed", "data", "initialization",
        "validation_comparator",
    )
    mismatched = [
        field for field in identity_fields
        if origin_identity.get(field) != execution_identity.get(field)]
    if origin_identity.get("ablation") != ablation:
        mismatched.append("ablation")
    if mismatched:
        raise ValueError(
            f"objective fork identity differs outside the contract: {sorted(mismatched)}")
    migration_profile = validate_execution_migration_settings(config)
    return {
        "schema": "ebjd-controlled-objective-fork-v1",
        "contract": contract,
        "origin_checkpoint_absolute_path": str(checkpoint.resolve()),
        "origin_checkpoint_sha256": actual_sha256,
        "origin_epoch": int(payload["epoch"]),
        "resume_start_epoch": int(payload["epoch"]) + 1,
        "origin_update_index": int(payload["update_index"]),
        "origin_source_commit": origin_identity.get("source_commit"),
        "origin_config_hash": origin_identity.get("config_hash"),
        "execution_source_commit": execution_identity["source_commit"],
        "execution_config_hash": execution_identity["config_hash"],
        "checked_config_diff": differences,
        "categorized_changes": {
            "identity": sorted(actual & identity_changes),
            "execution": sorted(actual & execution_fields),
            "semantic": sorted(actual & semantic_changes),
        },
        "validated_execution_migration_profile": migration_profile,
        "semantic_config_hash": semantic_config_hash(config),
        "full_state_policy": "model+optimizer+RNG+update+selection exact",
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
    if bool(args.objective_fork_from) != bool(args.objective_fork_from_sha256):
        raise ValueError(
            "--objective-fork-from and --objective-fork-from-sha256 must be "
            "provided together")
    if bool(args.objective_fork_from) != bool(args.objective_fork_contract):
        raise ValueError(
            "--objective-fork-contract is mandatory only for "
            "--objective-fork-from")
    if args.max_epochs_this_run is not None and args.max_epochs_this_run < 1:
        raise ValueError("--max-epochs-this-run must be positive")
    if args.absolute_stop_epoch is not None and args.absolute_stop_epoch < 1:
        raise ValueError("--absolute-stop-epoch must be positive")
    if args.migration_only and not (args.resume_from or args.objective_fork_from):
        raise ValueError(
            "--migration-only requires --resume-from or --objective-fork-from")
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
    objective_fork_receipt = None
    if args.resume or args.resume_from or args.objective_fork_from:
        checkpoint_path = Path(
            args.resume or args.resume_from or args.objective_fork_from).resolve()
        checkpoint_payload = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False)
        if args.resume:
            saved_identity = checkpoint_payload.get("run_identity", {})
            if "resume_origin" in saved_identity:
                identity["resume_origin"] = saved_identity["resume_origin"]
        elif args.resume_from:
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
        else:
            objective_fork_receipt = validate_objective_fork_payload(
                checkpoint_payload, checkpoint_path,
                args.objective_fork_from_sha256, config, identity,
                args.ablation, args.objective_fork_contract)
            identity["resume_origin"] = {
                "checkpoint_absolute_path": str(checkpoint_path),
                "checkpoint_sha256": args.objective_fork_from_sha256,
                "epoch": int(checkpoint_payload["epoch"]),
                "update_index": int(checkpoint_payload["update_index"]),
                "source_commit": checkpoint_payload["run_identity"].get(
                    "source_commit"),
                "run_id": checkpoint_payload["run_identity"].get("run_id"),
                "objective_fork_contract": args.objective_fork_contract,
            }
    if args.absolute_stop_epoch is not None:
        if checkpoint_payload is None:
            raise ValueError("absolute stop contracts require an epoch-boundary resume")
        saved_contract = checkpoint_payload.get("run_identity", {}).get(
            "bounded_training") if args.resume else None
        if saved_contract is not None:
            if int(saved_contract.get("absolute_stop_epoch", -1)) != int(
                    args.absolute_stop_epoch):
                raise ValueError("resume absolute stop differs from the persisted contract")
            identity["bounded_training"] = saved_contract
        else:
            identity["bounded_training"] = {
                "origin_epoch": int(checkpoint_payload["epoch"]),
                "absolute_stop_epoch": int(args.absolute_stop_epoch),
                "lr_schedule_total_epochs": int(config["training"]["epochs"]),
            }
    elif args.max_epochs_this_run is not None:
        identity["bounded_training"] = {
            "max_epochs_this_run": int(args.max_epochs_this_run),
            "lr_schedule_total_epochs": int(config["training"]["epochs"]),
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
        if not (args.resume or args.resume_from or args.objective_fork_from)
        else None)
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
    elif args.resume_from or args.objective_fork_from:
        active_receipt = (
            migration_receipt if args.resume_from else objective_fork_receipt)
        if checkpoint_payload is None or active_receipt is None:
            raise AssertionError("fork/migration payload was not validated")
        trainer.restore_checkpoint_payload(
            checkpoint_payload, loader_generator, require_identity=False)
        restoration_checks = audit_restored_state(
            trainer, checkpoint_payload, loader_generator)
        selector = ConstrainedSelector.from_state_dict(
            checkpoint_payload["selection_state"])
        start_epoch = int(checkpoint_payload["epoch"]) + 1
        active_receipt["restoration_checks"] = restoration_checks
        active_receipt["execution_run_identity"] = identity
        active_receipt["selection_state_summary"] = {
            "candidate_count": len(selector.candidates),
            "best_eligible": selector.best_eligible,
            "least_violation": selector.least_violation,
            "simultaneous_improvement": selector.best_eligible is not None,
        }
        receipt_name = (
            "MIGRATION_RECEIPT.json" if args.resume_from
            else "OBJECTIVE_FORK_RECEIPT.json")
        _atomic_json(output / receipt_name, active_receipt)
        _atomic_json(output / "SELECTION_STATUS.json", selector.state_dict())
        trainer.save_checkpoint(
            output / "last.pt", int(checkpoint_payload["epoch"]),
            selector.state_dict(), loader_generator,
            {"fork_or_migration_receipt": active_receipt})
        # Exercise the same strict restore path used by an ordinary resume from
        # the newly written execution identity before any optimizer update.
        migrated_payload = torch.load(
            output / "last.pt", map_location="cpu", weights_only=False)
        trainer.restore_checkpoint_payload(
            migrated_payload, loader_generator, require_identity=True)
        active_receipt["secondary_restore_checks"] = audit_restored_state(
            trainer, migrated_payload, loader_generator)
        _atomic_json(output / receipt_name, active_receipt)
        trainer.save_checkpoint(
            output / "last.pt", int(checkpoint_payload["epoch"]),
            selector.state_dict(), loader_generator,
            {"fork_or_migration_receipt": active_receipt})
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
    stop_epoch = resolve_stop_epoch(
        start_epoch, options.epochs, args.max_epochs_this_run,
        args.absolute_stop_epoch)
    bounded = identity.get("bounded_training", {})
    short_start_epoch = int(bounded.get("origin_epoch", start_epoch - 1)) + 1
    short_stop_epoch = int(bounded.get("absolute_stop_epoch", stop_epoch))
    short_path = output / "SHORT_RUN_SELECTION.json"
    if args.resume and short_path.is_file():
        short_roles = json.loads(short_path.read_text(encoding="utf-8"))
        if (
            int(short_roles.get("start_epoch", -1)) != short_start_epoch
            or int(short_roles.get("stop_epoch", -1)) != short_stop_epoch
        ):
            raise ValueError("persisted short selector window differs on resume")
    else:
        short_roles = new_short_selection(short_start_epoch, short_stop_epoch)
    updates_per_epoch = math.ceil(len(train_loader) / options.accumulation)
    origin_update = int(identity.get("resume_origin", {}).get(
        "update_index", trainer.update_index))
    run_contract = {
        "schema": "ebjd-absolute-epoch-pilot-contract-v1",
        "origin_epoch": short_start_epoch - 1,
        "absolute_stop_epoch": short_stop_epoch,
        "lr_schedule_total_epochs": options.epochs,
        "updates_per_epoch": updates_per_epoch,
        "origin_update_index": origin_update,
        "expected_terminal_update_index": origin_update + updates_per_epoch * (
            short_stop_epoch - short_start_epoch + 1),
        "current_resume_start_epoch": start_epoch,
        "current_invocation_stop_epoch": stop_epoch,
    }
    _atomic_json(output / "RUN_CONTRACT.json", run_contract)
    for epoch in range(start_epoch, stop_epoch + 1):
        epoch_started = time.perf_counter()
        epoch_update_start = trainer.update_index
        normal_step_seconds = 0.0
        rollout_step_seconds = 0.0
        projection_updates = 0
        rollout_updates = 0
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model.train()
        pending = []
        train_iterator = iter(train_loader)
        while True:
            try:
                with record_function("ebjd.cpu_data_loader"):
                    batch = next(train_iterator)
            except StopIteration:
                break
            with record_function("ebjd.cpu_augmentation"):
                augmented = augment_batch(batch, loader_generator)
            with record_function("ebjd.h2d_staging"):
                pending.append(
                    augmented if options.scene_forward_chunk else augmented.to(device))
            if len(pending) == options.accumulation:
                step_started = time.perf_counter()
                record = trainer.train_step(pending, epoch)
                step_seconds = time.perf_counter() - step_started
                record["step_wall_seconds"] = step_seconds
                if record["rollout_update"]:
                    rollout_step_seconds += step_seconds
                    rollout_updates += 1
                else:
                    normal_step_seconds += step_seconds
                projection_updates += int(record["projection_active"] > 0)
                nonfinite = [
                    name for name, value in record.items()
                    if isinstance(value, (int, float)) and not math.isfinite(float(value))]
                if nonfinite:
                    raise FloatingPointError(f"nonfinite training record fields: {nonfinite}")
                record.update({"epoch": epoch, "update": trainer.update_index})
                with record_function("ebjd.jsonl_logging"):
                    with log_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record, sort_keys=True) + "\n")
                pending = []
        if pending:
            step_started = time.perf_counter()
            record = trainer.train_step(pending, epoch)
            step_seconds = time.perf_counter() - step_started
            record["step_wall_seconds"] = step_seconds
            if record["rollout_update"]:
                rollout_step_seconds += step_seconds
                rollout_updates += 1
            else:
                normal_step_seconds += step_seconds
            projection_updates += int(record["projection_active"] > 0)
            nonfinite = [
                name for name, value in record.items()
                if isinstance(value, (int, float)) and not math.isfinite(float(value))]
            if nonfinite:
                raise FloatingPointError(f"nonfinite training record fields: {nonfinite}")
            record.update({"epoch": epoch, "update": trainer.update_index})
            with record_function("ebjd.jsonl_logging"):
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
        evaluation = config["evaluation"]
        worlds = 2 if args.smoke else int(evaluation["worlds"])
        steps = 2 if args.smoke else int(evaluation["steps"])
        validation_started = time.perf_counter()
        metrics = validate(
            model, validation_loader, str(device), worlds, steps,
            seed=int(evaluation["selection_seed"]))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        validation_seconds = time.perf_counter() - validation_started
        candidate = selector.consider(epoch, metrics)
        state = selector.state_dict()
        trainer.save_checkpoint(
            output / "last.pt", epoch, state, loader_generator,
            {"validation": metrics, "candidate": candidate})
        trainer.save_checkpoint(
            output / f"epoch_{epoch:03d}.pt", epoch, state, loader_generator,
            {"validation": metrics, "candidate": candidate,
             "checkpoint_role": "complete_epoch_pilot"})
        if selector.best_eligible is candidate:
            trainer.save_checkpoint(
                output / "best.pt", epoch, state, loader_generator,
                {"validation": metrics, "candidate": candidate})
        if selector.least_violation is candidate:
            trainer.save_checkpoint(
                output / "least_violation.pt", epoch, state, loader_generator,
                {"validation": metrics, "candidate": candidate})
        role_scores = short_role_scores(
            metrics, baseline,
            float(config["evaluation"].get("marginal_tolerance", 0)))
        for role, score in role_scores.items():
            current = short_roles[role]
            if current is None or score < current["score"]:
                short_roles[role] = {"epoch": epoch, "score": score,
                                     "metrics": metrics}
                trainer.save_checkpoint(
                    output / f"{role}.pt", epoch, state, loader_generator,
                    {"validation": metrics, "candidate": candidate,
                     "short_run_role": role, "short_run_score": score})
        _atomic_json(output / "SHORT_RUN_SELECTION.json", short_roles)
        (output / "SELECTION_STATUS.json").write_text(
            json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        epoch_summary = {
            "epoch": epoch,
            "start_update_exclusive": epoch_update_start,
            "end_update_inclusive": trainer.update_index,
            "updates": trainer.update_index - epoch_update_start,
            "wall_seconds": time.perf_counter() - epoch_started,
            "normal_step_seconds": normal_step_seconds,
            "rollout_step_seconds": rollout_step_seconds,
            "validation_seconds": validation_seconds,
            "rollout_updates": rollout_updates,
            "projection_active_updates": projection_updates,
            "peak_cuda_allocated_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda" else 0),
            "execution_strategy": config["training"].get(
                "execution_strategy", "legacy_or_oom_safe_v1"),
            "validation": metrics,
        }
        with (output / "EPOCH_SUMMARY.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(epoch_summary, sort_keys=True) + "\n")
        print(json.dumps({
            "epoch": epoch, "validation": metrics,
            "simultaneous_improvement": state["simultaneous_improvement"],
            "candidate": candidate, "epoch_summary": epoch_summary,
        }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
