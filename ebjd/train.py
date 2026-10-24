"""Command line training entry point: ``python -m ebjd.train``."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
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
    "paper_training_v1": {
        "scene_forward_chunk": 1,
        "scene_compaction": True,
        "physical_group_max_scenes": 4,
        "physical_group_max_agent_slots": 96,
        "physical_group_max_padding_ratio": 1.5,
        "rollout_block_checkpointing": False,
        "rollout_uncheckpointed_max_agent_slots": 0,
        "training_runtime_optimizations": True,
    },
    "prefetch_runtime_v1": {
        "scene_forward_chunk": 1,
        "scene_compaction": True,
        "physical_group_max_scenes": 4,
        "physical_group_max_agent_slots": 96,
        "physical_group_max_padding_ratio": 1.5,
        "rollout_block_checkpointing": False,
        "rollout_uncheckpointed_max_agent_slots": 32,
        "training_runtime_optimizations": True,
        "data_loader_workers": 4,
        "data_loader_prefetch_factor": 2,
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
    parser = argparse.ArgumentParser(
        description="Train Endpoint--Bridge Joint Diffusion (EBJD FULL)")
    parser.add_argument("--config", required=True)
    parser.add_argument("--ablation", choices=("none",), default="none")
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
    parser.add_argument(
        "--research-fork", choices=("ctf_decoder_energy_v1",),
        help="permit one explicitly audited semantic research fork")
    parser.add_argument(
        "--stop-after-epoch", type=int,
        help="stop cleanly after this epoch while retaining the configured schedule length")
    parser.add_argument("--run-id", help="new unique run id; defaults to UTC-like local timestamp")
    parser.add_argument(
        "--startup-contract",
        help="fail-closed formal launch contract checked before the first update")
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


EPOCH_METRIC_FIELDS = (
    "epoch", "train_loss", "minADE", "minFDE", "JADE", "JFDE",
    "epoch_minutes", "eta_minutes", "validated",
)


def format_epoch_summary(
    epoch: int, epochs: int, train_loss: float,
    metrics: dict[str, float] | None, epoch_minutes: float, eta_minutes: float,
) -> str:
    """Return the single human-readable line emitted after a complete epoch."""
    shown = {
        name: "—" if metrics is None else f"{metrics[name]:.6f}"
        for name in ("minADE", "minFDE", "JADE", "JFDE")
    }
    return (
        f"Epoch {epoch:03d}/{epochs:03d} | "
        f"loss={train_loss:.6f} | minADE={shown['minADE']} | "
        f"minFDE={shown['minFDE']} | JADE={shown['JADE']} | "
        f"JFDE={shown['JFDE']} | time={epoch_minutes:.2f}min | "
        f"ETA={eta_minutes / 60.0:.2f}h"
    )


def validate_startup_contract(
    contract_path: str | Path, config_path: str | Path, config: dict,
    args: argparse.Namespace,
) -> dict:
    """Verify the immutable ZARA1 formal contract before any optimizer update."""
    contract_file = Path(contract_path).resolve()
    payload = json.loads(contract_file.read_text(encoding="utf-8"))
    if payload.get("schema") != "ebjd-zara1-e250-rollout-off-launch-v1":
        raise ValueError("startup contract schema mismatch")
    resolved = Path(config_path).resolve()
    checks = {
        "resolved_config_path": str(resolved),
        "resolved_config_sha256": file_sha256(resolved),
        "semantic_config_hash": semantic_config_hash(config),
        "source_commit": subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parents[1]),
             "rev-parse", "HEAD"], check=True, capture_output=True,
            text=True).stdout.strip(),
        "epochs": int(config["training"]["epochs"]),
        "rollout_training_enabled": bool(
            config["training"].get("rollout_training_enabled", True)),
        "paired_marginal_actual_step_projection_enabled": bool(
            config["training"].get(
                "paired_marginal_actual_step_projection_enabled", True)),
        "hidden_dim": int(config["denoiser"]["hidden_dim"]),
        "attention_heads": int(config["denoiser"]["heads"]),
        "ff_dim": int(config["denoiser"]["ff_dim"]),
        "execution_strategy": str(config["training"].get("execution_strategy")),
        "training_runtime_optimizations": bool(
            config["training"].get("training_runtime_optimizations", False)),
        "data_loader_workers": int(
            config["training"].get("data_loader_workers", 0)),
        "data_loader_prefetch_factor": int(
            config["training"].get("data_loader_prefetch_factor", 0)),
        "evaluation_worlds": int(config["evaluation"]["worlds"]),
        "evaluation_steps": int(config["evaluation"]["steps"]),
        "run_id": args.run_id,
        "ablation": args.ablation,
        "device": args.device,
    }
    expected = payload.get("expected", {})
    mismatches = {
        key: {"expected": value, "actual": checks.get(key)}
        for key, value in expected.items() if checks.get(key) != value
    }
    if mismatches:
        raise ValueError(f"startup contract mismatch: {mismatches}")
    if checks["epochs"] != 250:
        raise ValueError("formal ZARA1 startup requires epochs=250")
    if checks["rollout_training_enabled"]:
        raise ValueError("formal ZARA1 startup requires rollout training OFF")
    if checks["paired_marginal_actual_step_projection_enabled"]:
        raise ValueError("formal ZARA1 startup requires paired projection OFF")
    if (checks["hidden_dim"], checks["attention_heads"], checks["ff_dim"]) != (
        192, 4, 768,
    ):
        raise ValueError("formal ZARA1 startup requires architecture 192/4/768")
    if checks["evaluation_worlds"] != 20 or checks["evaluation_steps"] != 20:
        raise ValueError("formal ZARA1 inference contract requires 20 worlds/20 steps")
    return {
        "schema": "ebjd-startup-gate-receipt-v1",
        "contract_path": str(contract_file),
        "contract_sha256": file_sha256(contract_file),
        "checks": checks,
        "resume": str(Path(args.resume).resolve()) if args.resume else None,
        "passed": True,
    }


def append_epoch_metrics(path: Path, row: dict[str, float | int]) -> None:
    """Atomically append one complete epoch without weakening raw JSONL logs."""
    unknown = set(row) - set(EPOCH_METRIC_FIELDS)
    missing = set(EPOCH_METRIC_FIELDS) - set(row)
    if unknown or missing:
        raise ValueError(
            f"epoch metric fields differ: unknown={sorted(unknown)}, "
            f"missing={sorted(missing)}")
    previous = path.read_text(encoding="utf-8") if path.exists() else ""
    if previous:
        reader = csv.DictReader(previous.splitlines())
        if tuple(reader.fieldnames or ()) != EPOCH_METRIC_FIELDS:
            raise ValueError("existing epoch_metrics.csv header is incompatible")
        epochs = [int(item["epoch"]) for item in reader]
        if epochs and int(row["epoch"]) <= epochs[-1]:
            raise ValueError("epoch_metrics.csv requires strictly increasing epochs")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        if previous:
            handle.write(previous)
            if not previous.endswith("\n"):
                handle.write("\n")
            writer = csv.DictWriter(handle, fieldnames=EPOCH_METRIC_FIELDS)
        else:
            writer = csv.DictWriter(handle, fieldnames=EPOCH_METRIC_FIELDS)
            writer.writeheader()
        writer.writerow({name: row[name] for name in EPOCH_METRIC_FIELDS})
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
    workers = 0 if smoke else int(
        config["training"].get("data_loader_workers", 0))
    arguments = {
        "dataset": _dataset(config, split, smoke),
        "batch_size": batch_size,
        "shuffle": split == "train",
        "collate_fn": collate_scenes,
        "num_workers": workers,
        "generator": generator,
    }
    if workers:
        arguments.update({
            "prefetch_factor": int(config["training"].get(
                "data_loader_prefetch_factor", 2)),
            "persistent_workers": True,
        })
    return DataLoader(**arguments)


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
        [
            "git", "-C", git_root, "status", "--porcelain", "--",
            "ebjd", "configs", "scripts", "tests", "pyproject.toml",
            "requirements.txt",
        ], check=True, capture_output=True, text=True).stdout.strip())
    if configured_source and configured_source != git_head:
        raise ValueError("experiment.source_commit does not match checked-out HEAD")
    if dirty and not smoke:
        raise ValueError("formal training refuses a dirty EBJD paper source tree")
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
            "training_runtime_optimizations": bool(
                config["training"].get(
                    "training_runtime_optimizations", False)),
            "data_loader_workers": int(
                config["training"].get("data_loader_workers", 0)),
            "data_loader_prefetch_factor": int(
                config["training"].get("data_loader_prefetch_factor", 0)),
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
    research_fork: str | None = None,
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
    differences = config_differences(origin_config, config)
    allowed_paths = {".".join(path) for path in SEMANTIC_CONFIG_EXCLUSIONS}
    research_paths: set[str] = set()
    research_receipt = None
    if research_fork is None:
        if old_semantic_hash != new_semantic_hash:
            raise ValueError("semantic configuration differs from origin checkpoint")
    elif research_fork == "ctf_decoder_energy_v1":
        research_paths = {
            "denoiser.ctf_feedback_basis",
            "training.ctf_feedback_transition_start_update",
            "training.ctf_feedback_transition_updates",
        }
        start = int(config["training"].get(
            "ctf_feedback_transition_start_update", -1))
        count = int(config["training"].get(
            "ctf_feedback_transition_updates", 0))
        if config["denoiser"].get("ctf_feedback_basis") != "decoder_energy":
            raise ValueError("CTF research fork requires decoder_energy basis")
        if start != int(payload["update_index"]):
            raise ValueError(
                "CTF transition must start at the parent optimizer update")
        if count <= 1:
            raise ValueError("CTF transition requires at least two optimizer updates")
        research_receipt = {
            "name": research_fork,
            "allowed_semantic_paths": sorted(research_paths),
            "transition_start_update": start,
            "transition_updates": count,
            "interpretation": "decoder-sensitivity heuristic, not an exact transform",
        }
    else:
        raise ValueError(f"unsupported research fork: {research_fork}")
    allowed_paths |= research_paths
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
    receipt = {
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
        "origin_semantic_config_hash": old_semantic_hash,
        "checked_semantic_diff": differences,
        "runtime_settings": execution_identity["execution"],
        "validated_execution_migration_profile": migration_profile,
        "allowed_changes": sorted(allowed_paths),
        "data_binding_checks": execution_identity["data"],
        "baseline_binding_checks": execution_identity["validation_comparator"],
        "origin_selected_artifacts": _selected_origin_artifacts(checkpoint),
        "origin_tensor_audit": audit,
    }
    if research_receipt is not None:
        receipt["research_semantic_change"] = research_receipt
    return receipt


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
    startup_gate = None
    if args.startup_contract:
        startup_gate = validate_startup_contract(
            args.startup_contract, args.config, config, args)
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
    identity["research_fork"] = args.research_fork or "none"
    identity["stop_after_epoch"] = args.stop_after_epoch
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
                args.resume_from_sha256, config, identity, args.ablation,
                args.research_fork)
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
    if startup_gate is not None:
        if args.resume and Path(args.resume).resolve() != (output / "last.pt").resolve():
            raise ValueError("formal startup contract permits resume only from this run's last.pt")
        _atomic_json(output / "STARTUP_GATE_RECEIPT.json", startup_gate)
    device = torch.device(args.device)
    model = build_model(config, args.ablation).to(device)
    options = trainer_options(config, args.ablation)
    if args.smoke:
        options.epochs = 1
        options.accumulation = 1
        options.rollout_start_epoch = 999
    print(
        f"Training | fold={identity['fold']} | seed={seed} | "
        f"model=FULL/Raw | epochs={options.epochs} | output={output.resolve()}",
        flush=True,
    )
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
        _atomic_json(output / "FITTED_SCALES.json", {
            "goal_scale_m": fitted_scales[0],
            "bridge_scale_m": fitted_scales[1],
            "source": "training_fold_only",
        })
    if args.migration_only:
        print(json.dumps({
            "migration_only": True,
            "output": str(output.resolve()),
            "epoch": start_epoch - 1,
            "update": trainer.update_index,
        }, sort_keys=True), flush=True)
        return
    end_epoch = options.epochs
    if args.stop_after_epoch is not None:
        if not start_epoch <= args.stop_after_epoch <= options.epochs:
            raise ValueError(
                "stop-after-epoch must be within the resumed training interval")
        end_epoch = args.stop_after_epoch
    log_path = output / "train.jsonl"
    epoch_metrics_path = output / "epoch_metrics.csv"
    elapsed_epoch_minutes: list[float] = []
    for epoch in range(start_epoch, end_epoch + 1):
        epoch_started = time.perf_counter()
        epoch_train_loss = 0.0
        epoch_update_count = 0
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
                record = trainer.train_step(pending, epoch)
                record.update({"epoch": epoch, "update": trainer.update_index})
                with record_function("ebjd.jsonl_logging"):
                    with log_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record, sort_keys=True) + "\n")
                epoch_train_loss += record["loss"]
                epoch_update_count += 1
                pending = []
        if pending:
            record = trainer.train_step(pending, epoch)
            record.update({"epoch": epoch, "update": trainer.update_index})
            with record_function("ebjd.jsonl_logging"):
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
            epoch_train_loss += record["loss"]
            epoch_update_count += 1
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
        if epoch == options.epochs:
            trainer.save_checkpoint(
                output / "final.pt", epoch, state, loader_generator,
                {"validation": metrics, "candidate": candidate,
                 "role": "final_epoch_not_automatic_selection"})
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
        if not epoch_update_count:
            raise RuntimeError("epoch completed without an optimizer update")
        epoch_minutes = (time.perf_counter() - epoch_started) / 60.0
        elapsed_epoch_minutes.append(epoch_minutes)
        eta_minutes = (
            sum(elapsed_epoch_minutes) / len(elapsed_epoch_minutes)
            * (options.epochs - epoch))
        train_loss = epoch_train_loss / epoch_update_count
        append_epoch_metrics(epoch_metrics_path, {
            "epoch": epoch,
            "train_loss": train_loss,
            "minADE": metrics["minADE"], "minFDE": metrics["minFDE"],
            "JADE": metrics["JADE"], "JFDE": metrics["JFDE"],
            "epoch_minutes": epoch_minutes,
            "eta_minutes": eta_minutes, "validated": True,
        })
        print(format_epoch_summary(
            epoch, options.epochs, train_loss, metrics, epoch_minutes,
            eta_minutes), flush=True)


if __name__ == "__main__":
    main()
