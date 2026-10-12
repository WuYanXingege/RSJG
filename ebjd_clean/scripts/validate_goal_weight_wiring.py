#!/usr/bin/env python3
"""Finite real-CUDA preflight for the paired endpoint-weight pilot.

Every comparison restores epoch 45 into an isolated trainer.  Diagnostic
optimizer applications are never checkpointed and cannot contaminate either
formal arm.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
import subprocess
import tempfile
import time
from pathlib import Path

import torch

from ebjd.config import build_model, trainer_options, validate_supported_config
from ebjd.data import SceneManifestDataset, collate_scenes
from ebjd.optimizer import (
    clip_global_norm, list_dot, list_norm, project_two_halfspaces,
)
from ebjd.train import (
    audit_restored_state, file_sha256, validate_execution_migration_settings,
)
from ebjd.trainer import EBJDTrainer


PROFILE = {
    "epochs": 100,
    "execution_strategy": "small_group_uncheckpoint_v1",
    "scene_forward_chunk": 1,
    "scene_compaction": True,
    "physical_group_max_scenes": 4,
    "physical_group_max_agent_slots": 96,
    "physical_group_max_padding_ratio": 1.5,
    "rollout_block_checkpointing": False,
    "rollout_uncheckpointed_max_agent_slots": 32,
}
PROJECTION_TOLERANCE = 1e-6
LOSS_FORMULA_TOLERANCE = 1e-5


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def category_norms(
    trainer: EBJDTrainer, gradients: list[torch.Tensor],
) -> dict[str, float]:
    prefixes = {
        "map_unet": "encoder.map_unet.",
        "denoiser": "denoiser.",
        "future_geometry": "denoiser.future_geometry_mlp.",
        "map_injection": "denoiser.map_injection.",
    }
    named = list(trainer.model.named_parameters())
    return {
        label: float(list_norm([
            gradient for (name, _), gradient in zip(named, gradients, strict=True)
            if name.startswith(prefix)
        ]))
        for label, prefix in prefixes.items()
    }


def make_config(payload: dict, goal_weight: float) -> dict:
    config = copy.deepcopy(payload["resolved_config"])
    config["training"].update(PROFILE)
    config["objectives"]["goal_v_weight"] = float(goal_weight)
    validate_supported_config(config)
    validate_execution_migration_settings(config)
    return config


def make_trainer(payload: dict, config: dict, device: torch.device):
    model = build_model(config).to(device)
    options = trainer_options(config)
    trainer = EBJDTrainer(model, options, config, {})
    loader_generator = torch.Generator().manual_seed(1)
    trainer.restore_checkpoint_payload(
        payload, loader_generator=loader_generator, require_identity=False)
    restoration = audit_restored_state(trainer, payload, loader_generator)
    schema_exact = list(model.state_dict()) == list(payload["model"])
    if not schema_exact:
        raise ValueError("model state schema changed while wiring goal weight")
    return trainer, restoration, schema_exact


def release(trainer: EBJDTrainer) -> None:
    del trainer
    gc.collect()
    torch.cuda.empty_cache()


def isolated_snapshot(
    payload: dict, config: dict, batch, device: torch.device,
    do_rollout: bool, rng_seed: int,
) -> dict:
    trainer, restoration, schema_exact = make_trainer(payload, config, device)
    torch.cuda.manual_seed_all(rng_seed)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    total, grad_a, grad_f, logs = trainer._scene_chunked_batch_gradients(
        batch, int(payload["epoch"]) + 1, do_rollout)
    torch.cuda.synchronize(device)
    gradient_seconds = time.perf_counter() - started
    clipped = clip_global_norm(total, trainer.options.gradient_clip_norm)
    candidate, pending = trainer.optimizer.propose(clipped)
    if do_rollout:
        decrement, projection = project_two_halfspaces(candidate, grad_a, grad_f)
    else:
        decrement, projection = candidate, None
    dots = {
        "marginal_ade": float(list_dot(grad_a, decrement)) if do_rollout else 0.0,
        "marginal_fde": float(list_dot(grad_f, decrement)) if do_rollout else 0.0,
    }
    before = [parameter.detach().clone() for parameter in trainer.parameters]
    trainer.optimizer.apply(decrement, pending)
    parameter_change_norm = float(list_norm([
        left.detach() - right for left, right in zip(
            trainer.parameters, before, strict=True)]))
    finite = (
        all(math.isfinite(float(value)) for value in logs.values())
        and all(torch.isfinite(parameter).all() for parameter in trainer.parameters)
        and all(torch.isfinite(value).all() for value in total)
    )
    output = {
        "active_goal_v_weight": trainer.options.goal_v_weight,
        "restoration_checks": restoration,
        "model_state_schema_exact": schema_exact,
        "restored_learning_rates": {
            "new_module": trainer.optimizer.lr,
            "unet": trainer.optimizer.unet_lr,
        },
        "logs": logs,
        "random_draw_sha256": trainer._last_random_draw_sha256,
        "total_gradient_norm": float(list_norm(total)),
        "gradient_category_norms": category_norms(trainer, total),
        "marginal_ade_gradient_norm": float(list_norm(grad_a)),
        "marginal_fde_gradient_norm": float(list_norm(grad_f)),
        "candidate_decrement_norm": float(list_norm(candidate)),
        "applied_decrement_norm": float(list_norm(decrement)),
        "projection_active_constraints": (
            0 if projection is None else projection.active_constraints),
        "applied_projection_dots": dots,
        "parameter_change_norm": parameter_change_norm,
        "finite": bool(finite),
        "gradient_seconds": gradient_seconds,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }
    release(trainer)
    return output


def paired_check(control: dict, candidate: dict, expected_checkpointed: float) -> dict:
    component_names = (
        "diffusion_goal_fine_unweighted",
        "diffusion_bridge_fine_unweighted",
        "diffusion_goal_coarse_unweighted",
        "diffusion_bridge_coarse_unweighted",
        "rollout", "soft_marginal_ade", "soft_marginal_fde",
    )
    component_differences = {
        name: abs(float(control["logs"][name]) - float(candidate["logs"][name]))
        for name in component_names
    }
    expected_increment = (
        float(control["logs"]["diffusion_goal_fine_unweighted"])
        + 0.25 * float(control["logs"]["diffusion_goal_coarse_unweighted"]))
    observed_increment = (
        float(candidate["logs"]["weighted_diffusion"])
        - float(control["logs"]["weighted_diffusion"]))
    required_connections = candidate["gradient_category_norms"]
    passed = (
        control["random_draw_sha256"] == candidate["random_draw_sha256"]
        and max(component_differences.values()) <= LOSS_FORMULA_TOLERANCE
        and abs(observed_increment - expected_increment) <= LOSS_FORMULA_TOLERANCE
        and all(control["restoration_checks"].values())
        and all(candidate["restoration_checks"].values())
        and control["model_state_schema_exact"]
        and candidate["model_state_schema_exact"]
        and all(value > 0 for value in required_connections.values())
        and min(candidate["applied_projection_dots"].values())
        >= -PROJECTION_TOLERANCE
        and control["finite"] and candidate["finite"]
        and candidate["parameter_change_norm"] > 0
        and abs(
            float(candidate["logs"]["rollout_step_checkpointed_group_fraction"])
            - expected_checkpointed) < 1e-12
    )
    return {
        "passed": bool(passed),
        "same_random_draws": (
            control["random_draw_sha256"] == candidate["random_draw_sha256"]),
        "unweighted_component_absolute_differences": component_differences,
        "expected_weighted_diffusion_increment": expected_increment,
        "observed_weighted_diffusion_increment": observed_increment,
        "formula_absolute_error": abs(observed_increment - expected_increment),
        "required_total_gradient_connections": required_connections,
        "candidate_projection_dots": candidate["applied_projection_dots"],
        "expected_rollout_checkpointed_group_fraction": expected_checkpointed,
        "control": control,
        "candidate": candidate,
    }


def select_batches(dataset):
    small_indices = []
    counts = set()
    dense_index = None
    for index in range(len(dataset)):
        item = dataset[index]
        count = int(item["observed"].shape[0])
        if count <= 8 and count not in counts and len(small_indices) < 4:
            small_indices.append(index)
            counts.add(count)
        if count == 57 and dense_index is None:
            dense_index = index
        if len(small_indices) == 4 and dense_index is not None:
            break
    if len(small_indices) != 4 or dense_index is None:
        raise ValueError("real preflight requires four unequal small scenes and N=57")
    return {
        "small": {
            "indices": small_indices,
            "batch": collate_scenes([dataset[index] for index in small_indices]),
            "expected_checkpointed": 0.0,
        },
        "n57": {
            "indices": [dense_index],
            "batch": collate_scenes([dataset[dense_index]]),
            "expected_checkpointed": 1.0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint).resolve()
    if file_sha256(checkpoint) != args.checkpoint_sha256:
        raise ValueError("epoch 45 checkpoint SHA256 mismatch")
    repository = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", "ebjd_clean"],
        cwd=repository, text=True).strip()
    if dirty:
        raise RuntimeError("real GPU wiring preflight requires a clean commit")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("real wiring preflight requires CUDA")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    control_config = make_config(payload, 1.0)
    candidate_config = make_config(payload, 2.0)
    dataset = SceneManifestDataset(control_config["paths"]["train_manifest"])
    batches = select_batches(dataset)
    started = time.perf_counter()
    results = {}
    all_passed = True
    for workload_index, (workload, spec) in enumerate(batches.items()):
        results[workload] = {
            "scene_indices": spec["indices"],
            "agent_counts": spec["batch"].valid.sum(1).tolist(),
        }
        for do_rollout in (False, True):
            seed = 26000 + workload_index * 100 + int(do_rollout)
            control = isolated_snapshot(
                payload, control_config, spec["batch"], device,
                do_rollout, seed)
            candidate = isolated_snapshot(
                payload, candidate_config, spec["batch"], device,
                do_rollout, seed)
            check = paired_check(
                control, candidate,
                spec["expected_checkpointed"] if do_rollout else 0.0)
            results[workload]["rollout_on" if do_rollout else "rollout_off"] = check
            all_passed = all_passed and check["passed"]
    result = {
        "schema": "ebjd-goal-weight-real-cuda-preflight-v1",
        "status": "PASSED" if all_passed else "FAILED",
        "execution_commit": commit,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256_before": args.checkpoint_sha256,
        "checkpoint_sha256_after": file_sha256(checkpoint),
        "origin_epoch": int(payload["epoch"]),
        "origin_update_index": int(payload["update_index"]),
        "profile": validate_execution_migration_settings(control_config),
        "goal_weights": [1.0, 2.0],
        "elapsed_seconds": time.perf_counter() - started,
        "results": results,
        "diagnostic_optimizer_updates_saved": 0,
        "formal_training_started": False,
    }
    atomic_json(Path(args.output).resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not all_passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
