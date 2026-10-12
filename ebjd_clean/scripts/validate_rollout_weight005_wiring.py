#!/usr/bin/env python3
"""Finite real-CUDA preflight for the sole-factor rollout-weight pilot."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import subprocess
import time
from pathlib import Path

import torch

from ebjd.config import build_model, trainer_options, validate_supported_config
from ebjd.data import SceneManifestDataset
from ebjd.optimizer import clip_global_norm, list_dot, list_norm, project_two_halfspaces
from ebjd.train import (
    audit_restored_state, file_sha256, validate_execution_migration_settings,
)
from ebjd.trainer import EBJDTrainer
from validate_goal_weight_wiring import (
    PROFILE, PROJECTION_TOLERANCE, atomic_json, category_norms, select_batches,
)


LOSS_TOLERANCE = 2e-5
# Each arm reconstructs the same BF16 graph independently.  The scalar loss
# identity is checked at the strict tolerance above; CUDA BF16 backward can
# differ slightly across repeated passes, so the vector identity uses a
# separately reported relative-norm tolerance.
GRADIENT_RELATIVE_TOLERANCE = 3e-3


def make_config(payload: dict, rollout_weight: float) -> dict:
    config = copy.deepcopy(payload["resolved_config"])
    config["training"].update(PROFILE)
    config["training"]["rollout_weight_end"] = float(rollout_weight)
    validate_supported_config(config)
    validate_execution_migration_settings(config)
    return config


def make_trainer(payload: dict, config: dict, device: torch.device):
    model = build_model(config).to(device)
    trainer = EBJDTrainer(model, trainer_options(config), config, {})
    loader_generator = torch.Generator().manual_seed(1)
    trainer.restore_checkpoint_payload(
        payload, loader_generator=loader_generator, require_identity=False)
    restoration = audit_restored_state(trainer, payload, loader_generator)
    if list(model.state_dict()) != list(payload["model"]):
        raise ValueError("model state schema changed while wiring rollout weight")
    return trainer, restoration


def release(trainer: EBJDTrainer) -> None:
    del trainer
    gc.collect()
    torch.cuda.empty_cache()


def cpu_gradients(values: list[torch.Tensor]) -> list[torch.Tensor]:
    return [value.detach().to(device="cpu", dtype=torch.float64) for value in values]


def subtract(left, right):
    return [a - b for a, b in zip(left, right, strict=True)]


def scale(values, factor: float):
    return [factor * value for value in values]


def maximum_absolute(values) -> float:
    return max((float(value.abs().max()) for value in values), default=0.0)


def isolated_snapshot(
    payload: dict, config: dict, batch, device: torch.device,
    do_rollout: bool, rng_seed: int,
) -> dict:
    trainer, restoration = make_trainer(payload, config, device)
    torch.cuda.manual_seed_all(rng_seed)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    total, grad_a, grad_f, logs = trainer._scene_chunked_batch_gradients(
        batch, int(payload["epoch"]) + 1, do_rollout)
    torch.cuda.synchronize(device)
    gradient_seconds = time.perf_counter() - started
    saved_gradients = {
        "total": cpu_gradients(total),
        "marginal_ade": cpu_gradients(grad_a),
        "marginal_fde": cpu_gradients(grad_f),
    }
    clipped = clip_global_norm(total, trainer.options.gradient_clip_norm)
    proposal, pending = trainer.optimizer.propose(clipped)
    if do_rollout:
        decrement, projection = project_two_halfspaces(proposal, grad_a, grad_f)
    else:
        decrement, projection = proposal, None
    dots = {
        "marginal_ade": float(list_dot(grad_a, decrement)) if do_rollout else 0.0,
        "marginal_fde": float(list_dot(grad_f, decrement)) if do_rollout else 0.0,
    }
    before = [parameter.detach().clone() for parameter in trainer.parameters]
    trainer.optimizer.apply(decrement, pending)
    parameter_change = float(list_norm([
        after.detach() - old for after, old in zip(
            trainer.parameters, before, strict=True)]))
    finite = (
        all(math.isfinite(float(value)) for value in logs.values())
        and all(torch.isfinite(value).all() for value in total)
        and all(torch.isfinite(parameter).all() for parameter in trainer.parameters)
    )
    result = {
        "active_rollout_weight": trainer.options.rollout_weight_end,
        "rollout_frequency_compensation": trainer.options.rollout_compensation,
        "effective_rollout_coefficient": (
            trainer.options.rollout_compensation * trainer.options.rollout_weight_end),
        "restoration_checks": restoration,
        "logs": logs,
        "random_draw_sha256": trainer._last_random_draw_sha256,
        "total_gradient_norm": float(list_norm(total)),
        "gradient_category_norms": category_norms(trainer, total),
        "projection_active_constraints": (
            0 if projection is None else projection.active_constraints),
        "applied_projection_dots": dots,
        "parameter_change_norm": parameter_change,
        "finite": bool(finite),
        "gradient_seconds": gradient_seconds,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "_gradients": saved_gradients,
    }
    release(trainer)
    return result


def paired_check(control: dict, candidate: dict, zero: dict | None, do_rollout: bool,
                 expected_checkpointed: float) -> dict:
    control_grad = control["_gradients"]
    candidate_grad = candidate["_gradients"]
    total_delta = subtract(candidate_grad["total"], control_grad["total"])
    protected_ade_delta = subtract(
        candidate_grad["marginal_ade"], control_grad["marginal_ade"])
    protected_fde_delta = subtract(
        candidate_grad["marginal_fde"], control_grad["marginal_fde"])
    expected_delta = [torch.zeros_like(value) for value in total_delta]
    rollout_gradient_norm = 0.0
    if do_rollout:
        if zero is None:
            raise AssertionError("rollout check requires a zero-weight diagnostic arm")
        zero_total = zero["_gradients"]["total"]
        rollout_gradient = scale(subtract(control_grad["total"], zero_total), 1.0 / 0.8)
        expected_delta = scale(rollout_gradient, -0.60)
        rollout_gradient_norm = float(list_norm(rollout_gradient))
    gradient_error = subtract(total_delta, expected_delta)
    gradient_error_norm = float(list_norm(gradient_error))
    expected_norm = float(list_norm(expected_delta))
    relative_error = gradient_error_norm / max(expected_norm, 1e-12)
    observed_loss_delta = (
        float(candidate["logs"]["loss"]) - float(control["logs"]["loss"]))
    expected_loss_delta = (
        -0.60 * float(control["logs"]["rollout"]) if do_rollout else 0.0)
    hashes = [control["random_draw_sha256"], candidate["random_draw_sha256"]]
    if zero is not None:
        hashes.append(zero["random_draw_sha256"])
    passed = (
        len(set(hashes)) == 1
        and abs(observed_loss_delta - expected_loss_delta) <= LOSS_TOLERANCE
        and (relative_error <= GRADIENT_RELATIVE_TOLERANCE if do_rollout
             else gradient_error_norm <= 1e-10)
        and maximum_absolute(protected_ade_delta) <= 1e-10
        and maximum_absolute(protected_fde_delta) <= 1e-10
        and all(control["restoration_checks"].values())
        and all(candidate["restoration_checks"].values())
        and control["finite"] and candidate["finite"]
        and candidate["parameter_change_norm"] > 0
        and all(value > 0 for value in candidate["gradient_category_norms"].values())
        and (not do_rollout or min(candidate["applied_projection_dots"].values())
             >= -PROJECTION_TOLERANCE)
        and abs(float(candidate["logs"][
            "rollout_step_checkpointed_group_fraction"])
            - expected_checkpointed) <= 1e-12
    )
    for snapshot in (control, candidate, zero):
        if snapshot is not None:
            snapshot.pop("_gradients", None)
    return {
        "passed": bool(passed),
        "same_random_draws": len(set(hashes)) == 1,
        "expected_loss_delta": expected_loss_delta,
        "observed_loss_delta": observed_loss_delta,
        "loss_formula_absolute_error": abs(observed_loss_delta - expected_loss_delta),
        "rollout_gradient_norm": rollout_gradient_norm,
        "observed_total_gradient_delta_norm": float(list_norm(total_delta)),
        "expected_total_gradient_delta_norm": expected_norm,
        "gradient_formula_error_norm": gradient_error_norm,
        "gradient_formula_relative_error": relative_error,
        "protected_marginal_ade_gradient_max_abs_delta": maximum_absolute(
            protected_ade_delta),
        "protected_marginal_fde_gradient_max_abs_delta": maximum_absolute(
            protected_fde_delta),
        "expected_rollout_checkpointed_group_fraction": expected_checkpointed,
        "control": control, "candidate": candidate,
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
        raise ValueError("epoch45 checkpoint SHA256 mismatch")
    repository = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", "ebjd_clean"],
        cwd=repository, text=True).strip()
    if dirty:
        raise RuntimeError("real CUDA preflight requires a clean production commit")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("real wiring preflight requires CUDA")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    configs = {
        "control": make_config(payload, 0.20),
        "candidate": make_config(payload, 0.05),
        "zero_diagnostic": make_config(payload, 0.0),
    }
    dataset = SceneManifestDataset(configs["control"]["paths"]["train_manifest"])
    workloads = select_batches(dataset)
    started = time.perf_counter()
    results = {}
    all_passed = True
    for workload_index, (name, spec) in enumerate(workloads.items()):
        results[name] = {
            "scene_indices": spec["indices"],
            "agent_counts": spec["batch"].valid.sum(1).tolist(),
        }
        for do_rollout in (False, True):
            seed = 27000 + workload_index * 100 + int(do_rollout)
            control = isolated_snapshot(
                payload, configs["control"], spec["batch"], device,
                do_rollout, seed)
            candidate = isolated_snapshot(
                payload, configs["candidate"], spec["batch"], device,
                do_rollout, seed)
            zero = (isolated_snapshot(
                payload, configs["zero_diagnostic"], spec["batch"], device,
                True, seed) if do_rollout else None)
            check = paired_check(
                control, candidate, zero, do_rollout,
                spec["expected_checkpointed"] if do_rollout else 0.0)
            results[name]["rollout_on" if do_rollout else "rollout_off"] = check
            all_passed &= check["passed"]
    result = {
        "schema": "ebjd-rollout-weight005-real-cuda-preflight-v1",
        "status": "PASSED" if all_passed else "FAILED",
        "execution_commit": commit,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256_before": args.checkpoint_sha256,
        "checkpoint_sha256_after": file_sha256(checkpoint),
        "origin_epoch": int(payload["epoch"]),
        "origin_update_index": int(payload["update_index"]),
        "profile": validate_execution_migration_settings(configs["candidate"]),
        "formal_weights": {"control": 0.20, "candidate": 0.05},
        "formal_effective_coefficients": {"control": 0.80, "candidate": 0.20},
        "zero_weight_arm_role": "diagnostic gradient decomposition only",
        "tolerances": {
            "loss_formula_absolute": LOSS_TOLERANCE,
            "bf16_repeated_backward_gradient_relative_norm": (
                GRADIENT_RELATIVE_TOLERANCE),
            "projection_dot": PROJECTION_TOLERANCE,
        },
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
