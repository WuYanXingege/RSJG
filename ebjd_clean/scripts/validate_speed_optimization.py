#!/usr/bin/env python3
"""Paired CUDA validation for EBJD padding/checkpoint/grouping optimization.

Every mode restores a supplied epoch-boundary checkpoint into a fresh trainer.
The script never writes a training checkpoint and no diagnostic-updated object
is reused for formal training.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
import statistics
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile
from torch.utils.data import DataLoader

from ebjd.config import build_model, trainer_options
from ebjd.data import SceneManifestDataset, augment_batch, collate_scenes
from ebjd.optimizer import (
    clip_global_norm, list_dot, list_norm, project_two_halfspaces,
)
from ebjd.sampling import differentiable_sample
from ebjd.trainer import EBJDTrainer, contiguous_scene_groups
from validate_oom_recovery import (
    DIRECTION_COSINE_MINIMUM, FAILURE_BATCH_INDICES,
    GRADIENT_RELATIVE_TOLERANCE, LOSS_ABS_TOLERANCE,
    MINIMUM_RESERVED_MARGIN_BYTES, OPTIMIZER_DIRECTION_COSINE_MINIMUM,
    OPTIMIZER_RELATIVE_TOLERANCE, PROJECTION_TOLERANCE,
    atomic_json, finite_tree, gradient_comparison, highest_agent_indices, sha256,
)


FINAL_STRATEGY = "compact_grouped_v1"
# Fixed before the new CUDA comparison.  These FP32 gates are four times the
# largest raw-gradient discrepancy (1.2406e-4) localized in the prior certified
# scene-staging audit, while remaining far below the configured BF16 bounds.
FP32_LOSS_ABS_TOLERANCE = 1e-4
FP32_RELATIVE_TOLERANCE = 5e-4
FP32_TRAJECTORY_RELATIVE_TOLERANCE = 5e-4
SEMANTIC_LOG_NAMES = (
    "loss", "diffusion", "geometry", "map", "rollout",
    "soft_marginal_ade", "soft_marginal_fde", "minimum_time", "maximum_time",
)
STRATEGIES = {
    "reference": {
        "scene_compaction": False,
        "physical_group_max_scenes": 1,
        "physical_group_max_agent_slots": 0,
        "physical_group_max_padding_ratio": 1.0,
        "rollout_block_checkpointing": True,
    },
    "compaction_only": {
        "scene_compaction": True,
        "physical_group_max_scenes": 1,
        "physical_group_max_agent_slots": 96,
        "physical_group_max_padding_ratio": 1.5,
        "rollout_block_checkpointing": True,
    },
    "step_checkpoint_only": {
        "scene_compaction": True,
        "physical_group_max_scenes": 1,
        "physical_group_max_agent_slots": 96,
        "physical_group_max_padding_ratio": 1.5,
        "rollout_block_checkpointing": False,
    },
    "group2": {
        "scene_compaction": True,
        "physical_group_max_scenes": 2,
        "physical_group_max_agent_slots": 96,
        "physical_group_max_padding_ratio": 1.5,
        "rollout_block_checkpointing": False,
    },
    FINAL_STRATEGY: {
        "scene_compaction": True,
        "physical_group_max_scenes": 4,
        "physical_group_max_agent_slots": 96,
        "physical_group_max_padding_ratio": 1.5,
        "rollout_block_checkpointing": False,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--mode", choices=("numeric", "benchmark", "profile", "continuous", "all"),
        default="all")
    parser.add_argument("--numeric-precision", choices=("fp32", "configured"),
                        default="configured")
    parser.add_argument("--benchmark-repeats", type=int, default=1)
    parser.add_argument("--profile-strategy", choices=tuple(STRATEGIES),
                        default=FINAL_STRATEGY)
    parser.add_argument(
        "--profile-workload", choices=("small", "mixed", "n57"),
        default="mixed")
    parser.add_argument("--profile-rollout", action="store_true")
    parser.add_argument("--trace", default=None)
    parser.add_argument("--continuous-updates", type=int, default=48)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def make_trainer(
    payload: dict, device: torch.device, strategy: str,
    precision: str | None = None,
) -> EBJDTrainer:
    config = copy.deepcopy(payload["resolved_config"])
    if precision is not None:
        config["training"]["mixed_precision"] = precision
    model = build_model(config).to(device)
    options = trainer_options(config)
    options.scene_forward_chunk = 1
    for name, value in STRATEGIES[strategy].items():
        setattr(options, name, value)
    trainer = EBJDTrainer(model, options)
    trainer.restore_checkpoint_payload(payload, require_identity=False)
    model.train()
    return trainer


def release(trainer: EBJDTrainer | None) -> None:
    if trainer is not None:
        del trainer
    gc.collect()
    torch.cuda.empty_cache()


def cpu_tensors(values: list[torch.Tensor]) -> list[torch.Tensor]:
    return [value.detach().float().cpu() for value in values]


def category_norms(
    trainer: EBJDTrainer, gradients: list[torch.Tensor],
) -> dict[str, float]:
    categories = {
        "map_unet": "encoder.map_unet.",
        "denoiser": "denoiser.",
        "future_geometry": "denoiser.future_geometry_mlp.",
        "map_sampling_injection": "denoiser.map_injection.",
    }
    named = list(trainer.model.named_parameters())
    result = {}
    for label, prefix in categories.items():
        selected = [
            gradient for (name, _), gradient in zip(named, gradients, strict=True)
            if name.startswith(prefix)
        ]
        result[label] = float(list_norm(selected)) if selected else 0.0
    return result


def gradient_snapshot(
    payload: dict, dataset, device: torch.device, strategy: str,
    batch, do_rollout: bool, precision: str, rng_seed: int,
) -> dict:
    trainer = make_trainer(payload, device, strategy, precision)
    torch.cuda.manual_seed_all(rng_seed)
    starting_rng = torch.cuda.get_rng_state().clone()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    total, grad_a, grad_f, logs = trainer._scene_chunked_batch_gradients(
        batch, int(payload["epoch"]) + 1, do_rollout)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    ending_rng = torch.cuda.get_rng_state().cpu()
    clipped = clip_global_norm(total, trainer.options.gradient_clip_norm)
    candidate, pending = trainer.optimizer.propose(clipped)
    if do_rollout and trainer.options.use_actual_step_constraint:
        decrement, projection = project_two_halfspaces(candidate, grad_a, grad_f)
    else:
        decrement, projection = candidate, None
    category = {
        "total": category_norms(trainer, total),
        "marginal_ade": category_norms(trainer, grad_a),
        "marginal_fde": category_norms(trainer, grad_f),
    }
    projection_dots = {
        "marginal_ade": float(list_dot(grad_a, decrement)),
        "marginal_fde": float(list_dot(grad_f, decrement)),
    }
    trainer.optimizer.apply(decrement, pending)
    output = {
        "logs": logs,
        "total": cpu_tensors(total),
        "marginal_ade": cpu_tensors(grad_a),
        "marginal_fde": cpu_tensors(grad_f),
        "candidate": cpu_tensors(candidate),
        "decrement": cpu_tensors(decrement),
        "parameters_after": cpu_tensors(trainer.parameters),
        "optimizer_moment_after": cpu_tensors([
            trainer.optimizer.state[id(parameter)]["moment"]
            for parameter in trainer.parameters]),
        "optimizer_variance_after": cpu_tensors([
            trainer.optimizer.state[id(parameter)]["variance"]
            for parameter in trainer.parameters]),
        "optimizer_steps_after": [
            int(trainer.optimizer.state[id(parameter)]["step"])
            for parameter in trainer.parameters],
        "category_gradient_norms": category,
        "projection_dots": projection_dots,
        "projection_active": 0 if projection is None else projection.active_constraints,
        "starting_rng": starting_rng.cpu(),
        "ending_rng": ending_rng,
        "elapsed_seconds": elapsed,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    release(trainer)
    return output


@torch.no_grad()
def trajectory_snapshot(
    payload: dict, device: torch.device, strategy: str, batch,
    precision: str, rng_seed: int,
) -> list[torch.Tensor]:
    trainer = make_trainer(payload, device, strategy, precision)
    torch.cuda.manual_seed_all(rng_seed)
    draws = trainer._prepare_batch_draws(batch, True)
    settings = STRATEGIES[strategy]
    if settings["scene_compaction"]:
        groups = contiguous_scene_groups(
            batch.valid, settings["physical_group_max_scenes"],
            settings["physical_group_max_agent_slots"],
            settings["physical_group_max_padding_ratio"])
    else:
        groups = [(index, index + 1) for index in range(batch.observed.shape[0])]
    trajectories: list[torch.Tensor] = []
    for start, stop in groups:
        if settings["scene_compaction"]:
            physical = batch.compact_scenes(start, stop).to(device)
            physical_draws = draws.compact_scenes(batch.valid, start, stop)
        else:
            physical = batch.slice_scenes(start, stop).to(device)
            physical_draws = draws.slice_scenes(start, stop)
        autocast_enabled = (
            precision != "fp32" and torch.cuda.is_bf16_supported())
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
            context = trainer.model.encode_context(
                physical.observed, physical.semantic_maps, physical.valid)
            free, _, _ = differentiable_sample(
                trainer.model, context, physical_draws.rollout_initial,
                trainer.options.rollout_steps,
                checkpoint_steps=False,
                checkpoint_blocks=settings["rollout_block_checkpointing"])
        for local in range(stop - start):
            count = int(physical.valid[local].sum())
            trajectories.append(free[local, :, :count].float().cpu())
    release(trainer)
    return trajectories


def tensor_list_comparison(reference, candidate) -> dict:
    difference = [
        left - right for left, right in zip(reference, candidate, strict=True)]
    reference_norm = float(list_norm(reference))
    difference_norm = float(list_norm(difference))
    return {
        "reference_norm": reference_norm,
        "absolute_difference_norm": difference_norm,
        "relative_difference_norm": difference_norm / max(reference_norm, 1e-30),
        "maximum_absolute_element_difference": max(
            float(value.abs().max()) for value in difference),
    }


def compare_snapshots(reference: dict, candidate: dict, precision: str) -> dict:
    names = (
        "total", "marginal_ade", "marginal_fde", "candidate", "decrement",
        "parameters_after", "optimizer_moment_after", "optimizer_variance_after",
    )
    comparisons = {
        name: gradient_comparison(reference[name], candidate[name])
        for name in names
    }
    log_differences = {
        name: abs(float(reference["logs"][name]) - float(candidate["logs"][name]))
        for name in SEMANTIC_LOG_NAMES
    }
    gradient_tolerance = (
        FP32_RELATIVE_TOLERANCE
        if precision == "fp32" else GRADIENT_RELATIVE_TOLERANCE)
    optimizer_tolerance = (
        FP32_RELATIVE_TOLERANCE
        if precision == "fp32" else OPTIMIZER_RELATIVE_TOLERANCE)
    gradient_names = ("total", "marginal_ade", "marginal_fde")
    optimizer_names = (
        "candidate", "decrement", "parameters_after",
        "optimizer_moment_after", "optimizer_variance_after")
    category = candidate["category_gradient_norms"]
    rollout_enabled = float(candidate["logs"]["rollout"]) != 0.0
    required_connections = {
        "base_total_to_trainable_unet": category["total"]["map_unet"],
        "base_total_to_denoiser": category["total"]["denoiser"],
        "base_total_to_future_geometry": category["total"]["future_geometry"],
    }
    if rollout_enabled:
        required_connections.update({
            "rollout_ade_to_trainable_unet": category["marginal_ade"]["map_unet"],
            "rollout_fde_to_trainable_unet": category["marginal_fde"]["map_unet"],
            "rollout_ade_to_future_geometry": category[
                "marginal_ade"]["future_geometry"],
            "rollout_fde_to_map_sampling": category[
                "marginal_fde"]["map_sampling_injection"],
        })
    connections_present = all(value > 0.0 for value in required_connections.values())
    passed = (
        max(log_differences.values()) <= (
            FP32_LOSS_ABS_TOLERANCE
            if precision == "fp32" else LOSS_ABS_TOLERANCE)
        and all(
            comparisons[name]["relative_difference_norm"] <= gradient_tolerance
            and comparisons[name]["direction_cosine"] >= DIRECTION_COSINE_MINIMUM
            for name in gradient_names
            if comparisons[name]["reference_norm"] > 1e-12)
        and all(
            comparisons[name]["relative_difference_norm"] <= optimizer_tolerance
            and comparisons[name]["direction_cosine"] >= (
                0.999999 if precision == "fp32"
                else OPTIMIZER_DIRECTION_COSINE_MINIMUM)
            for name in optimizer_names
            if comparisons[name]["reference_norm"] > 1e-12)
        and min(candidate["projection_dots"].values()) >= -PROJECTION_TOLERANCE
        and reference["optimizer_steps_after"] == candidate["optimizer_steps_after"]
        and torch.equal(reference["ending_rng"], candidate["ending_rng"])
        and connections_present
    )
    return {
        "passed": passed,
        "log_absolute_differences": log_differences,
        "tensor_differences": comparisons,
        "candidate_projection_dots": candidate["projection_dots"],
        "category_gradient_norms": candidate["category_gradient_norms"],
        "required_gradient_connections": required_connections,
        "required_gradient_connections_present": connections_present,
        "cuda_rng_exact": torch.equal(
            reference["ending_rng"], candidate["ending_rng"]),
        "reference_resources": {
            key: reference[key] for key in (
                "elapsed_seconds", "peak_allocated_bytes", "peak_reserved_bytes")},
        "candidate_resources": {
            key: candidate[key] for key in (
                "elapsed_seconds", "peak_allocated_bytes", "peak_reserved_bytes")},
        "fixed_tolerances": {
            "loss_absolute": (
                FP32_LOSS_ABS_TOLERANCE
                if precision == "fp32" else LOSS_ABS_TOLERANCE),
            "gradient_relative": gradient_tolerance,
            "optimizer_relative": optimizer_tolerance,
        },
    }


def small_unequal_batch(dataset):
    indices = []
    counts = set()
    for index in range(len(dataset)):
        item = dataset[index]
        count = int(item["observed"].shape[0])
        if count <= 8 and count not in counts:
            indices.append(index)
            counts.add(count)
        if len(indices) == 4:
            break
    if len(indices) != 4:
        raise ValueError("four unequal small-agent scenes were not found")
    return indices, collate_scenes([dataset[index] for index in indices])


def numerical_validation(payload: dict, dataset, device, precision: str) -> dict:
    indices, batch = small_unequal_batch(dataset)
    result = {
        "precision": precision, "scene_indices": indices,
        "agent_counts": batch.valid.sum(1).tolist(),
    }
    all_passed = True
    for do_rollout in (False, True):
        reference = gradient_snapshot(
            payload, dataset, device, "reference", batch, do_rollout,
            precision, 7319 + int(do_rollout))
        repeated_reference = gradient_snapshot(
            payload, dataset, device, "reference", batch, do_rollout,
            precision, 7319 + int(do_rollout))
        candidate = gradient_snapshot(
            payload, dataset, device, FINAL_STRATEGY, batch, do_rollout,
            precision, 7319 + int(do_rollout))
        comparison = compare_snapshots(reference, candidate, precision)
        comparison["reference_repeat_differences"] = {
            name: gradient_comparison(reference[name], repeated_reference[name])
            for name in (
                "total", "marginal_ade", "marginal_fde", "candidate",
                "decrement", "parameters_after", "optimizer_moment_after",
                "optimizer_variance_after")
        }
        repeat_logs = {
            name: abs(
                float(reference["logs"][name])
                - float(repeated_reference["logs"][name]))
            for name in SEMANTIC_LOG_NAMES
        }
        gradient_tolerance = (
            FP32_RELATIVE_TOLERANCE
            if precision == "fp32" else GRADIENT_RELATIVE_TOLERANCE)
        optimizer_tolerance = (
            FP32_RELATIVE_TOLERANCE
            if precision == "fp32" else OPTIMIZER_RELATIVE_TOLERANCE)
        repeat_passed = (
            torch.equal(reference["ending_rng"], repeated_reference["ending_rng"])
            and max(repeat_logs.values()) <= (
                FP32_LOSS_ABS_TOLERANCE
                if precision == "fp32" else LOSS_ABS_TOLERANCE)
            and all(
                comparison["reference_repeat_differences"][name][
                    "relative_difference_norm"] <= gradient_tolerance
                for name in ("total", "marginal_ade", "marginal_fde")
                if comparison["reference_repeat_differences"][name][
                    "reference_norm"] > 1e-12)
            and all(
                comparison["reference_repeat_differences"][name][
                    "relative_difference_norm"] <= optimizer_tolerance
                for name in (
                    "candidate", "decrement", "parameters_after",
                    "optimizer_moment_after", "optimizer_variance_after")
                if comparison["reference_repeat_differences"][name][
                    "reference_norm"] > 1e-12))
        comparison["reference_repeat_semantic_log_differences"] = repeat_logs
        comparison["reference_repeat_rng_exact"] = torch.equal(
            reference["ending_rng"], repeated_reference["ending_rng"])
        comparison["reference_repeat_passed"] = repeat_passed
        comparison["passed"] = comparison["passed"] and repeat_passed
        if do_rollout:
            reference_trajectory = trajectory_snapshot(
                payload, device, "reference", batch, precision, 9143)
            candidate_trajectory = trajectory_snapshot(
                payload, device, FINAL_STRATEGY, batch, precision, 9143)
            trajectory = tensor_list_comparison(
                reference_trajectory, candidate_trajectory)
            trajectory_tolerance = (
                FP32_TRAJECTORY_RELATIVE_TOLERANCE
                if precision == "fp32" else 2e-2)
            trajectory["tolerance"] = trajectory_tolerance
            trajectory["passed"] = (
                trajectory["relative_difference_norm"] <= trajectory_tolerance)
            comparison["valid_rollout_trajectory_difference"] = trajectory
            comparison["passed"] = comparison["passed"] and trajectory["passed"]
        result["rollout_on" if do_rollout else "rollout_off"] = comparison
        all_passed = all_passed and comparison["passed"]
        del reference, repeated_reference, candidate
        gc.collect()
    result["passed"] = all_passed
    return result


def failure_batches(dataset):
    return [
        collate_scenes([dataset[index] for index in indices])
        for indices in FAILURE_BATCH_INDICES]


def measured_update(
    trainer: EBJDTrainer, batches: list, epoch: int, augmentation_seconds: float = 0.0,
) -> dict:
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    free_before, total = torch.cuda.mem_get_info()
    reserved_before = torch.cuda.memory_reserved()
    started = time.perf_counter()
    record = trainer.train_step(batches, epoch)
    torch.cuda.synchronize()
    update_elapsed = time.perf_counter() - started
    free_after, _ = torch.cuda.mem_get_info()
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    non_pytorch_before = max(total - free_before - reserved_before, 0)
    conservative_free_at_peak = max(total - non_pytorch_before - peak_reserved, 0)
    return {
        "record": record,
        "update_elapsed_seconds": update_elapsed,
        "augmentation_plus_update_seconds": augmentation_seconds + update_elapsed,
        "driver_free_before_bytes": free_before,
        "driver_free_after_bytes": free_after,
        "estimated_driver_free_at_pytorch_reserved_peak_bytes": conservative_free_at_peak,
        "total_device_bytes": total,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "finite": finite_tree(record),
    }


def prepare_measured_trainer(payload, device, strategy, do_rollout, seed):
    trainer = make_trainer(payload, device, strategy)
    modulo = trainer.update_index % trainer.options.rollout_every
    if do_rollout:
        trainer.update_index += (-modulo) % trainer.options.rollout_every
    elif modulo == 0:
        trainer.update_index += 1
    torch.cuda.manual_seed_all(seed)
    return trainer


def warmup_strategy(payload, dataset, device, strategy):
    _, small = small_unequal_batch(dataset)
    for do_rollout in (False, True):
        trainer = prepare_measured_trainer(
            payload, device, strategy, do_rollout,
            11000 + int(do_rollout))
        measured_update(trainer, [small], int(payload["epoch"]) + 1)
        release(trainer)


def benchmark(payload: dict, dataset, device, repeats: int) -> dict:
    if not 1 <= repeats <= 5:
        raise ValueError("benchmark repeats must be between one and five")
    batches = failure_batches(dataset)
    result = {
        "scene_indices": [list(item) for item in FAILURE_BATCH_INDICES],
        "agent_counts": [batch.valid.sum(1).tolist() for batch in batches],
        "note": "update timing includes physical H2D, full objectives, three VJPs, AdamW, QP and apply; loader I/O and JSONL logging are excluded",
        "strategies": {},
    }
    for strategy in STRATEGIES:
        warmup_strategy(payload, dataset, device, strategy)
        strategy_result = {"settings": STRATEGIES[strategy]}
        for do_rollout in (False, True):
            records = []
            for repeat in range(repeats):
                trainer = prepare_measured_trainer(
                    payload, device, strategy, do_rollout,
                    12000 + 100 * int(do_rollout) + repeat)
                measured = measured_update(
                    trainer, batches, int(payload["epoch"]) + 1)
                records.append(measured)
                release(trainer)
            latencies = [item["update_elapsed_seconds"] for item in records]
            strategy_result["rollout" if do_rollout else "normal"] = {
                "records": records,
                "p50_seconds": statistics.median(latencies),
                "p90_seconds": sorted(latencies)[
                    min(len(latencies) - 1, math.ceil(0.9 * len(latencies)) - 1)],
                "maximum_peak_allocated_bytes": max(
                    item["peak_allocated_bytes"] for item in records),
                "maximum_peak_reserved_bytes": max(
                    item["peak_reserved_bytes"] for item in records),
                "minimum_conservative_driver_free_bytes": min(
                    item["estimated_driver_free_at_pytorch_reserved_peak_bytes"]
                    for item in records),
                "passed": all(item["finite"] for item in records),
            }
        normal = strategy_result["normal"]["p50_seconds"]
        rollout = strategy_result["rollout"]["p50_seconds"]
        strategy_result["weighted_p50_seconds"] = 0.75 * normal + 0.25 * rollout
        result["strategies"][strategy] = strategy_result
    reference = result["strategies"]["reference"]["weighted_p50_seconds"]
    final = result["strategies"][FINAL_STRATEGY]["weighted_p50_seconds"]
    result["final_speedup_over_reference"] = reference / final
    result["passed"] = (
        all(
            item["normal"]["passed"] and item["rollout"]["passed"]
            for item in result["strategies"].values())
        and result["strategies"][FINAL_STRATEGY]["rollout"][
            "minimum_conservative_driver_free_bytes"] >= MINIMUM_RESERVED_MARGIN_BYTES
        and result["final_speedup_over_reference"] > 1.0)
    return result


def profile_update(
    payload, dataset, device, strategy, workload, do_rollout, trace_path,
):
    if workload == "small":
        _, selected = small_unequal_batch(dataset)
    elif workload == "mixed":
        selected = failure_batches(dataset)[0]
    else:
        selected = collate_scenes([dataset[3302]])
    batches = [selected]
    trainer = prepare_measured_trainer(payload, device, strategy, do_rollout, 17771)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        # This is intentionally lightweight: shape/stack/memory retention is
        # excluded because the full differentiable rollout emits enough events
        # to distort memory use and make key aggregation itself expensive.
        record_shapes=False, profile_memory=False, with_stack=False,
    ) as profiler:
        measured = measured_update(
            trainer, batches, int(payload["epoch"]) + 1)
    if trace_path:
        Path(trace_path).parent.mkdir(parents=True, exist_ok=True)
        profiler.export_chrome_trace(trace_path)
    rows = []
    for event in profiler.key_averages(group_by_input_shape=False):
        rows.append({
            "key": event.key,
            "count": int(event.count),
            "cpu_time_total_us": float(event.cpu_time_total),
            "cuda_time_total_us": float(getattr(
                event, "device_time_total", getattr(event, "cuda_time_total", 0.0))),
            "self_cpu_time_total_us": float(event.self_cpu_time_total),
            "self_cuda_time_total_us": float(getattr(
                event, "self_device_time_total", getattr(
                    event, "self_cuda_time_total", 0.0))),
        })
    rows.sort(key=lambda item: item["cuda_time_total_us"], reverse=True)
    marker_rows = [item for item in rows if item["key"].startswith("ebjd.")]
    checkpoint_rows = [
        item for item in rows
        if "checkpoint" in item["key"].lower() or "recompute" in item["key"].lower()
    ]
    release(trainer)
    return {
        "strategy": strategy, "settings": STRATEGIES[strategy],
        "workload": workload,
        "agent_counts": selected.valid.sum(1).tolist(),
        "logical_microbatches": 1,
        "rollout": do_rollout, "measured": measured,
        "trace_path": str(Path(trace_path).resolve()) if trace_path else None,
        "markers": marker_rows,
        "checkpoint_related_events": checkpoint_rows,
        "top_cuda_events": rows[:50],
        "warning": "lightweight profile latency is diagnostic only and is excluded from formal peak/timing acceptance; shape/stack/memory retention is disabled",
        "passed": measured["finite"],
    }


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)]


def continuous_validation(payload, dataset, manifest_path, device, updates):
    if not 32 <= updates <= 64:
        raise ValueError("continuous validation requires 32--64 updates")
    trainer = make_trainer(payload, device, FINAL_STRATEGY)
    generator = torch.Generator()
    generator.set_state(payload["rng"]["loader_generator"])
    loader = DataLoader(
        dataset, batch_size=4, shuffle=True, collate_fn=collate_scenes,
        num_workers=0, generator=generator)
    iterator = iter(loader)
    failure = failure_batches(dataset)
    dense_indices = highest_agent_indices(dataset, manifest_path)
    dense = collate_scenes([dataset[index] for index in dense_indices])
    next_rollout = (-trainer.update_index) % trainer.options.rollout_every
    failure_ordinal = next_rollout + 1
    dense_ordinal = failure_ordinal + trainer.options.rollout_every
    tail_ordinal = updates
    records = []
    for ordinal in range(1, updates + 1):
        started = time.perf_counter()
        if ordinal == failure_ordinal:
            raw = failure
            kind = "reconstructed_original_failure"
        elif ordinal == dense_ordinal:
            raw = [dense, dense, dense, dense]
            kind = "dense_n57_b4xacc4"
        elif ordinal == tail_ordinal:
            raw = []
            for logical_size in (4, 4, 1):
                items = []
                while len(items) < logical_size:
                    try:
                        loaded = next(iterator)
                    except StopIteration:
                        iterator = iter(loader)
                        loaded = next(iterator)
                    # Split a loader B4 only for the B1 tail without modifying
                    # scene order or identities.
                    if logical_size == 1:
                        loaded = loaded.slice_scenes(0, 1)
                    items.append(loaded)
                    break
                raw.append(items[0])
            kind = "tail_k3_b4_b4_b1"
        else:
            raw = []
            while len(raw) < 4:
                try:
                    raw.append(next(iterator))
                except StopIteration:
                    iterator = iter(loader)
            kind = "loader_mixed"
        augmented = [augment_batch(batch, generator) for batch in raw]
        augmentation_elapsed = time.perf_counter() - started
        measured = measured_update(
            trainer, augmented, int(payload["epoch"]) + 1,
            augmentation_seconds=augmentation_elapsed)
        measured.update({
            "ordinal": ordinal, "kind": kind,
            "scene_ids": [batch.scene_ids for batch in raw],
            "padded_N": [int(batch.observed.shape[1]) for batch in raw],
            "valid_agents": [int(batch.valid.sum()) for batch in raw],
        })
        records.append(measured)
    normal = [
        item["update_elapsed_seconds"] for item in records
        if item["record"]["rollout_update"] == 0]
    rollout = [
        item["update_elapsed_seconds"] for item in records
        if item["record"]["rollout_update"] == 1]
    e2e = [item["augmentation_plus_update_seconds"] for item in records]
    minimum_free = min(
        item["estimated_driver_free_at_pytorch_reserved_peak_bytes"]
        for item in records)
    result = {
        "strategy": FINAL_STRATEGY, "settings": STRATEGIES[FINAL_STRATEGY],
        "updates": len(records), "rollout_updates": len(rollout),
        "failure_ordinal": failure_ordinal, "dense_ordinal": dense_ordinal,
        "tail_ordinal": tail_ordinal,
        "normal_p50_seconds": statistics.median(normal),
        "normal_p90_seconds": percentile(normal, 0.9),
        "rollout_p50_seconds": statistics.median(rollout),
        "rollout_p90_seconds": percentile(rollout, 0.9),
        "actual_frequency_mean_update_seconds": statistics.mean(
            item["update_elapsed_seconds"] for item in records),
        "augmentation_plus_update_p50_seconds": statistics.median(e2e),
        "augmentation_plus_update_p90_seconds": percentile(e2e, 0.9),
        "maximum_peak_allocated_bytes": max(
            item["peak_allocated_bytes"] for item in records),
        "maximum_peak_reserved_bytes": max(
            item["peak_reserved_bytes"] for item in records),
        "minimum_conservative_driver_free_bytes": minimum_free,
        "records": records,
    }
    result["passed"] = (
        len(records) == updates and all(item["finite"] for item in records)
        and records[failure_ordinal - 1]["record"]["rollout_update"] == 1
        and records[dense_ordinal - 1]["record"]["rollout_update"] == 1
        and records[tail_ordinal - 1]["kind"] == "tail_k3_b4_b4_b1"
        and minimum_free >= MINIMUM_RESERVED_MARGIN_BYTES)
    release(trainer)
    return result


def base_output(args, checkpoint, payload, device):
    return {
        "schema": "ebjd-speed-optimization-gpu-validation-v1",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_epoch": int(payload["epoch"]),
        "checkpoint_update_index": int(payload["update_index"]),
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(device),
        "allocator_environment": {
            "PYTORCH_CUDA_ALLOC_CONF": os.environ.get(
                "PYTORCH_CUDA_ALLOC_CONF", "")},
        "parameter_count": sum(
            parameter.numel() for parameter in build_model(
                payload["resolved_config"]).parameters()),
        "final_strategy": FINAL_STRATEGY,
        "final_strategy_settings": STRATEGIES[FINAL_STRATEGY],
        "minimum_driver_free_acceptance_bytes": MINIMUM_RESERVED_MARGIN_BYTES,
    }


def main() -> None:
    args = parse_args()
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("speed validation requires the real CUDA device")
    checkpoint = Path(args.checkpoint).resolve()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("epoch_boundary_only") is not True:
        raise ValueError("validation requires an epoch-boundary checkpoint")
    manifest_path = Path(
        payload["resolved_config"]["paths"]["train_manifest"]).resolve()
    dataset = SceneManifestDataset(manifest_path)
    device = torch.device(args.device)
    output = base_output(args, checkpoint, payload, device)
    modes = (
        ("numeric", "benchmark", "profile", "continuous")
        if args.mode == "all" else (args.mode,))
    for mode in modes:
        if mode == "numeric":
            precision = (
                "fp32" if args.numeric_precision == "fp32"
                else payload["resolved_config"]["training"]["mixed_precision"])
            output[mode] = numerical_validation(
                payload, dataset, device, precision)
        elif mode == "benchmark":
            output[mode] = benchmark(
                payload, dataset, device, args.benchmark_repeats)
        elif mode == "profile":
            output[mode] = profile_update(
                payload, dataset, device, args.profile_strategy,
                args.profile_workload, args.profile_rollout, args.trace)
        else:
            output[mode] = continuous_validation(
                payload, dataset, manifest_path, device,
                args.continuous_updates)
        atomic_json(Path(args.output), output)
    output["passed"] = all(output[mode]["passed"] for mode in modes)
    atomic_json(Path(args.output), output)
    print(json.dumps({
        "output": str(Path(args.output).resolve()), "modes": modes,
        "passed": output["passed"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
