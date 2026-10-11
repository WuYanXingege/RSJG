#!/usr/bin/env python3
"""Paired validation for execution-only speedups of the active HOTEL seed.

Each invocation compares exactly one candidate with the already-certified
``compact_grouped_v1`` reference.  Loading a fresh checkpoint-backed trainer
for every repeat makes data, model/optimizer state and CUDA RNG seeds paired.
The script never writes a training checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from pathlib import Path

import torch

import validate_speed_optimization as base
from ebjd.config import build_model
from ebjd.data import SceneManifestDataset
from validate_oom_recovery import MINIMUM_RESERVED_MARGIN_BYTES, atomic_json, sha256


REFERENCE = {
    "scene_compaction": True,
    "physical_group_max_scenes": 4,
    "physical_group_max_agent_slots": 96,
    "physical_group_max_padding_ratio": 1.5,
    "rollout_block_checkpointing": False,
    "cache_fp32_map_features": False,
    "rollout_uncheckpointed_max_agent_slots": 0,
    "batched_protected_vjp": False,
}

CANDIDATES = {
    "map_cache": {
        "cache_fp32_map_features": True,
    },
    "uncheckpoint_slots8": {
        "rollout_uncheckpointed_max_agent_slots": 8,
    },
    "uncheckpoint_slots16": {
        "rollout_uncheckpointed_max_agent_slots": 16,
    },
    "uncheckpoint_slots24": {
        "rollout_uncheckpointed_max_agent_slots": 24,
    },
    "uncheckpoint_slots32": {
        "rollout_uncheckpointed_max_agent_slots": 32,
    },
    "map_cache_uncheckpoint_slots16": {
        "cache_fp32_map_features": True,
        "rollout_uncheckpointed_max_agent_slots": 16,
    },
    "map_cache_uncheckpoint_slots24": {
        "cache_fp32_map_features": True,
        "rollout_uncheckpointed_max_agent_slots": 24,
    },
    "map_cache_uncheckpoint_slots32": {
        "cache_fp32_map_features": True,
        "rollout_uncheckpointed_max_agent_slots": 32,
    },
    "batched_vjp": {
        "batched_protected_vjp": True,
    },
}


def settings(candidate: str) -> dict[str, dict]:
    selected = dict(REFERENCE)
    selected.update(CANDIDATES[candidate])
    return {"reference": dict(REFERENCE), candidate: selected}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--candidate", choices=tuple(CANDIDATES), required=True)
    parser.add_argument(
        "--mode", choices=("numeric", "benchmark", "profile", "continuous"),
        required=True)
    parser.add_argument("--numeric-precision", choices=("fp32", "configured"),
                        default="configured")
    parser.add_argument("--normal-repeats", type=int, default=8)
    parser.add_argument("--rollout-repeats", type=int, default=4)
    parser.add_argument("--continuous-updates", type=int, default=32)
    parser.add_argument(
        "--profile-workload", choices=("single", "small", "mixed", "n57"),
        default="mixed")
    parser.add_argument("--profile-rollout", action="store_true")
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)]


def paired_benchmark(payload, dataset, device, candidate, normal_repeats, rollout_repeats):
    if normal_repeats < 8 or rollout_repeats < 4:
        raise ValueError("screening requires at least 8 normal and 4 rollout repeats")
    batches = base.failure_batches(dataset)
    result = {
        "normal_repeats": normal_repeats,
        "rollout_repeats": rollout_repeats,
        "scene_indices": [list(item) for item in base.FAILURE_BATCH_INDICES],
        "agent_counts": [batch.valid.sum(1).tolist() for batch in batches],
        "strategies": {},
        "timing_scope": (
            "physical H2D, context, objectives, all three VJPs, clipping, "
            "AdamW proposal, FP64 projection and parameter commit"),
    }
    for strategy in ("reference", candidate):
        base.warmup_strategy(payload, dataset, device, strategy)
        strategy_result = {"settings": base.STRATEGIES[strategy]}
        for do_rollout, repeats in ((False, normal_repeats), (True, rollout_repeats)):
            records = []
            for repeat in range(repeats):
                trainer = base.prepare_measured_trainer(
                    payload, device, strategy, do_rollout,
                    24000 + 100 * int(do_rollout) + repeat)
                measured = base.measured_update(
                    trainer, batches, int(payload["epoch"]) + 1)
                records.append(measured)
                base.release(trainer)
            latency = [item["update_elapsed_seconds"] for item in records]
            strategy_result["rollout" if do_rollout else "normal"] = {
                "records": records,
                "p50_seconds": statistics.median(latency),
                "p90_seconds": percentile(latency, 0.9),
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
    selected = result["strategies"][candidate]["weighted_p50_seconds"]
    result["speedup_over_compact_grouped_v1"] = reference / selected
    result["weighted_reduction_fraction"] = 1.0 - selected / reference
    result["passed"] = (
        all(item["normal"]["passed"] and item["rollout"]["passed"]
            for item in result["strategies"].values())
        and result["strategies"][candidate]["rollout"][
            "minimum_conservative_driver_free_bytes"]
        >= MINIMUM_RESERVED_MARGIN_BYTES)
    return result


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("validation requires the real CUDA device")
    checkpoint = Path(args.checkpoint).resolve()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("epoch_boundary_only") is not True:
        raise ValueError("validation requires an epoch-boundary checkpoint")
    manifest_path = Path(
        payload["resolved_config"]["paths"]["train_manifest"]).resolve()
    dataset = SceneManifestDataset(manifest_path)
    device = torch.device("cuda")
    base.STRATEGIES = settings(args.candidate)
    base.FINAL_STRATEGY = args.candidate
    output = {
        "schema": "ebjd-current-seed-speedup-validation-v1",
        "candidate": args.candidate,
        "candidate_settings": base.STRATEGIES[args.candidate],
        "reference_settings": base.STRATEGIES["reference"],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_epoch": int(payload["epoch"]),
        "checkpoint_update_index": int(payload["update_index"]),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(device),
        "parameter_count": sum(
            parameter.numel() for parameter in build_model(
                payload["resolved_config"]).parameters()),
        "allocator_environment": {
            "PYTORCH_CUDA_ALLOC_CONF": os.environ.get(
                "PYTORCH_CUDA_ALLOC_CONF", "")},
        "mode": args.mode,
    }
    if args.mode == "numeric":
        precision = (
            "fp32" if args.numeric_precision == "fp32"
            else payload["resolved_config"]["training"]["mixed_precision"])
        result = base.numerical_validation(payload, dataset, device, precision)
    elif args.mode == "benchmark":
        result = paired_benchmark(
            payload, dataset, device, args.candidate,
            args.normal_repeats, args.rollout_repeats)
    elif args.mode == "profile":
        result = base.profile_update(
            payload, dataset, device, args.candidate,
            args.profile_workload, args.profile_rollout, None)
    else:
        result = base.continuous_validation(
            payload, dataset, manifest_path, device, args.continuous_updates)
    output[args.mode] = result
    output["passed"] = bool(result["passed"])
    atomic_json(Path(args.output), output)
    print(json.dumps({
        "candidate": args.candidate, "mode": args.mode,
        "output": str(Path(args.output).resolve()),
        "passed": output["passed"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
