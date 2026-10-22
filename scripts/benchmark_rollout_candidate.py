#!/usr/bin/env python3
"""Paired real-update benchmark for bounded rollout checkpoint candidates."""

from __future__ import annotations

import argparse
import copy
import json
import platform
import time
from pathlib import Path

import torch

from benchmark_training_speed import (
    atomic_json, baseline_config, file_sha256, load_config, paired_checks,
    run_arm,
)
from check_baseline_repeatability import update_ledger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/resolved/hotel.json")
    parser.add_argument(
        "--checkpoint", default="assets/checkpoints/hotel/epoch_045_full.pt")
    parser.add_argument("--candidate", choices=("small-group",), required=True)
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--warmup-updates", type=int, default=0)
    parser.add_argument("--epoch", type=int, default=46)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    parser.add_argument("--task-state", required=True)
    return parser.parse_args()


def candidate_config(source: dict, candidate: str) -> dict:
    result = baseline_config(source)
    if candidate == "small-group":
        result["training"].update({
            "execution_strategy": "small_group_uncheckpoint_v1",
            "rollout_uncheckpointed_max_agent_slots": 32,
            "training_runtime_optimizations": False,
        })
    return result


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    state_path = Path(args.task_state)
    started_wall = time.time()
    started = time.perf_counter()
    active = {
        "name": f"candidate_{args.candidate}_{args.updates}_updates",
        "started_unix": started_wall,
        "updates_per_arm": args.updates,
    }
    update_ledger(state_path, status="GPU_RUNNING", active_call=active)
    outcome = "FAILED"
    failure = None
    try:
        if not torch.cuda.is_available() or not args.device.startswith("cuda"):
            raise RuntimeError("this benchmark requires an available CUDA device")
        if args.updates < 4:
            raise ValueError("at least four measured updates are required")
        device = torch.device(args.device)
        source = load_config(args.config)
        checkpoint = Path(args.checkpoint)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload.get("schema") != "ebjd-checkpoint-v2":
            raise ValueError("candidate gate requires an EBJD v2 checkpoint")
        baseline_cfg = baseline_config(source)
        candidate_cfg = candidate_config(source, args.candidate)
        baseline, baseline_state = run_arm(
            "compact_grouped_v1", baseline_cfg, payload, args.updates,
            args.warmup_updates, args.epoch, device)
        progress_path = output_path.with_suffix(".progress.json")
        atomic_json(progress_path, {
            "schema": "ebjd-rollout-candidate-progress-v1",
            "candidate": args.candidate,
            "completed_arms": ["baseline"],
            "baseline": baseline,
            "call_started_unix": started_wall,
        })
        candidate, candidate_state = run_arm(
            args.candidate, copy.deepcopy(candidate_cfg), payload, args.updates,
            args.warmup_updates, args.epoch, device)
        checks = paired_checks(
            baseline, candidate, baseline_state, candidate_state)
        speedup = {
            "off": baseline["off_seconds_per_update"] /
            candidate["off_seconds_per_update"],
            "on": baseline["on_seconds_per_update"] /
            candidate["on_seconds_per_update"],
            "mixed": baseline["mixed_seconds_per_update"] /
            candidate["mixed_seconds_per_update"],
            "end_to_end": baseline["end_to_end_seconds_per_update"] /
            candidate["end_to_end_seconds_per_update"],
        }
        result = {
            "schema": "ebjd-rollout-candidate-benchmark-v1",
            "formal_training_artifact": False,
            "candidate_name": args.candidate,
            "config": str(Path(args.config).resolve()),
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": file_sha256(checkpoint),
            "checkpoint_epoch": int(payload["epoch"]),
            "checkpoint_update": int(payload["update_index"]),
            "epoch_used_for_schedule": args.epoch,
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(device),
            },
            "baseline": baseline,
            "candidate": candidate,
            "speedup": speedup,
            "correctness": checks,
            "gpu_wall_seconds": time.perf_counter() - started,
        }
        atomic_json(output_path, result)
        outcome = "PASS" if checks["pass"] else "NUMERICAL_FAIL"
        print(json.dumps({
            "output": str(output_path.resolve()),
            "speedup": speedup,
            "correctness_pass": checks["pass"],
            "model_max_abs": checks["model"]["max_abs"],
            "optimizer_max_abs": checks["optimizer"]["max_abs"],
            "gpu_wall_seconds": result["gpu_wall_seconds"],
        }, sort_keys=True), flush=True)
        if not checks["pass"]:
            raise SystemExit(2)
    except BaseException as error:
        failure = f"{type(error).__name__}: {error}"
        raise
    finally:
        update_ledger(state_path, status="CPU_ANALYSIS", call={
            "name": active["name"],
            "started_unix": started_wall,
            "wall_seconds": time.perf_counter() - started,
            "outcome": outcome,
            "failure": failure,
            "output": str(output_path.resolve()),
        })


if __name__ == "__main__":
    main()
