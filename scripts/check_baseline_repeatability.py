#!/usr/bin/env python3
"""Paired baseline-vs-baseline real-update reproducibility gate.

Both arms restore the same complete checkpoint and use the same execution
configuration.  The task ledger is updated before CUDA work and from finally,
so failed calls remain part of the declared GPU budget.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import time
from pathlib import Path

import torch

from benchmark_training_speed import (
    atomic_json, baseline_config, collect_microbatches, file_sha256,
    load_config, paired_checks, run_arm, _loader,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/resolved/hotel.json")
    parser.add_argument(
        "--checkpoint", default="assets/checkpoints/hotel/epoch_045_full.pt")
    parser.add_argument("--updates", type=int, default=16)
    parser.add_argument("--warmup-updates", type=int, default=1)
    parser.add_argument("--epoch", type=int, default=46)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    parser.add_argument("--task-state", required=True)
    return parser.parse_args()


def update_ledger(
    path: Path, *, status: str, call: dict | None = None,
    active_call: dict | None = None,
) -> None:
    if path.exists():
        ledger = json.loads(path.read_text(encoding="utf-8"))
    else:
        ledger = {
            "schema": "ebjd-rollout-backward-budget-v1",
            "gpu_budget_seconds": 1800.0,
            "recorded_gpu_wall_seconds": 0.0,
            "gpu_calls": [],
        }
    if call is not None:
        ledger.setdefault("gpu_calls", []).append(call)
        ledger["recorded_gpu_wall_seconds"] = sum(
            float(item["wall_seconds"]) for item in ledger["gpu_calls"])
    ledger["status"] = status
    if active_call is None:
        ledger.pop("active_call", None)
    else:
        ledger["active_call"] = active_call
    atomic_json(path, ledger)


def main() -> None:
    args = parse_args()
    state_path = Path(args.task_state)
    output_path = Path(args.output)
    started_wall = time.time()
    started = time.perf_counter()
    active = {
        "name": "baseline_repeatability_16_updates",
        "started_unix": started_wall,
        "updates_per_arm": args.updates,
    }
    update_ledger(state_path, status="GPU_RUNNING", active_call=active)
    outcome = "FAILED"
    failure = None
    try:
        if not torch.cuda.is_available() or not args.device.startswith("cuda"):
            raise RuntimeError("this gate requires an available CUDA device")
        if args.updates < 4:
            raise ValueError("at least four measured updates are required")
        device = torch.device(args.device)
        source = load_config(args.config)
        checkpoint = Path(args.checkpoint)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload.get("schema") != "ebjd-checkpoint-v2":
            raise ValueError("repeatability gate requires an EBJD v2 checkpoint")

        # Populate the OS cache without building a model or using CUDA.
        warm_generator = torch.Generator().manual_seed(1)
        warm_loader = _loader(source, "train", False, warm_generator)
        warm_iterator = iter(warm_loader)
        for _ in range(min(args.warmup_updates + args.updates, 17) * int(
            source["training"]["gradient_accumulation"]
        )):
            next(warm_iterator)
        del warm_loader, warm_iterator

        config = baseline_config(source)
        first, first_state = run_arm(
            "baseline_A", config, payload, args.updates, args.warmup_updates,
            args.epoch, device)
        progress = output_path.with_suffix(".progress.json")
        atomic_json(progress, {
            "schema": "ebjd-baseline-repeatability-progress-v1",
            "completed_arms": ["baseline_A"],
            "baseline_A": first,
            "call_started_unix": started_wall,
        })
        second, second_state = run_arm(
            "baseline_B", copy.deepcopy(config), payload, args.updates,
            args.warmup_updates, args.epoch, device)
        checks = paired_checks(first, second, first_state, second_state)
        result = {
            "schema": "ebjd-baseline-repeatability-v1",
            "formal_training_artifact": False,
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
                "gpu_total_memory_bytes": torch.cuda.get_device_properties(
                    device).total_memory,
            },
            "baseline_A": first,
            "baseline_B": second,
            "correctness": checks,
            "gpu_wall_seconds": time.perf_counter() - started,
        }
        atomic_json(output_path, result)
        outcome = "PASS" if checks["pass"] else "NUMERICAL_FAIL"
        print(json.dumps({
            "output": str(output_path.resolve()),
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
        elapsed = time.perf_counter() - started
        update_ledger(state_path, status="CPU_ANALYSIS", call={
            "name": active["name"],
            "started_unix": started_wall,
            "wall_seconds": elapsed,
            "outcome": outcome,
            "failure": failure,
            "output": str(output_path.resolve()),
        })


if __name__ == "__main__":
    main()
