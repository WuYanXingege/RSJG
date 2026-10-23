#!/usr/bin/env python3
"""Short paired benchmark for the rollout-OFF ZARA1 wide training path.

This script creates a temporary fresh 192/4/768 state, then restores that
identical state for the baseline and prefetch/runtime arms.  Its outputs are
engineering evidence only and are never valid training checkpoints.
"""

from __future__ import annotations

import argparse
import copy
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch

from benchmark_training_speed import (
    atomic_json, paired_checks, profile_one_baseline_on, run_arm,
)
from ebjd.config import build_model, load_config, trainer_options
from ebjd.trainer import EBJDTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/resolved/zara1.json")
    parser.add_argument("--legacy-checkpoint", required=True)
    parser.add_argument("--updates", type=int, default=4)
    parser.add_argument("--warmup-updates", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile", action="store_true")
    return parser.parse_args()


def arm_config(source: dict, optimized: bool) -> dict:
    result = copy.deepcopy(source)
    result["training"].update({
        "rollout_training_enabled": False,
        "paired_marginal_actual_step_projection_enabled": False,
        "execution_strategy": (
            "prefetch_runtime_v1" if optimized else
            "small_group_uncheckpoint_v1"),
        "training_runtime_optimizations": optimized,
        "data_loader_workers": 4 if optimized else 0,
        "data_loader_prefetch_factor": 2 if optimized else 0,
    })
    return result


def fresh_payload(config: dict, legacy_checkpoint: Path) -> dict:
    legacy = torch.load(legacy_checkpoint, map_location="cpu", weights_only=False)
    scales = legacy["scales"]
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = build_model(config)
    model.representation.set_scales(
        float(scales["representation_goal"]), float(scales["bridge"]))
    model.encoder.goal_scale.fill_(float(scales["encoder_goal"]))
    trainer = EBJDTrainer(model, trainer_options(config), config, {})
    loader_generator = torch.Generator().manual_seed(seed)
    # ActualStepAdamW represents a never-stepped parameter by an absent state.
    # A serialized list necessarily has one entry per parameter, so materialize
    # the mathematically equivalent step-0 zero moments for this temporary
    # restore-only benchmark fixture.
    optimizer_state = copy.deepcopy(trainer.optimizer.state_dict())
    optimizer_state["states"] = [
        {
            "moment": torch.zeros_like(parameter, dtype=torch.float32),
            "variance": torch.zeros_like(parameter, dtype=torch.float32),
            "step": 0,
        }
        for parameter in trainer.optimizer.parameters
    ]
    payload = {
        "schema": "ebjd-checkpoint-v2",
        "epoch_boundary_only": True,
        "model": copy.deepcopy(model.state_dict()),
        "optimizer": optimizer_state,
        "epoch": 0,
        "update_index": 0,
        "resolved_config": config,
        "run_identity": {},
        "selection_state": {},
        "scales": scales,
        "rng": trainer._rng_state(loader_generator),
        "extra": {"formal_training_artifact": False},
    }
    del trainer, model, legacy
    return payload


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.updates < 4:
        raise ValueError("at least four measured updates are required")
    source = load_config(args.config)
    baseline_cfg = arm_config(source, False)
    optimized_cfg = arm_config(source, True)
    payload = fresh_payload(source, Path(args.legacy_checkpoint))
    device = torch.device(args.device)
    started = time.perf_counter()
    baseline, baseline_state = run_arm(
        "workers0_runtime_off", baseline_cfg, payload, args.updates,
        args.warmup_updates, 1, device)
    optimized, optimized_state = run_arm(
        "workers4_prefetch2_runtime_on", optimized_cfg, payload, args.updates,
        args.warmup_updates, 1, device)
    correctness = paired_checks(
        baseline, optimized, baseline_state, optimized_state)
    profile = (
        profile_one_baseline_on(baseline_cfg, payload, 1, device)
        if args.profile else None)
    result = {
        "schema": "ebjd-zara1-wide-rollout-off-speed-v1",
        "formal_training_artifact": False,
        "architecture": {"hidden_dim": 192, "heads": 4, "ff_dim": 768},
        "updates": args.updates,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
        },
        "baseline": baseline,
        "optimized": optimized,
        "speedup": {
            "gpu_update": baseline["mixed_seconds_per_update"] /
            optimized["mixed_seconds_per_update"],
            "data": baseline["data_seconds_per_update"] /
            optimized["data_seconds_per_update"],
            "end_to_end": baseline["end_to_end_seconds_per_update"] /
            optimized["end_to_end_seconds_per_update"],
        },
        "correctness": correctness,
        "baseline_profile": profile,
        "gpu_wall_seconds": time.perf_counter() - started,
    }
    atomic_json(Path(args.output), result)
    print(json.dumps({
        "output": str(Path(args.output).resolve()),
        "speedup": result["speedup"],
        "correctness_pass": correctness["pass"],
        "gpu_wall_seconds": result["gpu_wall_seconds"],
    }, sort_keys=True), flush=True)
    if not correctness["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
