#!/usr/bin/env python3
"""Measure full formal ZARA1 validation with workers 0 versus 4."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from benchmark_training_speed import atomic_json
from benchmark_zara1_off_training import arm_config, fresh_payload
from ebjd.config import build_model, load_config
from ebjd.train import _loader, validate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/resolved/zara1.json")
    parser.add_argument("--legacy-checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def run(config: dict, payload: dict, device: torch.device) -> dict:
    generator = torch.Generator().manual_seed(int(config["seed"]))
    model = build_model(config).to(device)
    model.load_state_dict(payload["model"], strict=True)
    loader = _loader(config, "validation", False, generator)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    metrics = validate(
        model, loader, str(device), int(config["evaluation"]["worlds"]),
        int(config["evaluation"]["steps"]),
        int(config["evaluation"]["selection_seed"]))
    torch.cuda.synchronize(device)
    return {
        "seconds": time.perf_counter() - started,
        "metrics": metrics,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }


def main() -> None:
    args = parse_args()
    source = load_config(args.config)
    payload = fresh_payload(source, Path(args.legacy_checkpoint))
    device = torch.device(args.device)
    baseline = run(arm_config(source, False), payload, device)
    torch.cuda.empty_cache()
    optimized = run(arm_config(source, True), payload, device)
    result = {
        "schema": "ebjd-zara1-wide-validation-loader-v1",
        "formal_training_artifact": False,
        "protocol": {"worlds": 20, "steps": 20, "seed": 2035},
        "baseline": baseline,
        "optimized": optimized,
        "speedup": baseline["seconds"] / optimized["seconds"],
        "metrics_identical": baseline["metrics"] == optimized["metrics"],
    }
    atomic_json(Path(args.output), result)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
