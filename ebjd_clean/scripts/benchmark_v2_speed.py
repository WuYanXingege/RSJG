#!/usr/bin/env python3
"""Bounded same-state real-ETH CUDA benchmark for EBJD execution changes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ebjd.config import build_model, load_config, trainer_options
from ebjd.data import augment_batch, collate_scenes
from ebjd.train import _dataset
from ebjd.trainer import EBJDTrainer


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def state_digest(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))
    return float(ordered[index])


def run_window(
    config: dict, payload: dict, ablation: str, *, v2: bool,
    rollout: bool, optimized: bool, device: torch.device,
    warmup: int, measured: int,
) -> tuple[dict, dict[str, torch.Tensor]]:
    torch.manual_seed(int(config["seed"]))
    torch.cuda.manual_seed_all(int(config["seed"]))
    model = build_model(config, ablation).to(device)
    model.denoiser.static_context_cache_enabled = bool(optimized)
    options = trainer_options(config, ablation)
    options.lazy_off_projection_gradients = bool(optimized)
    options.use_rollout_loss = bool(rollout)
    options.use_actual_step_constraint = bool(rollout)
    loader_generator = torch.Generator().manual_seed(int(config["seed"]) + 17)
    trainer = EBJDTrainer(model, options, config, {"benchmark": True})
    if v2:
        trainer.restore_named_extension_payload(payload, loader_generator)
    else:
        trainer.restore_checkpoint_payload(
            payload, loader_generator, require_identity=False)
    dataset = _dataset(config, "train", False, int(config["seed"]))
    loader = DataLoader(
        dataset, batch_size=int(config["training"]["batch_scenes"]),
        shuffle=True, collate_fn=collate_scenes, num_workers=0,
        generator=loader_generator)
    iterator = iter(loader)
    times, records = [], []
    starts, ends = [], []
    scenes = agents = 0
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    wall_start = None
    for index in range(warmup + measured):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = augment_batch(batch, loader_generator)
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        if index == warmup:
            torch.cuda.synchronize(device)
            wall_start = time.perf_counter()
        start.record()
        record = trainer.train_step([batch], epoch=21)
        end.record()
        if index >= warmup:
            starts.append(start)
            ends.append(end)
            records.append(record)
            scenes += len(batch.scene_ids)
            agents += int(batch.valid.sum())
    torch.cuda.synchronize(device)
    wall = time.perf_counter() - wall_start
    times = [float(start.elapsed_time(end) / 1000) for start, end in zip(starts, ends)]
    state = {name: value.detach().cpu().clone()
             for name, value in model.state_dict().items()}
    result = {
        "optimized": optimized, "rollout_enabled": rollout,
        "warmup_updates": warmup, "measured_updates": measured,
        "update_seconds": times,
        "mean_update_seconds": float(statistics.fmean(times)),
        "p50_update_seconds": quantile(times, 0.50),
        "p90_update_seconds": quantile(times, 0.90),
        "wall_seconds": float(wall),
        "scenes_per_second": float(scenes / wall),
        "agents_per_second": float(agents / wall),
        "rollout_update_count": int(sum(row["rollout_update"] for row in records)),
        "ordinary_update_mean_seconds": float(statistics.fmean(
            value for value, row in zip(times, records) if not row["rollout_update"])),
        "rollout_update_mean_seconds": (
            float(statistics.fmean(
                value for value, row in zip(times, records) if row["rollout_update"]))
            if any(row["rollout_update"] for row in records) else None),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "input_receipts": [{key: row[key] for key in (
            "logical_scene_agent_identity_sha256", "augmentation_sha256",
            "base_random_draw_sha256", "rollout_random_draw_sha256")}
            for row in records],
        "state_sha256": state_digest(state),
    }
    del trainer, model, loader, iterator, dataset
    torch.cuda.empty_cache()
    return result, state


def compare(reference: tuple[dict, dict], optimized: tuple[dict, dict]) -> dict:
    left, left_state = reference
    right, right_state = optimized
    maximum = 0.0
    exact = True
    for name in left_state:
        a, b = left_state[name], right_state[name]
        exact = exact and torch.equal(a, b)
        if a.is_floating_point():
            maximum = max(maximum, float((a.float() - b.float()).abs().max()))
    receipts_equal = left["input_receipts"] == right["input_receipts"]
    speedup = left["wall_seconds"] / right["wall_seconds"]
    return {
        "reference": left, "optimized": right,
        "input_and_rng_receipts_exact": receipts_equal,
        "final_model_state_bitwise_exact": exact,
        "final_model_maximum_absolute_difference": maximum,
        "speedup": float(speedup),
        "training_only_epoch_seconds_estimate_reference": float(
            left["mean_update_seconds"] * 257),
        "training_only_epoch_seconds_estimate_optimized": float(
            right["mean_update_seconds"] * 257),
        "compatibility": (
            "BITWISE" if exact and receipts_equal else
            "NUMERIC_FP32_1E-5" if receipts_equal and maximum <= 1e-5 else "FAIL"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-config", required=True)
    parser.add_argument("--v2-config", required=True)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--updates", type=int, default=16)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("speed audit requires CUDA")
    if sha256(args.parent) != args.expected_sha256:
        raise ValueError("speed audit parent identity mismatch")
    if args.updates < 16 or args.updates % 4:
        raise ValueError("speed audit requires at least 16 updates in multiples of four")
    payload = torch.load(args.parent, map_location="cpu", weights_only=False)
    legacy, v2 = load_config(args.legacy_config), load_config(args.v2_config)
    device = torch.device(args.device)
    workloads = {}
    definitions = (
        ("legacy_off", legacy, "rollout_mechanism_off", False, False),
        ("legacy_on", legacy, "none", False, True),
        ("v2_full_off", v2, "v2_full", True, False),
    )
    for name, config, ablation, is_v2, rollout in definitions:
        reference = run_window(
            config, payload, ablation, v2=is_v2, rollout=rollout,
            optimized=False, device=device, warmup=args.warmup,
            measured=args.updates)
        optimized = run_window(
            config, payload, ablation, v2=is_v2, rollout=rollout,
            optimized=True, device=device, warmup=args.warmup,
            measured=args.updates)
        workloads[name] = compare(reference, optimized)
    compatibility_pass = all(
        row["compatibility"] != "FAIL" for row in workloads.values())
    receipt = {
        "schema": "ebjd-v2-integrated-speed-audit-v1",
        "status": "PASS" if compatibility_pass else "FAIL",
        "speed_target_met_all_workloads": all(
            row["speedup"] >= 1.3 for row in workloads.values()),
        "speed_target_is_goal_not_acceptance_requirement": True,
        "parent_sha256": args.expected_sha256,
        "environment": {
            "gpu": torch.cuda.get_device_name(device),
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "python": platform.python_version(),
            "precision": "bf16_with_fp32_geometry_loss_optimizer",
            "cuda_device": str(device),
        },
        "enabled_optimizations": [
            "single-context native candidate/anchor embedding cache",
            "no DDIM-loop scalar host synchronization",
            "lazy OFF-only projection gradient trees",
            "explicit non-reentrant checkpoint cache dependencies",
        ],
        "workloads": workloads,
    }
    atomic_json(Path(args.output), receipt)
    print(json.dumps({name: {"speedup": row["speedup"],
                            "compatibility": row["compatibility"]}
                      for name, row in workloads.items()}, sort_keys=True))
    if not compatibility_pass:
        raise RuntimeError("speed optimization compatibility failed")


if __name__ == "__main__":
    main()
