#!/usr/bin/env python3
"""Paired real-data optimizer-update benchmark for EBJD FULL training.

The baseline and optimized arms are restored independently from the same v2
checkpoint, including optimizer and RNG state.  No output is suitable for
resume, selection, or research evaluation.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from ebjd.config import build_model, load_config, trainer_options
from ebjd.data import SceneBatch, augment_batch
from ebjd.train import _loader, file_sha256
from ebjd.trainer import EBJDTrainer


ATOL = 2e-5
RTOL = 2e-5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/resolved/hotel.json")
    parser.add_argument(
        "--checkpoint", default="assets/checkpoints/hotel/epoch_045_full.pt")
    parser.add_argument("--updates", type=int, default=16)
    parser.add_argument("--warmup-updates", type=int, default=1)
    parser.add_argument("--epoch", type=int, default=46)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--profile-baseline-on", action="store_true")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def optimized_config(config: dict) -> dict:
    result = copy.deepcopy(config)
    result["training"].update({
        "execution_strategy": "paper_training_v1",
        "rollout_uncheckpointed_max_agent_slots": 0,
        "training_runtime_optimizations": True,
    })
    return result


def baseline_config(config: dict) -> dict:
    result = copy.deepcopy(config)
    result["training"].update({
        "execution_strategy": "compact_grouped_v1",
        "rollout_uncheckpointed_max_agent_slots": 0,
        "training_runtime_optimizations": False,
    })
    return result


def tensor_hash(digest: "hashlib._Hash", tensor: torch.Tensor) -> None:
    value = tensor.detach().contiguous().cpu().numpy()
    digest.update(str(value.dtype).encode())
    digest.update(str(value.shape).encode())
    digest.update(value.tobytes())


def batch_hash(microbatches: list[SceneBatch]) -> str:
    digest = hashlib.sha256()
    for batch in microbatches:
        digest.update(json.dumps(batch.scene_ids, separators=(",", ":")).encode())
        for value in (batch.observed, batch.future, batch.semantic_maps, batch.valid):
            tensor_hash(digest, value)
    return digest.hexdigest()


def cpu_state(trainer: EBJDTrainer) -> dict:
    return {
        "model": {
            name: value.detach().cpu().clone()
            for name, value in trainer.model.state_dict().items()
        },
        "optimizer": copy.deepcopy(trainer.optimizer.state_dict()),
        "update_index": trainer.update_index,
        "torch_cpu_rng": torch.get_rng_state().clone(),
        "torch_cuda_rng": [value.cpu().clone() for value in torch.cuda.get_rng_state_all()],
    }


def collect_microbatches(iterator, loader_generator, accumulation: int):
    batches = []
    data_seconds = 0.0
    for _ in range(accumulation):
        start = time.perf_counter()
        batch = next(iterator)
        batch = augment_batch(batch, loader_generator)
        data_seconds += time.perf_counter() - start
        batches.append(batch)
    audit_start = time.perf_counter()
    identity = batch_hash(batches)
    audit_seconds = time.perf_counter() - audit_start
    return batches, data_seconds, audit_seconds, identity


def run_arm(
    name: str, config: dict, payload: dict, updates: int, warmup_updates: int,
    epoch: int, device: torch.device,
) -> tuple[dict, dict]:
    loader_generator = torch.Generator()
    model = build_model(config).to(device)
    trainer = EBJDTrainer(model, trainer_options(config), config, {})
    trainer.restore_checkpoint_payload(
        payload, loader_generator=loader_generator, require_identity=False)
    model.train()
    loader = _loader(config, "train", False, loader_generator)
    iterator = iter(loader)
    accumulation = int(config["training"]["gradient_accumulation"])

    records = []
    for index in range(warmup_updates + updates):
        batches, data_seconds, audit_seconds, identity = collect_microbatches(
            iterator, loader_generator, accumulation)
        torch.cuda.synchronize(device)
        if index == warmup_updates:
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        record = trainer.train_step(batches, epoch)
        torch.cuda.synchronize(device)
        update_seconds = time.perf_counter() - start
        if index >= warmup_updates:
            records.append({
                "ordinal": index - warmup_updates,
                "source_update_index": trainer.update_index - 1,
                "rollout": bool(record["rollout_update"]),
                "data_seconds": data_seconds,
                "update_seconds": update_seconds,
                "end_to_end_seconds": data_seconds + update_seconds,
                "audit_hash_seconds_excluded": audit_seconds,
                "batch_sha256": identity,
                "losses": {
                    key: record[key] for key in (
                        "loss", "diffusion", "geometry", "map", "rollout",
                        "soft_marginal_ade", "soft_marginal_fde")
                },
                "constraint": {
                    key: record[key] for key in (
                        "projection_active", "projection_changed_fraction",
                        "candidate_dot_marginal_ade", "candidate_dot_marginal_fde",
                        "applied_dot_marginal_ade", "applied_dot_marginal_fde")
                },
                "rollout_step_checkpointed_group_fraction": record[
                    "rollout_step_checkpointed_group_fraction"],
                "maximum_physical_agent_slots": record[
                    "maximum_physical_agent_slots"],
            })
        del batches

    def mean_for(field: str, rollout: bool | None = None) -> float:
        selected = [
            item[field] for item in records
            if rollout is None or item["rollout"] is rollout]
        return statistics.mean(selected)

    summary = {
        "name": name,
        "updates": updates,
        "warmup_updates_excluded": warmup_updates,
        "rollout_updates": sum(item["rollout"] for item in records),
        "off_seconds_per_update": mean_for("update_seconds", False),
        "on_seconds_per_update": mean_for("update_seconds", True),
        "mixed_seconds_per_update": mean_for("update_seconds"),
        "data_seconds_per_update": mean_for("data_seconds"),
        "end_to_end_seconds_per_update": mean_for("end_to_end_seconds"),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "records": records,
    }
    state = cpu_state(trainer)
    state["loader_generator"] = loader_generator.get_state().clone()
    del trainer, model, loader, iterator
    gc.collect()
    torch.cuda.empty_cache()
    return summary, state


def compare_tensor_trees(left, right, prefix="") -> dict:
    result = {"max_abs": 0.0, "l2": 0.0, "tensor_count": 0, "bad": []}

    def walk(a, b, name):
        if torch.is_tensor(a):
            if not torch.is_tensor(b) or a.shape != b.shape or a.dtype != b.dtype:
                result["bad"].append(name + ": metadata")
                return
            difference = a.double() - b.double()
            maximum = float(difference.abs().max()) if difference.numel() else 0.0
            result["max_abs"] = max(result["max_abs"], maximum)
            result["l2"] += float(difference.square().sum())
            result["tensor_count"] += 1
            if not torch.allclose(a, b, atol=ATOL, rtol=RTOL):
                result["bad"].append(name)
        elif isinstance(a, dict):
            if not isinstance(b, dict) or list(a) != list(b):
                result["bad"].append(name + ": keys")
                return
            for key in a:
                walk(a[key], b[key], f"{name}.{key}" if name else str(key))
        elif isinstance(a, (list, tuple)):
            if not isinstance(b, type(a)) or len(a) != len(b):
                result["bad"].append(name + ": length")
                return
            for index, (aa, bb) in enumerate(zip(a, b, strict=True)):
                walk(aa, bb, f"{name}[{index}]")
        elif a != b:
            result["bad"].append(name + f": {a!r} != {b!r}")

    walk(left, right, prefix)
    result["l2"] = result["l2"] ** 0.5
    result["pass"] = not result["bad"]
    result["bad"] = result["bad"][:20]
    return result


def paired_checks(baseline: dict, optimized: dict, left_state: dict, right_state: dict) -> dict:
    record_checks = []
    for left, right in zip(baseline["records"], optimized["records"], strict=True):
        loss_delta = max(
            abs(left["losses"][key] - right["losses"][key])
            for key in left["losses"])
        constraint_delta = max(
            abs(left["constraint"][key] - right["constraint"][key])
            for key in left["constraint"])
        record_checks.append({
            "ordinal": left["ordinal"],
            "same_batch_sha256": left["batch_sha256"] == right["batch_sha256"],
            "same_rollout_schedule": left["rollout"] == right["rollout"],
            "max_loss_abs_delta": loss_delta,
            "max_constraint_abs_delta": constraint_delta,
        })
    model = compare_tensor_trees(left_state["model"], right_state["model"], "model")
    optimizer = compare_tensor_trees(
        left_state["optimizer"], right_state["optimizer"], "optimizer")
    rng = {
        "torch_cpu_exact": torch.equal(
            left_state["torch_cpu_rng"], right_state["torch_cpu_rng"]),
        "torch_cuda_exact": all(torch.equal(a, b) for a, b in zip(
            left_state["torch_cuda_rng"], right_state["torch_cuda_rng"], strict=True)),
        "loader_exact": torch.equal(
            left_state["loader_generator"], right_state["loader_generator"]),
    }
    passed = (
        model["pass"] and optimizer["pass"] and all(rng.values())
        and all(item["same_batch_sha256"] and item["same_rollout_schedule"]
                and item["max_loss_abs_delta"] <= ATOL
                and item["max_constraint_abs_delta"] <= ATOL
                for item in record_checks)
    )
    return {
        "atol": ATOL, "rtol": RTOL, "pass": passed,
        "model": model, "optimizer": optimizer, "rng": rng,
        "per_update": record_checks,
    }


def profile_one_baseline_on(config: dict, payload: dict, epoch: int, device: torch.device) -> dict:
    loader_generator = torch.Generator()
    model = build_model(config).to(device)
    trainer = EBJDTrainer(model, trainer_options(config), config, {})
    trainer.restore_checkpoint_payload(payload, loader_generator, require_identity=False)
    model.train()
    loader = _loader(config, "train", False, loader_generator)
    iterator = iter(loader)
    accumulation = int(config["training"]["gradient_accumulation"])
    # Advance exact full updates until the next rollout boundary.  The profile
    # itself is not included in the speed claim.
    while trainer.update_index % trainer.options.rollout_every:
        batches, _, _, _ = collect_microbatches(iterator, loader_generator, accumulation)
        trainer.train_step(batches, epoch)
        del batches
    batches, _, _, _ = collect_microbatches(iterator, loader_generator, accumulation)
    torch.cuda.synchronize(device)
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        profile_memory=True,
    ) as profiler:
        record = trainer.train_step(batches, epoch)
        torch.cuda.synchronize(device)
    rows = []
    for event in profiler.key_averages():
        if event.key.startswith("ebjd."):
            rows.append({
                "name": event.key,
                "cpu_time_total_us": float(event.cpu_time_total),
                "device_time_total_us": float(event.device_time_total),
                "cpu_memory_usage": int(event.cpu_memory_usage),
                "device_memory_usage": int(event.device_memory_usage),
                "calls": int(event.count),
            })
    rows.sort(key=lambda item: item["device_time_total_us"], reverse=True)
    result = {
        "rollout_update": bool(record["rollout_update"]),
        "note": "high-overhead profiler; excluded from timing claims",
        "record_function_totals": rows,
    }
    del trainer, model, loader, iterator, batches, profiler
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("this benchmark requires an available CUDA device")
    if args.updates < 4:
        raise ValueError("at least four measured updates are required")
    device = torch.device(args.device)
    source = load_config(args.config)
    checkpoint = Path(args.checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("schema") != "ebjd-checkpoint-v2":
        raise ValueError("benchmark requires an EBJD v2 checkpoint")

    # Warm the OS page cache only; no GPU/model work is performed here.
    warm_generator = torch.Generator().manual_seed(1)
    warm_loader = _loader(source, "train", False, warm_generator)
    warm_iterator = iter(warm_loader)
    for _ in range(min(args.warmup_updates + args.updates, 17) * int(
        source["training"]["gradient_accumulation"])):
        next(warm_iterator)
    del warm_loader, warm_iterator

    baseline_cfg = baseline_config(source)
    optimized_cfg = optimized_config(source)
    started = time.perf_counter()
    baseline, baseline_state = run_arm(
        "compact_grouped_v1", baseline_cfg, payload, args.updates,
        args.warmup_updates, args.epoch, device)
    progress_path = Path(args.output).with_suffix(".progress.json")
    atomic_json(progress_path, {
        "schema": "ebjd-paper-training-speed-benchmark-progress-v1",
        "completed_arms": ["baseline"],
        "baseline": baseline,
    })
    optimized, optimized_state = run_arm(
        "paper_training_v1", optimized_cfg, payload, args.updates,
        args.warmup_updates, args.epoch, device)
    atomic_json(progress_path, {
        "schema": "ebjd-paper-training-speed-benchmark-progress-v1",
        "completed_arms": ["baseline", "optimized"],
        "baseline": baseline,
        "optimized": optimized,
    })
    correctness = paired_checks(
        baseline, optimized, baseline_state, optimized_state)
    profile = (
        profile_one_baseline_on(baseline_cfg, payload, args.epoch, device)
        if args.profile_baseline_on else None)
    speedup = {
        "off": baseline["off_seconds_per_update"] / optimized["off_seconds_per_update"],
        "on": baseline["on_seconds_per_update"] / optimized["on_seconds_per_update"],
        "mixed": baseline["mixed_seconds_per_update"] / optimized["mixed_seconds_per_update"],
        "end_to_end": baseline["end_to_end_seconds_per_update"] /
        optimized["end_to_end_seconds_per_update"],
    }
    result = {
        "schema": "ebjd-paper-training-speed-benchmark-v1",
        "formal_training_artifact": False,
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint),
        "checkpoint_epoch": int(payload["epoch"]),
        "checkpoint_update": int(payload["update_index"]),
        "epoch_used_for_schedule": args.epoch,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "gpu_total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
        },
        "baseline": baseline, "optimized": optimized,
        "speedup": speedup, "correctness": correctness,
        "baseline_profile": profile,
        "gpu_wall_seconds": time.perf_counter() - started,
    }
    atomic_json(Path(args.output), result)
    print(json.dumps({
        "output": str(Path(args.output).resolve()),
        "speedup": speedup, "correctness_pass": correctness["pass"],
        "gpu_wall_seconds": result["gpu_wall_seconds"],
    }, sort_keys=True), flush=True)
    if not correctness["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
