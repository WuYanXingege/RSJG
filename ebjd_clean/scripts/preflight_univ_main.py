#!/usr/bin/env python3
"""Targeted real-CUDA preflight for the formal UNIV EBJD configuration."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from ebjd.config import build_model, load_config, trainer_options
from ebjd.data import SceneManifestDataset, collate_scenes
from ebjd.metrics import hard_metrics
from ebjd.sampling import predict
from ebjd.train import validate_execution_migration_settings, verify_formal_baseline_binding
from ebjd.trainer import EBJDTrainer


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(payload: dict, path: Path) -> None:
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


def ranked_indices(manifest_path: Path, count: int) -> list[tuple[int, int]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ranked = []
    dataset_index = 0
    for descriptor in manifest["shards"]:
        with np.load(manifest_path.parent / descriptor["path"], allow_pickle=True) as payload:
            for observed in payload["observed"]:
                ranked.append((int(np.asarray(observed).shape[0]), dataset_index))
                dataset_index += 1
    return sorted(ranked, reverse=True)[:count]


def finite_record(record: dict) -> bool:
    return all(
        math.isfinite(float(value))
        for value in record.values() if isinstance(value, (int, float)))


def measured_update(config: dict, batches: list, epoch: int, update_index: int) -> dict:
    model = build_model(config).to("cuda")
    options = trainer_options(config)
    trainer = EBJDTrainer(model, options, config, {"preflight": True})
    trainer.update_index = update_index
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    free_before, total = torch.cuda.mem_get_info()
    torch.cuda.synchronize()
    started = time.perf_counter()
    record = trainer.train_step(batches, epoch)
    torch.cuda.synchronize()
    result = {
        "elapsed_seconds": time.perf_counter() - started,
        "record": record,
        "finite": finite_record(record),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "driver_free_before_bytes": int(free_before),
        "device_total_bytes": int(total),
        "agent_counts": [batch.valid.sum(1).tolist() for batch in batches],
        "scene_counts": [int(batch.observed.shape[0]) for batch in batches],
    }
    del trainer, model
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("UNIV formal preflight requires the real CUDA device")
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    profile = validate_execution_migration_settings(config)
    baseline = verify_formal_baseline_binding(config)
    train_manifest = Path(config["paths"]["train_manifest"]).resolve()
    validation_manifest = Path(config["paths"]["validation_manifest"]).resolve()
    train = SceneManifestDataset(train_manifest)
    validation = SceneManifestDataset(validation_manifest)

    top_train = ranked_indices(train_manifest, 16)
    logical_batches = [
        collate_scenes([train[index] for _, index in top_train[start:start + 4]])
        for start in range(0, 16, 4)
    ]
    # At epoch >=21, update_index mod 4 selects the real rollout path.
    normal = measured_update(config, logical_batches, epoch=22, update_index=1)
    rollout = measured_update(config, logical_batches, epoch=22, update_index=0)
    tail_indices = list(range(len(train) - (len(train) % 4), len(train)))
    if len(tail_indices) != 3:
        raise RuntimeError(f"expected three-scene UNIV loader tail, got {tail_indices}")
    tail_batch = collate_scenes([train[index] for index in tail_indices])
    tail = measured_update(config, [tail_batch], epoch=1, update_index=1)

    largest_validation = ranked_indices(validation_manifest, 1)[0]
    large_batch = collate_scenes([validation[largest_validation[1]]]).to("cuda")
    model = build_model(config).to("cuda").float().eval()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    generator = torch.Generator(device="cuda").manual_seed(2035)
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.no_grad():
        prediction, _ = predict(
            model, large_batch.observed, large_batch.semantic_maps,
            large_batch.valid, worlds=20, steps=20, generator=generator)
        metrics = hard_metrics(prediction, large_batch.future, large_batch.valid)
    torch.cuda.synchronize()
    large_inference = {
        "dataset_index": largest_validation[1],
        "agents": largest_validation[0], "worlds": 20, "steps": 20,
        "prediction_shape": list(prediction.shape),
        "finite": bool(torch.isfinite(prediction).all()),
        "metrics": metrics.__dict__,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }
    del model, prediction, large_batch
    torch.cuda.empty_cache()

    checks = {
        "normal_is_not_rollout": normal["record"]["rollout_update"] == 0,
        "rollout_is_active": rollout["record"]["rollout_update"] == 1,
        "rollout_effective_coefficient_is_0_08_at_epoch22": math.isclose(
            rollout["record"]["effective_rollout_coefficient"], 0.08,
            rel_tol=0, abs_tol=1e-8),
        "rollout_worlds_steps_exact": (
            config["training"]["rollout_worlds"] == 4
            and config["training"]["rollout_steps"] == 20),
        "full_weight_effective_coefficient_is_0_8": math.isclose(
            config["training"]["rollout_frequency_loss_compensation"]
            * config["training"]["rollout_weight_end"], 0.8,
            rel_tol=0, abs_tol=1e-12),
        "normal_finite": normal["finite"],
        "rollout_finite": rollout["finite"],
        "tail_finite": tail["finite"],
        "tail_has_three_scenes": tail["scene_counts"] == [3],
        "large_n_is_57": largest_validation[0] == 57,
        "large_n_p20x20_finite": large_inference["finite"],
    }
    result = {
        "schema": "ebjd-univ-targeted-cuda-preflight-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "config": str(config_path), "config_sha256": sha256(config_path),
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "execution_profile": profile, "baseline_binding": baseline,
        "train_scenes": len(train), "validation_scenes": len(validation),
        "top_train_indices_and_agents": top_train,
        "normal_update": normal, "rollout_update": rollout,
        "tail_update": tail, "large_n_inference": large_inference,
        "checks": checks,
    }
    atomic_json(result, Path(args.output).resolve())
    print(json.dumps({
        "status": result["status"], "output": str(Path(args.output).resolve()),
        "normal_seconds": normal["elapsed_seconds"],
        "rollout_seconds": rollout["elapsed_seconds"],
        "large_n_seconds": large_inference["elapsed_seconds"],
    }, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
