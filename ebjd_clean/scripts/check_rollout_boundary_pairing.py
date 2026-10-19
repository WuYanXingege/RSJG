#!/usr/bin/env python3
"""One-update real-CUDA preflight for the rollout ON/OFF boundary pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("real-CUDA preflight requires CUDA")
    if sha256(args.checkpoint) != args.expected_sha256:
        raise ValueError("preflight checkpoint SHA256 mismatch")
    config = load_config(args.config)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    device = torch.device(args.device)
    records = {}

    for arm, ablation in (("on", "none"), ("off", "rollout_mechanism_off")):
        loader_generator = torch.Generator().manual_seed(int(config["seed"]) + 17)
        model = build_model(config, ablation).to(device)
        trainer = EBJDTrainer(
            model, trainer_options(config, ablation), config, {})
        trainer.restore_checkpoint_payload(
            payload, loader_generator, require_identity=False)
        # Exercise the real epoch-21 phase without changing any persisted state.
        trainer.update_index = 5140
        dataset = _dataset(config, "train", False, int(config["seed"]))
        loader = DataLoader(
            dataset, batch_size=int(config["training"]["batch_scenes"]),
            shuffle=True, collate_fn=collate_scenes, num_workers=0,
            generator=loader_generator)
        batch = augment_batch(next(iter(loader)), loader_generator)
        torch.cuda.reset_peak_memory_stats(device)
        record = trainer.train_step([batch], epoch=21)
        torch.cuda.synchronize(device)
        records[arm] = {
            key: record[key] for key in (
                "logical_scene_agent_identity_sha256", "augmentation_sha256",
                "base_random_draw_sha256", "rollout_random_draw_sha256",
                "learning_rate", "unet_learning_rate", "rollout_update",
                "projection_executed", "projection_active",
                "active_rollout_weight", "effective_rollout_coefficient",
                "loss", "rollout")}
        records[arm]["peak_cuda_allocated_bytes"] = int(
            torch.cuda.max_memory_allocated(device))
        del trainer, model, loader, dataset, batch
        torch.cuda.empty_cache()

    paired_fields = (
        "logical_scene_agent_identity_sha256", "augmentation_sha256",
        "base_random_draw_sha256", "learning_rate", "unet_learning_rate")
    checks = {
        "base_inputs_paired": all(
            records["on"][key] == records["off"][key]
            for key in paired_fields),
        "on_rollout_executed": records["on"]["rollout_update"] == 1.0,
        "on_projection_executed": records["on"]["projection_executed"] == 1.0,
        "off_rollout_skipped": records["off"]["rollout_update"] == 0.0,
        "off_projection_skipped": records["off"]["projection_executed"] == 0.0,
        "off_rollout_rng_unused": records["off"]["rollout_random_draw_sha256"] == "NONE",
        "on_early_ramp_weight": records["on"]["active_rollout_weight"] == 0.01,
    }
    receipt = {
        "schema": "ebjd-rollout-boundary-real-cuda-preflight-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": args.expected_sha256,
        "diagnostic_epoch": 21, "diagnostic_update_index_before": 5140,
        "persisted_update": False,
        "paired_fields": list(paired_fields), "checks": checks,
        "records": records,
    }
    atomic_json(Path(args.output), receipt)
    print(json.dumps(receipt, sort_keys=True))
    if receipt["status"] != "PASS":
        raise RuntimeError("rollout boundary pairing preflight failed")


if __name__ == "__main__":
    main()
