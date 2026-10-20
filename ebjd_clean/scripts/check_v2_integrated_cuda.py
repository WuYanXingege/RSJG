#!/usr/bin/env python3
"""One non-persisted real-ETH CUDA update exercising V2 rollout ON."""

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
    parser.add_argument("--parent", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("V2 real-batch preflight requires CUDA")
    if sha256(args.parent) != args.expected_sha256:
        raise ValueError("V2 CUDA preflight parent SHA256 mismatch")

    config = load_config(args.config)
    payload = torch.load(args.parent, map_location="cpu", weights_only=False)
    device = torch.device(args.device)
    torch.manual_seed(int(config["seed"]))
    model = build_model(config, "v2_full").to(device)
    options = trainer_options(config, "v2_full")
    # This is the required wiring-only intervention.  Formal V2 arms retain
    # rollout/projection OFF and this updated state is deliberately discarded.
    options.use_rollout_loss = True
    options.use_actual_step_constraint = True
    loader_generator = torch.Generator().manual_seed(int(config["seed"]) + 17)
    trainer = EBJDTrainer(model, options, config, {"diagnostic": True})
    restore = trainer.restore_named_extension_payload(payload, loader_generator)
    dataset = _dataset(config, "train", False, int(config["seed"]))
    loader = DataLoader(
        dataset, batch_size=int(config["training"]["batch_scenes"]),
        shuffle=True, collate_fn=collate_scenes, num_workers=0,
        generator=loader_generator)
    batch = augment_batch(next(iter(loader)), loader_generator)
    new_names = tuple(restore["new_parameter_names"])
    before = {
        name: value.detach().cpu().clone() for name, value in model.named_parameters()
        if name in new_names}
    torch.cuda.reset_peak_memory_stats(device)
    record = trainer.train_step([batch], epoch=21)
    torch.cuda.synchronize(device)
    changed = [
        name for name, value in model.named_parameters()
        if name in before and not torch.equal(value.detach().cpu(), before[name])]
    checks = {
        "real_rollout_executed": record["rollout_update"] == 1.0,
        "real_projection_executed": record["projection_executed"] == 1.0,
        "twenty_step_protocol": int(options.rollout_steps) == 20,
        "four_world_protocol": int(options.rollout_worlds) == 4,
        "new_parameters_updated": bool(changed),
        "finite_loss": bool(torch.isfinite(torch.tensor(record["loss"]))),
        "parent_file_unchanged": sha256(args.parent) == args.expected_sha256,
    }
    receipt = {
        "schema": "ebjd-v2-integrated-real-cuda-preflight-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "persisted_update": False,
        "discarded_state_after_check": True,
        "parent": {"path": str(Path(args.parent).resolve()),
                   "sha256": args.expected_sha256,
                   "epoch": int(payload["epoch"]),
                   "update_index": int(payload["update_index"])},
        "checks": checks,
        "record": {name: record[name] for name in (
            "rollout_update", "projection_executed", "projection_active",
            "loss", "rollout", "rollout_keep", "keep", "association",
            "base_random_draw_sha256", "rollout_random_draw_sha256")},
        "new_parameters_changed": changed,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }
    atomic_json(Path(args.output), receipt)
    print(json.dumps(receipt, sort_keys=True))
    if receipt["status"] != "PASS":
        raise RuntimeError("V2 real-CUDA preflight failed")


if __name__ == "__main__":
    main()
