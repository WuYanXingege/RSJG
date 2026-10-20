#!/usr/bin/env python3
"""Real-CUDA mathematical, restore, pairing, and one-update pilot preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ebjd.config import build_model, load_config, trainer_options
from ebjd.data import augment_batch, collate_scenes
from ebjd.train import _dataset, audit_restored_state
from ebjd.trainer import BatchRandomDraws, EBJDTrainer


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def trainer(config: dict, payload: dict, device: torch.device) -> tuple[EBJDTrainer, torch.Generator]:
    generator = torch.Generator().manual_seed(int(config["seed"]) + 17)
    model = build_model(config, "v2_no_token").to(device)
    value = EBJDTrainer(model, trainer_options(config, "v2_no_token"), config, {})
    value.restore_checkpoint_payload(payload, generator, require_identity=False)
    return value, generator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-config", type=Path, required=True)
    parser.add_argument("--candidate-config", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    started = time.perf_counter()
    if not torch.cuda.is_available():
        raise RuntimeError("fixed-anchor path preflight requires CUDA")
    if sha256(args.parent) != args.expected_sha256:
        raise ValueError("parent SHA256 mismatch")
    device = torch.device(args.device)
    control_config = load_config(args.control_config)
    candidate_config = load_config(args.candidate_config)
    payload = torch.load(args.parent, map_location="cpu", weights_only=False)
    if (payload["epoch"], payload["update_index"]) != (25, 6425):
        raise ValueError("preflight parent is not epoch25/update6425")

    # Materialize exactly four logical batch4 microbatches from the restored
    # loader stream.  Augmentation is performed once, then shared by both arms.
    loader_generator = torch.Generator().manual_seed(0)
    EBJDTrainer._restore_rng(payload["rng"], loader_generator)
    dataset = _dataset(control_config, "train", False, int(control_config["seed"]))
    loader = DataLoader(
        dataset, batch_size=int(control_config["training"]["batch_scenes"]),
        shuffle=True, collate_fn=collate_scenes, num_workers=0,
        generator=loader_generator)
    iterator = iter(loader)
    microbatches = [augment_batch(next(iterator), loader_generator) for _ in range(4)]

    # Lambda=0 must replay the missing-key legacy path exactly at fixed inputs.
    legacy_config = dict(control_config)
    legacy_config["objectives"] = dict(control_config["objectives"])
    legacy_config["objectives"].pop("fixed_anchor_path_weight")
    physical = microbatches[0].compact_scenes(0, 1).to(device)
    noise_generator = torch.Generator(device=device).manual_seed(991)
    noise = torch.randn(
        physical.observed.shape[0], 1, physical.observed.shape[1], 12, 2,
        generator=noise_generator, device=device)
    replay = {}
    noise_checks = {}
    torch.cuda.reset_peak_memory_stats(device)
    for tau in (0.10, 0.50):
        draws = BatchRandomDraws(
            torch.full((physical.observed.shape[0],), tau, device=device),
            noise.clone(), None)
        legacy, _ = trainer(legacy_config, payload, device)
        control, _ = trainer(control_config, payload, device)
        legacy_grad, _, _, legacy_log = legacy._batch_gradients(
            physical, 26, False, draws)
        control.restore_checkpoint_payload(payload, None, require_identity=False)
        control_grad, _, _, control_log = control._batch_gradients(
            physical, 26, False, draws)
        max_replay = max(float((a - b).abs().max()) for a, b in zip(
            legacy_grad, control_grad, strict=True))
        replay[str(tau)] = {
            "gradient_max_abs_error": max_replay,
            "loss_abs_error": abs(legacy_log["loss"] - control_log["loss"]),
        }
        candidate, _ = trainer(candidate_config, payload, device)
        candidate_grad, _, _, candidate_log = candidate._batch_gradients(
            physical, 26, False, draws)
        names = [name for name, _ in candidate.optimizer.named_parameters]
        bridge_indices = [index for index, name in enumerate(names)
                          if "bridge_head" in name]
        bridge_delta = math.sqrt(sum(float(
            (candidate_grad[index] - control_grad[index]).double().square().sum())
            for index in bridge_indices))
        noise_checks[str(tau)] = {
            "path_loss": candidate_log["fixed_anchor_path"],
            "bridge_gradient_delta_norm": bridge_delta,
            "finite": math.isfinite(candidate_log["fixed_anchor_path"]),
        }
        del legacy, control, candidate, legacy_grad, control_grad, candidate_grad
        torch.cuda.empty_cache()

    # One real batch4/accum4 optimizer update per arm from the same full state.
    records = {}
    restorations = {}
    for arm, config in (("control", control_config), ("candidate", candidate_config)):
        value, generator = trainer(config, payload, device)
        restorations[arm] = audit_restored_state(value, payload, generator)
        record = value.train_step(microbatches, epoch=26)
        records[arm] = {key: record[key] for key in (
            "logical_scene_agent_identity_sha256", "augmentation_sha256",
            "base_random_draw_sha256", "learning_rate", "unet_learning_rate",
            "fixed_anchor_path", "active_fixed_anchor_path_weight",
            "rollout_update", "projection_executed", "loss")}
        records[arm]["terminal_update_index"] = value.update_index
        optimizer = value.optimizer.state_dict()
        records[arm]["optimizer_state_count"] = sum(bool(state) for state in optimizer["states"])
        del value
        torch.cuda.empty_cache()
    paired = ("logical_scene_agent_identity_sha256", "augmentation_sha256",
              "base_random_draw_sha256", "learning_rate", "unet_learning_rate")
    checks = {
        "restoration_exact": all(all(value.values()) for value in restorations.values()),
        "lambda0_legacy_replay": all(
            row["gradient_max_abs_error"] == 0 and row["loss_abs_error"] == 0
            for row in replay.values()),
        "low_mid_path_finite_positive": all(
            row["finite"] and row["path_loss"] > 0 for row in noise_checks.values()),
        "low_mid_bridge_gradient_nonzero": all(
            row["bridge_gradient_delta_norm"] > 0 for row in noise_checks.values()),
        "formal_inputs_paired": all(
            records["control"][key] == records["candidate"][key] for key in paired),
        "formal_accum4_update": all(
            records[arm]["terminal_update_index"] == 6426 for arm in records),
        "optimizer_state_present": all(
            records[arm]["optimizer_state_count"] > 0 for arm in records),
        "control_path_skipped": records["control"]["fixed_anchor_path"] == 0
        and records["control"]["active_fixed_anchor_path_weight"] == 0,
        "candidate_path_active": records["candidate"]["fixed_anchor_path"] > 0
        and records["candidate"]["active_fixed_anchor_path_weight"] == 0.1,
        "rollout_projection_off": all(
            records[arm]["rollout_update"] == 0
            and records[arm]["projection_executed"] == 0 for arm in records),
    }
    torch.cuda.synchronize(device)
    receipt = {
        "schema": "ebjd-fixed-anchor-path-real-cuda-preflight-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "persisted_update": False, "checks": checks,
        "restorations": restorations, "lambda0_replay": replay,
        "noise_checks": noise_checks, "formal_update_records": records,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "wall_seconds": time.perf_counter() - started,
    }
    atomic_json(args.output, receipt)
    print(json.dumps(receipt, sort_keys=True))
    if receipt["status"] != "PASS":
        raise RuntimeError("fixed-anchor path preflight failed")


if __name__ == "__main__":
    main()
