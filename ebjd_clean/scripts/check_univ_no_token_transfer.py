#!/usr/bin/env python3
"""Non-persisted real-CUDA preflight for the UNIV NO_TOKEN transfer."""

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
from ebjd.representation import cv_baseline
from ebjd.train import _dataset, audit_restored_state
from ebjd.trainer import EBJDTrainer


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def make_batches(config: dict, payload: dict):
    generator = torch.Generator().manual_seed(0)
    EBJDTrainer._restore_rng(payload["rng"], generator)
    loader = DataLoader(
        _dataset(config, "train", False, int(config["seed"])),
        batch_size=int(config["training"]["batch_scenes"]), shuffle=True,
        collate_fn=collate_scenes, num_workers=0, generator=generator)
    iterator = iter(loader)
    return [augment_batch(next(iterator), generator) for _ in range(4)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-config", required=True)
    parser.add_argument("--candidate-config", required=True)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    started = time.perf_counter()
    if not torch.cuda.is_available():
        raise RuntimeError("UNIV transfer preflight requires real CUDA")
    if sha256(args.parent) != args.expected_sha256:
        raise ValueError("UNIV parent SHA256 mismatch")
    payload = torch.load(args.parent, map_location="cpu", weights_only=False)
    if (int(payload["epoch"]), int(payload["update_index"])) != (73, 18615):
        raise ValueError("UNIV parent is not epoch73/update18615")
    control_config = load_config(args.control_config)
    candidate_config = load_config(args.candidate_config)
    device = torch.device(args.device)

    control_batches = make_batches(control_config, payload)
    candidate_batches = make_batches(candidate_config, payload)
    paired_batches = True
    native_shape_ok = True
    for left, right in zip(control_batches, candidate_batches, strict=True):
        paired_batches &= (
            left.scene_ids == right.scene_ids and left.agent_ids == right.agent_ids
            and torch.equal(left.observed, right.observed)
            and torch.equal(left.future, right.future)
            and torch.equal(left.valid, right.valid)
            and left.augmentation_sha256 == right.augmentation_sha256)
        native_shape_ok &= (
            right.native_goals is not None
            and right.native_goals.shape[:3] == (*right.valid.shape, 20)
            and right.native_goals.shape[-1] == 2
            and bool(torch.isfinite(right.native_goals).all()))

    torch.cuda.reset_peak_memory_stats(device)
    control_gen = torch.Generator().manual_seed(0)
    control_model = build_model(control_config, "rollout_mechanism_off").to(device)
    control = EBJDTrainer(
        control_model, trainer_options(control_config, "rollout_mechanism_off"),
        control_config, {"preflight": True})
    control.restore_checkpoint_payload(payload, control_gen, require_identity=False)
    control_restore = audit_restored_state(control, payload, control_gen)

    candidate_gen = torch.Generator().manual_seed(0)
    candidate_model = build_model(candidate_config, "v2_no_token").to(device)
    candidate = EBJDTrainer(
        candidate_model, trainer_options(candidate_config, "v2_no_token"),
        candidate_config, {"preflight": True})
    extension = candidate.restore_named_extension_payload(payload, candidate_gen)

    sample = candidate_batches[0].to(device)
    context = candidate_model.encode_context(
        sample.observed, sample.semantic_maps, sample.valid, sample.native_goals)
    context, _labels = candidate_model.assign_training_anchors(context, sample.future)
    anchors = context.world_anchors[:, 0]
    baseline, _ = cv_baseline(sample.observed)
    latent = candidate_model.representation.encode_target(
        sample.future, baseline, anchors)
    reconstructed, reconstructed_goal = candidate_model.representation.decode(
        latent, baseline, anchors)
    representation_error = float((reconstructed - sample.future).abs().max())
    goal_error = float((reconstructed_goal - sample.future[..., -1, :]).abs().max())
    assembled, permutations, _ = candidate_model.assemble_world_anchors(
        context, sample.scene_ids, sample.agent_ids, seed=2035, rounds=2)
    anchor_set_error = 0.0
    for batch_index in range(sample.valid.shape[0]):
        for agent_index in range(sample.valid.shape[1]):
            if bool(sample.valid[batch_index, agent_index]):
                proposed = assembled.world_anchors[batch_index, :, agent_index]
                native = sample.native_goals[batch_index, agent_index]
                anchor_set_error = max(anchor_set_error, float(
                    torch.cdist(proposed, native).min(-1).values.max()))
    del context, assembled, sample, latent, reconstructed, reconstructed_goal

    # Both arms begin their non-persisted update from the identical parent
    # global RNG boundary.  The V2 pairing generator is intentionally separate.
    EBJDTrainer._restore_rng(payload["rng"], None)
    control_record = control.train_step(control_batches, epoch=74)
    EBJDTrainer._restore_rng(payload["rng"], None)
    candidate_record = candidate.train_step(candidate_batches, epoch=74)
    paired_keys = (
        "logical_scene_agent_identity_sha256", "augmentation_sha256",
        "base_random_draw_sha256", "learning_rate", "unet_learning_rate")
    records_paired = all(
        control_record[key] == candidate_record[key] for key in paired_keys)
    record_names = (
        *paired_keys, "loss", "rollout_update", "projection_executed",
        "native_goal_gate_gradient", "native_goal_gate",
        "native_goal_condition_norm_ratio", "fixed_anchor_path",
        "active_fixed_anchor_path_weight")
    records = {
        "control": {name: control_record[name] for name in record_names},
        "candidate": {name: candidate_record[name] for name in record_names},
    }
    records["control"]["terminal_update_index"] = control.update_index
    records["candidate"]["terminal_update_index"] = candidate.update_index
    checks = {
        "parent_unchanged": sha256(args.parent) == args.expected_sha256,
        "control_full_state_restore_exact": all(control_restore.values()),
        "candidate_shared_state_restore_exact": bool(
            extension["shared_model_state_exact"]
            and extension["shared_optimizer_state_by_name_exact"]),
        "candidate_new_optimizer_state_empty": bool(
            extension["new_optimizer_state_empty"]),
        "four_real_batch4_inputs_paired": paired_batches,
        "native_p20_finite": native_shape_ok,
        "anchor_residual_roundtrip": representation_error <= 2e-5 and goal_error <= 2e-5,
        "joint_assembly_uses_native_set": anchor_set_error <= 1e-6,
        "joint_permutation_shape": permutations.shape[1] == 20,
        "one_accum4_update_each": (
            control.update_index == 18616 and candidate.update_index == 18616),
        "formal_random_inputs_paired": records_paired,
        "rollout_projection_off": all(
            row["rollout_update"] == 0 and row["projection_executed"] == 0
            for row in (control_record, candidate_record)),
        "path_loss_off": all(
            row["fixed_anchor_path"] == 0
            and row["active_fixed_anchor_path_weight"] == 0
            for row in (control_record, candidate_record)),
        "finite_losses": all(math.isfinite(row["loss"])
                             for row in (control_record, candidate_record)),
        "candidate_extension_receives_gradient": (
            candidate_record["native_goal_gate_gradient"] != 0),
    }
    torch.cuda.synchronize(device)
    receipt = {
        "schema": "ebjd-univ-no-token-transfer-cuda-preflight-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "persisted_update": False, "discarded_state_after_check": True,
        "checks": checks, "control_restore": control_restore,
        "candidate_extension_restore": extension,
        "representation_max_abs_error": representation_error,
        "goal_max_abs_error": goal_error,
        "native_anchor_set_max_error": anchor_set_error,
        "formal_update_records": records,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "wall_seconds": time.perf_counter() - started,
    }
    atomic_json(Path(args.output), receipt)
    print(json.dumps(receipt, sort_keys=True))
    if receipt["status"] != "PASS":
        raise RuntimeError("UNIV NO_TOKEN transfer preflight failed")


if __name__ == "__main__":
    main()
