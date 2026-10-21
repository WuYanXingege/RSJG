#!/usr/bin/env python3
"""Freeze the UNIV legacy-control versus NO_TOKEN transfer protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from copy import deepcopy
from pathlib import Path

import torch

from ebjd.config import config_differences, config_hash, validate_supported_config


PARENT_SHA = "f526506055f346513b7c57994f9935d40e624162d74e110104073312cfc4bbe9"
PROVIDER_SHA = "ebfbae9de25463497c7bcc20981022f0638bc0349c7eeeb38bfc235dfaf61a3e"
HELDOUT_SHA = "46c52540aac6f0313865f01f53d7c735d91cb8116fdbee39cb65f0cf545c7d8e"
TRAIN_SHA = "6f6fca362ec1e591bf9a79c9b1a0204123947a0c7b754992f131432af40dfd5d"
SEEDS = tuple(range(2035, 2040))


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def validate_bank(
    path: Path, seed: int, source_sha: str, scenes: int, agents: int,
) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "status": "COMPLETED", "seed": seed, "candidates": 20,
        "source_manifest_sha256": source_sha,
        "provider_checkpoint_sha256": PROVIDER_SHA,
        "future_consumed_by_provider": False,
        "scene_count": scenes, "agent_occurrence_count": agents,
    }
    mismatches = {
        name: {"expected": value, "actual": payload.get(name)}
        for name, value in expected.items() if payload.get(name) != value}
    if mismatches:
        raise ValueError(f"native bank identity differs at {path}: {mismatches}")
    for item in payload["files"]:
        shard = path.parent / item["path"]
        if sha256(shard) != item["sha256"]:
            raise ValueError(f"native bank shard hash mismatch: {shard}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--bank-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    source_root = project_root.parent
    commit = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(source_root), "status", "--porcelain", "--", "ebjd_clean"],
        text=True).strip()
    if dirty:
        raise ValueError("refusing to freeze UNIV transfer from a dirty source tree")

    parent = args.parent.resolve()
    if sha256(parent) != PARENT_SHA:
        raise ValueError("UNIV epoch73 parent SHA256 differs")
    payload = torch.load(parent, map_location="cpu", weights_only=False)
    if (payload.get("schema"), payload.get("epoch_boundary_only")) != (
            "ebjd-checkpoint-v2", True):
        raise ValueError("UNIV parent is not an epoch-boundary checkpoint")
    if (int(payload["epoch"]), int(payload["update_index"])) != (73, 18615):
        raise ValueError("UNIV parent must be epoch73/update18615")
    base = deepcopy(payload["resolved_config"])
    if config_hash(base) != payload["run_identity"]["config_hash"]:
        raise ValueError("UNIV parent embedded config hash differs")
    if sha256(base["paths"]["validation_manifest"]) != HELDOUT_SHA:
        raise ValueError("UNIV held-out manifest SHA256 differs")
    if sha256(base["paths"]["train_manifest"]) != TRAIN_SHA:
        raise ValueError("UNIV train manifest SHA256 differs")
    if sha256(base["paths"]["matched_gdts_validation_checkpoint"]) != PROVIDER_SHA:
        raise ValueError("UNIV matched GDTS provider SHA256 differs")

    bank_root = args.bank_root.resolve()
    train_path = bank_root / "train_seed3101/manifest.json"
    train_bank = validate_bank(train_path, 3101, TRAIN_SHA, 4067, 15186)
    heldout_paths: dict[str, str] = {}
    heldout_hashes: dict[str, str] = {}
    heldout_elapsed = 0.0
    provider_commits = {train_bank["provider_code_commit"]}
    for seed in SEEDS:
        path = bank_root / f"validation_seed{seed}/manifest.json"
        bank = validate_bank(path, seed, HELDOUT_SHA, 947, 24334)
        provider_commits.add(bank["provider_code_commit"])
        heldout_paths[str(seed)] = str(path)
        heldout_hashes[str(seed)] = sha256(path)
        heldout_elapsed += float(bank["elapsed_seconds"])
    if len(provider_commits) != 1:
        raise ValueError("UNIV native banks use different provider code commits")
    provider_commit = provider_commits.pop()

    run_root = args.run_root.resolve()
    control = deepcopy(base)
    control["experiment"]["source_commit"] = commit
    control["paths"]["output_root"] = str(run_root)
    candidate = deepcopy(control)
    candidate["experiment"]["implementation_base"] = "ebjd-v2-integrated-network-v1"
    candidate["training"]["lazy_off_projection_gradients"] = True
    candidate["training"]["static_conditioning_cache"] = True
    candidate["objectives"]["fixed_anchor_path_weight"] = 0.0
    candidate["native_goal_conditioning"] = {
        "enabled": True,
        "variant": "native_gdts_ttst20_cross_attention_v1",
        "candidates": 20,
        "train_cache_manifest": str(train_path),
        "train_cache_sha256": sha256(train_path),
        "validation_cache_by_seed": heldout_paths,
        "provider_checkpoint_sha256": PROVIDER_SHA,
        "provider_code_commit": provider_commit,
        "goal_scale_source": "representation_goal_scale",
    }
    candidate["v2"] = {
        "enabled": True, "variant": "no_token", "memory_mode": "legacy_joint",
        "motion_summaries": False, "map_memory_grid": 8,
        "anchor_residual": True, "joint_goal_assembly": True,
        "geometry_mapping": "decoder_jacobian", "keep_weight": 0.01,
        "association_weight": 0.05, "rollout_keep_weight": 0.01,
        "pairing_seed_offset": 3_000_007, "pairing_rounds": 2,
    }
    validate_supported_config(control)
    validate_supported_config(candidate)

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    control_path = output / "CONTROL_CONFIG.json"
    candidate_path = output / "CANDIDATE_CONFIG.json"
    atomic_json(control_path, control)
    atomic_json(candidate_path, candidate)
    updates_per_epoch = math.ceil(math.ceil(4067 / 4) / 4)
    plan = {
        "schema": "ebjd-univ-no-token-transfer-plan-v1",
        "status": "FROZEN_BEFORE_FORMAL_UPDATES",
        "source_commit": commit,
        "parent": {
            "path": str(parent), "sha256": PARENT_SHA,
            "epoch": 73, "update_index": 18615,
            "representation_scales": payload["scales"],
        },
        "assets": {
            "train_manifest_sha256": TRAIN_SHA,
            "heldout_manifest_sha256": HELDOUT_SHA,
            "provider_checkpoint_sha256": PROVIDER_SHA,
            "provider_code_commit": provider_commit,
            "train_bank_manifest_sha256": sha256(train_path),
            "heldout_bank_manifest_sha256_by_seed": heldout_hashes,
            "bank_export_elapsed_seconds": (
                float(train_bank["elapsed_seconds"]) + heldout_elapsed),
        },
        "configs": {
            "control": {"path": str(control_path), "sha256": sha256(control_path)},
            "candidate": {
                "path": str(candidate_path), "sha256": sha256(candidate_path)},
        },
        "config_differences_from_parent": {
            "control": config_differences(base, control),
            "candidate": config_differences(base, candidate),
        },
        "protocol": {
            "training_seed": 3101, "epochs": list(range(74, 82)),
            "updates_per_epoch": updates_per_epoch,
            "updates_per_arm": updates_per_epoch * 8,
            "terminal_update_index": 18615 + updates_per_epoch * 8,
            "inference_seeds": list(SEEDS), "worlds": 20,
            "ddim_steps": 20, "eta": 0.0, "formal_epoch": 81,
            "rollout": False, "actual_step_projection": False,
            "candidate_fixed_anchor_path_weight": 0.0,
            "gpu_budget_seconds": 21600,
        },
        "sole_comparison": (
            "legacy UNIV network Raw versus transferred NO_TOKEN network Fixed-anchor"),
    }
    atomic_json(output / "PLAN.json", plan)
    print(json.dumps({"status": "READY", "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
