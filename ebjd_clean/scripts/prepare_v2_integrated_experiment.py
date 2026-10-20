#!/usr/bin/env python3
"""Freeze the ETH EBJD-V2 integrated four-arm protocol before updates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from ebjd.data import SceneManifestDataset


PARENT_SHA = "54bec78024e356857b7c93866921521a88a87e9f9ef4572d2d23d5a17648ad0e"
HELDOUT_SHA = "03f9b6bef89d43137b31782bbf0eb5b16d19f8ddfe378e2c7c62e700e128397b"
PROVIDER_SHA = "126acf2a34f52986c536c397fe3acb04c769a3cde95077461d7971a1b0792950"


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


def nearest_anchor_p95(config: dict) -> tuple[float, int]:
    dataset = SceneManifestDataset(
        config["paths"]["train_manifest"],
        native_goal_bank=config["native_goal_conditioning"]["train_cache_manifest"])
    errors: list[np.ndarray] = []
    for row in dataset:
        candidates = row["native_goals"].numpy()
        target = row["future"][:, -1].numpy()
        errors.append(np.linalg.norm(candidates - target[:, None], axis=-1).min(1))
    values = np.concatenate(errors)
    return float(np.quantile(values, 0.95)), int(len(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    source_root = project_root.parent
    commit = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(source_root), "status", "--porcelain", "--", "ebjd_clean"],
        text=True).strip()
    if dirty:
        raise ValueError("refusing to freeze V2 protocol from a dirty ebjd_clean tree")

    parent = Path(args.parent).resolve()
    actual = sha256(parent)
    if actual != PARENT_SHA:
        raise ValueError(f"shared epoch20 SHA mismatch: {actual}")
    payload = torch.load(parent, map_location="cpu", weights_only=False)
    if payload.get("schema") != "ebjd-checkpoint-v2" or \
            payload.get("epoch_boundary_only") is not True:
        raise ValueError("parent is not an epoch-boundary v2 checkpoint")
    if (int(payload.get("epoch", -1)), int(payload.get("update_index", -1))) != (20, 5140):
        raise ValueError("parent must be epoch20/update5140")
    base = deepcopy(payload["resolved_config"])
    if sha256(base["paths"]["validation_manifest"]) != HELDOUT_SHA:
        raise ValueError("held-out manifest identity differs")
    if sha256(base["paths"]["matched_gdts_validation_checkpoint"]) != PROVIDER_SHA:
        raise ValueError("matched GDTS provider identity differs")
    native = base["native_goal_conditioning"]
    if sha256(native["train_cache_manifest"]) != native["train_cache_sha256"]:
        raise ValueError("train native-goal bank manifest identity differs")
    validation_bank_hashes = {
        seed: sha256(path) for seed, path in native["validation_cache_by_seed"].items()}

    run_root = Path(args.run_root).resolve()
    definitions = {
        "full": {
            "memory_mode": "split_gated", "motion_summaries": True,
            "map_memory_grid": 4, "anchor_residual": True,
            "joint_goal_assembly": True, "geometry_mapping": "decoder_jacobian",
            "keep_weight": 0.01, "association_weight": 0.05,
            "rollout_keep_weight": 0.01,
        },
        "no_token": {
            "memory_mode": "legacy_joint", "motion_summaries": False,
            "map_memory_grid": 8, "anchor_residual": True,
            "joint_goal_assembly": True, "geometry_mapping": "decoder_jacobian",
            "keep_weight": 0.01, "association_weight": 0.05,
            "rollout_keep_weight": 0.01,
        },
        "no_anchor": {
            "memory_mode": "split_gated", "motion_summaries": True,
            "map_memory_grid": 4, "anchor_residual": False,
            "joint_goal_assembly": False, "geometry_mapping": "decoder_jacobian",
            "keep_weight": 0.0, "association_weight": 0.0,
            "rollout_keep_weight": 0.0,
        },
    }
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    config_paths = {}
    for variant, fields in definitions.items():
        config = deepcopy(base)
        config["experiment"]["source_commit"] = commit
        config["experiment"]["implementation_base"] = "ebjd-v2-integrated-network-v1"
        config["paths"]["output_root"] = str(run_root)
        config["training"]["lazy_off_projection_gradients"] = True
        config["training"]["static_conditioning_cache"] = True
        config["map_encoder"]["memory_grid"] = [fields["map_memory_grid"]] * 2
        config["denoiser"]["memory_tokens_per_agent"] = (
            (12 if fields["motion_summaries"] else 8)
            + fields["map_memory_grid"] ** 2)
        config["v2"] = {
            "enabled": True, "variant": variant, **fields,
            "pairing_seed_offset": 3_000_007, "pairing_rounds": 2,
        }
        path = output / f"V2_{variant.upper()}_CONFIG.json"
        atomic_json(path, config)
        config_paths[variant] = path

    p95, train_agents = nearest_anchor_p95(base)
    plan = {
        "schema": "ebjd-v2-integrated-eth-plan-v1",
        "status": "FROZEN_BEFORE_FORMAL_UPDATES",
        "implementation_commit": commit,
        "branch": "research/ebjd-v2-integrated-network-v1",
        "parent": {
            "path": str(parent), "sha256": actual,
            "epoch": 20, "update_index": 5140,
            "source_commit": payload["run_identity"].get("source_commit"),
            "complete_state": True,
        },
        "assets": {
            "heldout_manifest_sha256": HELDOUT_SHA,
            "provider_checkpoint_sha256": PROVIDER_SHA,
            "train_goal_bank_manifest_sha256": sha256(native["train_cache_manifest"]),
            "validation_goal_bank_manifest_sha256_by_seed": validation_bank_hashes,
            "train_nearest_anchor_endpoint_error_p95_m": p95,
            "train_anchor_stat_agent_occurrences": train_agents,
        },
        "configs": {name: {"path": str(path), "sha256": sha256(path)}
                    for name, path in config_paths.items()},
        "arms": {
            "C": {"reuse_candidate": "rollout_boundary_OFF_epoch25",
                  "fallback": "rerun_from_shared_epoch20", "rollout": False},
            "V2_FULL": {"ablation": "v2_full", "epochs": [21, 22, 23, 24, 25]},
            "V2_NO_TOKEN": {"ablation": "v2_no_token", "epochs": [21, 22, 23, 24, 25]},
            "V2_NO_ANCHOR": {"ablation": "v2_no_anchor", "epochs": [21, 22, 23, 24, 25]},
        },
        "fixed_protocol": {
            "training_seed": 3101, "inference_seeds": list(range(2035, 2040)),
            "worlds": 20, "ddim_steps": 20, "eta": 0.0,
            "formal_checkpoint_epoch": 25, "rollout": False,
            "actual_step_projection": False, "lr_schedule_epochs": 100,
            "batch_scenes": 4, "gradient_accumulation": 4,
            "precision": "bf16_with_fp32_geometry_loss_optimizer",
            "new_gpu_budget_hours": 10, "speed_audit_budget_minutes": 45,
        },
        "execution": {
            "serial_cuda_only": True,
            "protected_service": "rsjg-ebjd-eth-rollout-boundary-v2.service",
            "speed_target_not_claim": "at least 1.3x if measured and compatible",
        },
    }
    atomic_json(output / "PLAN.json", plan)
    (output / "COMMANDS.md").write_text(
        "# EBJD V2 integrated ETH protocol\n\n"
        f"Source: `{source_root}`\n\nParent: `{parent}`\n\n"
        "Run only after the protected rollout service is inactive and CUDA is free:\n\n"
        "```bash\ncd " + str(project_root) + "\n"
        "bash scripts/run_v2_integrated_pipeline.sh\n```\n",
        encoding="utf-8")
    print(json.dumps(plan, sort_keys=True))


if __name__ == "__main__":
    main()
