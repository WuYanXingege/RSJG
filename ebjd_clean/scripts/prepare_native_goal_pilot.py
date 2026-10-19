#!/usr/bin/env python3
"""Freeze paired configs and PILOT_PLAN after native-goal banks are complete."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from copy import deepcopy
from pathlib import Path

import torch


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def validate_bank(path: Path, seed: int, provider_sha: str, provider_commit: str) -> dict:
    payload = json.loads(path.read_text())
    expected = {
        "schema": "ebjd-native-goal-bank-v1", "status": "COMPLETED",
        "seed": seed, "candidates": 20, "unit": "m",
        "provider_checkpoint_sha256": provider_sha,
        "provider_code_commit": provider_commit,
        "future_consumed_by_provider": False,
    }
    bad = {key: (payload.get(key), value) for key, value in expected.items()
           if payload.get(key) != value}
    if bad:
        raise ValueError(f"native goal bank identity mismatch: {bad}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", required=True)
    parser.add_argument("--expected-parent-sha256", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-output-root", required=True)
    args = parser.parse_args()
    parent = Path(args.parent).resolve()
    if sha256(parent) != args.expected_parent_sha256:
        raise ValueError("parent checkpoint SHA256 mismatch")
    payload = torch.load(parent, map_location="cpu", weights_only=False)
    if int(payload["epoch"]) != 5 or int(payload["update_index"]) != 1285:
        raise ValueError("pilot parent is not the frozen epoch-5 boundary")
    cache_root = Path(args.cache_root).resolve()
    provider_sha = "126acf2a34f52986c536c397fe3acb04c769a3cde95077461d7971a1b0792950"
    provider_commit = "092ed91f1f4117f50aa835ec8e47cd2c9ce74a98"
    train = cache_root / "train_seed3101_provider092_rngfix/manifest.json"
    validate_bank(train, 3101, provider_sha, provider_commit)
    heldout = {}
    for seed in range(2035, 2040):
        path = cache_root / f"validation_seed{seed}_provider092_rngfix/manifest.json"
        validate_bank(path, seed, provider_sha, provider_commit)
        heldout[str(seed)] = str(path)

    source_root = Path(__file__).resolve().parents[2]
    source_commit = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(
            ["git", "-C", str(source_root), "status", "--porcelain", "--", "ebjd_clean"],
            text=True).strip():
        raise ValueError("refusing to freeze a pilot from a dirty ebjd_clean tree")
    control = deepcopy(payload["resolved_config"])
    control["experiment"]["source_commit"] = source_commit
    control["paths"]["output_root"] = str(Path(args.run_output_root).resolve())
    candidate = deepcopy(control)
    candidate["native_goal_conditioning"] = {
        "enabled": True,
        "variant": "native_gdts_ttst20_cross_attention_v1",
        "candidates": 20,
        "train_cache_manifest": str(train),
        "train_cache_sha256": sha256(train),
        "validation_cache_by_seed": heldout,
        "provider_checkpoint_sha256": provider_sha,
        "provider_code_commit": provider_commit,
        "goal_scale_source": "representation_goal_scale",
    }
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    control_path, candidate_path = output / "CONTROL_CONFIG.json", output / "CANDIDATE_CONFIG.json"
    atomic_json(control_path, control)
    atomic_json(candidate_path, candidate)
    plan = {
        "schema": "ebjd-native-goal-conditioning-pilot-plan-v1",
        "status": "FROZEN_BEFORE_FORMAL_UPDATES",
        "implementation_commit": source_commit,
        "parent": {"path": str(parent), "sha256": sha256(parent),
                   "epoch": 5, "update_index": 1285},
        "provider": {"checkpoint_sha256": provider_sha,
                     "code_commit": provider_commit},
        "arms": {
            "control": {"config": str(control_path), "config_sha256": sha256(control_path)},
            "candidate": {"config": str(candidate_path), "config_sha256": sha256(candidate_path)},
        },
        "sole_arm_difference": "candidate adds native_goal_conditioning section and parameters",
        "training": {"seed": 3101, "epochs": [6, 7, 8, 9, 10],
                     "formal_comparison_checkpoint": "epoch_010.pt",
                     "rollout_active": False, "gpu_budget_hours": 10,
                     "runtime_decision": "five epochs per arm; historical ~12.5 min/epoch"},
        "evaluation": {"manifest_sha256":
            "03f9b6bef89d43137b31782bbf0eb5b16d19f8ddfe378e2c7c62e700e128397b",
            "seeds": list(range(2035, 2040)), "worlds": 20, "steps": 20,
            "precision": "FP32", "independent_test": False},
        "continuation_signal": {
            "candidate_vs_control_minFDE_relative_improvement_minimum": 0.05,
            "minFDE_seed_improvements_minimum": 4,
            "minADE_JADE_JFDE_must_not_worsen": True},
        "contract": {"candidate_minADE_minFDE_not_above_matched_GDTS": True,
                     "candidate_JADE_JFDE_strictly_below_matched_GDTS": True},
    }
    atomic_json(output / "PILOT_PLAN.json", plan)
    print(json.dumps(plan, sort_keys=True))


if __name__ == "__main__":
    main()
