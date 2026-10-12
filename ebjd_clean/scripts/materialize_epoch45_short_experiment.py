#!/usr/bin/env python3
"""Materialize the preregistered five-epoch control and goal-weight fork."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from copy import deepcopy
from pathlib import Path

import yaml

from ebjd.config import (
    config_differences, config_hash, load_config, semantic_config_hash,
    validate_supported_config,
)
from ebjd.train import validate_execution_migration_settings


EXECUTION_PROFILE = {
    "epochs": 100,
    "execution_strategy": "small_group_uncheckpoint_v1",
    "scene_forward_chunk": 1,
    "scene_compaction": True,
    "physical_group_max_scenes": 4,
    "physical_group_max_agent_slots": 96,
    "physical_group_max_padding_ratio": 1.5,
    "rollout_block_checkpointing": False,
    "rollout_uncheckpointed_max_agent_slots": 32,
}


def atomic_yaml(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create, but do not run, the epoch-45 five-epoch paired configs")
    parser.add_argument("--source-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experiment-output-root", required=True)
    args = parser.parse_args()
    source = load_config(args.source_config)
    repository = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", "ebjd_clean"],
        cwd=repository, text=True).strip()
    if dirty:
        raise RuntimeError("refuse to materialize formal configs from a dirty tree")
    common = deepcopy(source)
    common["experiment"]["source_commit"] = commit
    common["paths"]["output_root"] = str(
        Path(args.experiment_output_root).resolve())
    common["training"].update(EXECUTION_PROFILE)
    control = deepcopy(common)
    candidate = deepcopy(common)
    candidate["objectives"]["goal_v_weight"] = 2.0
    validate_supported_config(control)
    validate_supported_config(candidate)
    control_profile = validate_execution_migration_settings(control)
    candidate_profile = validate_execution_migration_settings(candidate)
    if control_profile != candidate_profile:
        raise RuntimeError("paired arms do not share an identical execution profile")
    identity_fields = {"experiment.source_commit", "paths.output_root"}
    execution_fields = {
        f"training.{name}" for name in EXECUTION_PROFILE
        if source["training"].get(name, "<MISSING>") != EXECUTION_PROFILE[name]
    }
    expected_control = identity_fields | execution_fields
    expected_candidate = expected_control | {"objectives.goal_v_weight"}
    control_differences = config_differences(source, control)
    candidate_differences = config_differences(source, candidate)
    arm_differences = config_differences(control, candidate)
    if set(control_differences) != expected_control:
        raise RuntimeError(
            "control config differs outside categorized identity/execution fields")
    if set(candidate_differences) != expected_candidate:
        raise RuntimeError(
            "candidate config differs outside identity/execution/goal contract")
    if set(arm_differences) != {"objectives.goal_v_weight"}:
        raise RuntimeError("paired semantic configs differ by more than goal weight")
    if semantic_config_hash(control) == semantic_config_hash(candidate):
        raise RuntimeError("goal weight was incorrectly excluded from semantic identity")
    output = Path(args.output_dir).resolve()
    control_path = output / "CONTROL_GOAL_WEIGHT_1.yaml"
    candidate_path = output / "CANDIDATE_GOAL_WEIGHT_2.yaml"
    atomic_yaml(control, control_path)
    atomic_yaml(candidate, candidate_path)
    receipt = {
        "schema": "ebjd-goal-weight-paired-materialization-v2",
        "source_commit": commit, "training_launched": False,
        "control": str(control_path), "candidate": str(candidate_path),
        "control_config_hash": config_hash(control),
        "candidate_config_hash": config_hash(candidate),
        "control_semantic_hash": semantic_config_hash(control),
        "candidate_semantic_hash": semantic_config_hash(candidate),
        "execution_profile": control_profile,
        "categorized_changes": {
            "identity": sorted(identity_fields),
            "execution": sorted(execution_fields),
            "candidate_only_semantic": ["objectives.goal_v_weight"],
        },
        "control_differences": control_differences,
        "candidate_differences": candidate_differences,
        "paired_arm_differences": arm_differences,
    }
    atomic_json(receipt, output / "MATERIALIZATION_RECEIPT.json")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
