#!/usr/bin/env python3
"""Materialize the sole-factor epoch-45 rollout-weight candidate config."""

from __future__ import annotations

import argparse
import json
import math
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


class ExactFloatSafeDumper(yaml.SafeDumper):
    pass


def _represent_exact_float(dumper: yaml.SafeDumper, value: float):
    if math.isnan(value):
        text = ".nan"
    elif math.isinf(value):
        text = ".inf" if value > 0 else "-.inf"
    else:
        text = repr(value)
        if "e" in text.lower():
            significand, exponent = text.lower().split("e", 1)
            if "." not in significand:
                significand += ".0"
            text = f"{significand}e{int(exponent):+03d}"
        elif "." not in text:
            text += ".0"
    return dumper.represent_scalar("tag:yaml.org,2002:float", text)


ExactFloatSafeDumper.add_representer(float, _represent_exact_float)


def atomic_yaml(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            yaml.dump(payload, handle, Dumper=ExactFloatSafeDumper, sort_keys=False)
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


def normalized_differences(control: dict, candidate: dict) -> dict:
    left, right = deepcopy(control), deepcopy(candidate)
    for payload in (left, right):
        payload["experiment"]["source_commit"] = "<IDENTITY>"
        payload["paths"]["output_root"] = "<IDENTITY>"
        payload["training"].update(EXECUTION_PROFILE)
    return config_differences(left, right)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-config", required=True)
    parser.add_argument("--reused-control-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experiment-output-root", required=True)
    args = parser.parse_args()
    source = load_config(args.source_config)
    control = load_config(args.reused_control_config)
    repository = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", "ebjd_clean"],
        cwd=repository, text=True).strip()
    if dirty:
        raise RuntimeError("refuse to materialize a formal config from a dirty tree")

    candidate = deepcopy(source)
    candidate["experiment"]["source_commit"] = commit
    candidate["paths"]["output_root"] = str(
        Path(args.experiment_output_root).resolve())
    candidate["training"].update(EXECUTION_PROFILE)
    candidate["training"]["rollout_weight_end"] = 0.05
    validate_supported_config(candidate)
    profile = validate_execution_migration_settings(candidate)
    if candidate["objectives"]["goal_v_weight"] != 1.0:
        raise RuntimeError("rollout pilot must retain goal_v_weight=1")

    identity_fields = {"experiment.source_commit", "paths.output_root"}
    execution_fields = {
        f"training.{name}" for name in EXECUTION_PROFILE
        if source["training"].get(name, "<MISSING>") != EXECUTION_PROFILE[name]
    }
    expected = identity_fields | execution_fields | {"training.rollout_weight_end"}
    source_differences = config_differences(source, candidate)
    if set(source_differences) != expected:
        raise RuntimeError(
            "candidate differs outside identity/execution/rollout-weight contract: "
            f"{sorted(source_differences)}")
    arm_differences = normalized_differences(control, candidate)
    if arm_differences != {
        "training.rollout_weight_end": {"old": 0.2, "new": 0.05}
    }:
        raise RuntimeError(
            f"normalized reused-control/candidate diff is not sole-factor: {arm_differences}")
    if semantic_config_hash(control) == semantic_config_hash(candidate):
        raise RuntimeError("rollout weight was incorrectly excluded from semantic identity")

    output = Path(args.output_dir).resolve()
    candidate_path = output / "CANDIDATE_ROLLOUT_WEIGHT_005.yaml"
    atomic_yaml(candidate, candidate_path)
    disk = load_config(candidate_path)
    if disk != candidate:
        raise RuntimeError("materialized YAML round-trip changed values or types")
    receipt = {
        "schema": "ebjd-rollout-weight005-materialization-v1",
        "source_commit": commit,
        "training_launched": False,
        "candidate": str(candidate_path),
        "candidate_config_hash": config_hash(disk),
        "candidate_semantic_hash": semantic_config_hash(disk),
        "reused_control_config": str(Path(args.reused_control_config).resolve()),
        "reused_control_config_hash": config_hash(control),
        "disk_round_trip_exact": True,
        "execution_profile": profile,
        "source_differences": source_differences,
        "normalized_paired_arm_differences": arm_differences,
        "sole_semantic_change": "training.rollout_weight_end: 0.20 -> 0.05",
    }
    atomic_json(receipt, output / "MATERIALIZATION_RECEIPT.json")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
