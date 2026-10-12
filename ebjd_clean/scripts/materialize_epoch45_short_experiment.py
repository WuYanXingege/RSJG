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

from ebjd.config import config_differences, load_config


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
    control = deepcopy(common)
    candidate = deepcopy(common)
    candidate["objectives"]["goal_v_weight"] = 2.0
    expected_control = {"experiment.source_commit", "paths.output_root"}
    expected_candidate = {*expected_control, "objectives.goal_v_weight"}
    if set(config_differences(source, control)) != expected_control:
        raise RuntimeError("control config differs outside identity/output fields")
    if set(config_differences(source, candidate)) != expected_candidate:
        raise RuntimeError("candidate config differs outside the preregistered goal weight")
    output = Path(args.output_dir).resolve()
    control_path = output / "CONTROL_GOAL_WEIGHT_1.yaml"
    candidate_path = output / "CANDIDATE_GOAL_WEIGHT_2.yaml"
    atomic_yaml(control, control_path)
    atomic_yaml(candidate, candidate_path)
    print(json.dumps({
        "source_commit": commit, "training_launched": False,
        "control": str(control_path), "candidate": str(candidate_path),
        "control_differences": config_differences(source, control),
        "candidate_differences": config_differences(source, candidate),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
