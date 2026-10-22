#!/usr/bin/env python3
"""Materialize one validated runtime config from the paper config stack."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from copy import deepcopy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ebjd.config import validate_supported_config  # noqa: E402


def _read(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"configuration fragment is not a mapping: {path}")
    return payload


def _merge(left: dict, right: dict) -> dict:
    result = deepcopy(left)
    for key, value in right.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _atomic_json(path: Path, payload: dict) -> None:
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fold", required=True, choices=("hotel", "univ", "zara1"))
    parser.add_argument("--output")
    args = parser.parse_args()
    payload = _merge(
        _read(ROOT / "configs" / "architecture.yaml"),
        _read(ROOT / "configs" / "training.yaml"),
    )
    payload = _merge(payload, _read(ROOT / "configs" / "folds" / f"{args.fold}.yaml"))
    payload["experiment"]["source_commit"] = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], check=True,
        capture_output=True, text=True).stdout.strip()
    validate_supported_config(payload)
    output = Path(args.output) if args.output else (
        ROOT / "configs" / "resolved" / f"{args.fold}.json")
    if not output.is_absolute():
        output = ROOT / output
    _atomic_json(output, payload)
    print(output)


if __name__ == "__main__":
    main()
