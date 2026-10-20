#!/usr/bin/env python3
"""Materialize the frozen two-arm NO_TOKEN fixed-anchor path pilot."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path

import torch

from ebjd.config import config_hash, validate_supported_config


EXPECTED_PARENT = "cb201fe03d661bb7687b0592ad5ac45481b79cb06503fa73bdfb4b453c6043ff"


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    if sha256(args.parent) != EXPECTED_PARENT:
        raise ValueError("epoch25 parent SHA256 mismatch")
    payload = torch.load(args.parent, map_location="cpu", weights_only=False)
    if (payload.get("epoch"), payload.get("update_index")) != (25, 6425):
        raise ValueError("parent must be epoch25/update6425")
    origin = payload["resolved_config"]
    if config_hash(origin) != payload["run_identity"]["config_hash"]:
        raise ValueError("parent embedded config hash is inconsistent")
    if origin.get("objectives", {}).get("fixed_anchor_path_weight") is not None:
        raise ValueError("parent unexpectedly contains fixed-anchor path objective")
    configs = {}
    for arm, weight in (("CONTROL", 0.0), ("CANDIDATE", 0.10)):
        config = copy.deepcopy(origin)
        config["experiment"]["source_commit"] = args.source_commit
        config["paths"]["output_root"] = str(args.run_root.resolve())
        config["objectives"]["fixed_anchor_path_weight"] = weight
        validate_supported_config(config)
        path = args.output / f"{arm}_CONFIG.json"
        atomic_json(path, config)
        configs[arm] = {
            "path": str(path.resolve()), "config_hash": config_hash(config),
            "sha256": sha256(path), "fixed_anchor_path_weight": weight,
        }
    plan = {
        "schema": "ebjd-no-token-fixed-anchor-path-pilot-v1",
        "source_commit": args.source_commit,
        "parent": str(args.parent.resolve()),
        "parent_sha256": EXPECTED_PARENT,
        "parent_epoch": 25, "parent_update_index": 6425,
        "epochs": [26, 27, 28], "updates_per_epoch": 257,
        "updates_per_arm": 771, "terminal_update_index": 7196,
        "training_seed": 3101,
        "inference_seeds": [2035, 2036, 2037, 2038, 2039],
        "manifest_sha256": sha256(Path(origin["paths"]["validation_manifest"])),
        "formal_output": "fixed_anchor_rho0",
        "optimizer_updates_outside_two_arms": 0,
        "configs": configs,
    }
    atomic_json(args.output / "PLAN.json", plan)
    print(json.dumps({"status": "READY", "output": str(args.output.resolve())}))


if __name__ == "__main__":
    main()
