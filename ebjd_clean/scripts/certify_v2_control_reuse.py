#!/usr/bin/env python3
"""Certify the completed rollout-OFF epoch25 as the V2 experiment control."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", required=True)
    parser.add_argument("--parent-sha256", required=True)
    parser.add_argument("--control-checkpoint", required=True)
    parser.add_argument("--control-result", required=True)
    parser.add_argument("--control-receipt", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    parent, control = Path(args.parent).resolve(), Path(args.control_checkpoint).resolve()
    if sha256(parent) != args.parent_sha256:
        raise ValueError("control reuse parent identity mismatch")
    payload = torch.load(control, map_location="cpu", weights_only=False)
    result, migration = read(args.control_result), read(args.control_receipt)
    checks = {
        "origin_is_shared_epoch20": migration.get("origin_checkpoint_sha256") == args.parent_sha256,
        "control_is_epoch25": int(payload.get("epoch", -1)) == 25,
        "control_update_is_6425": int(payload.get("update_index", -1)) == 6425,
        "control_ablation_is_rollout_off": payload.get("run_identity", {}).get(
            "ablation") == "rollout_mechanism_off",
        "result_checkpoint_exact": result.get("checkpoint_sha256") == sha256(control),
        "result_manifest_exact": result.get("manifest_sha256") ==
            "03f9b6bef89d43137b31782bbf0eb5b16d19f8ddfe378e2c7c62e700e128397b",
        "result_is_five_seed_P20": len(result.get("per_seed", [])) == 5
            and int(result.get("worlds", -1)) == 20
            and int(result.get("steps", -1)) == 20,
    }
    receipt = {
        "schema": "ebjd-v2-control-reuse-certificate-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "parent": {"path": str(parent), "sha256": args.parent_sha256},
        "control": {"checkpoint": str(control), "sha256": sha256(control),
                    "result": str(Path(args.control_result).resolve()),
                    "result_sha256": sha256(args.control_result)},
        "boundary": (
            "C is the frozen legacy native-goal rollout-OFF control; it is not "
            "an execution-speed comparison against V2 and has no V2 modules."),
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)
    print(json.dumps(receipt, sort_keys=True))
    if receipt["status"] != "PASS":
        raise RuntimeError("legacy control reuse certification failed")


if __name__ == "__main__":
    main()
