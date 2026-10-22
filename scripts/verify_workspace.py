#!/usr/bin/env python3
"""Verify assets, strict checkpoint loading and paper-workspace isolation."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ebjd.config import build_model, load_config  # noqa: E402


EXPECTED = {
    "hotel": {
        "checkpoint": "assets/checkpoints/hotel/epoch_045_full.pt",
        "sha256": "425988097793b639eb868fcc9ff28e41a3022c08598d13f396383da89268e1b1",
        "epoch": 45,
        "update_index": 11970,
    },
    "univ_epoch73": {
        "config_fold": "univ",
        "checkpoint": "assets/checkpoints/univ/epoch_073_selected_formal.pt",
        "sha256": "f526506055f346513b7c57994f9935d40e624162d74e110104073312cfc4bbe9",
        "epoch": 73,
    },
    "univ_epoch81_control": {
        "config_fold": "univ",
        "checkpoint": "assets/checkpoints/univ/control_epoch_081_full.pt",
        "sha256": "e9fd76be585fef0d9cf6ab82422e5a5775392030c59283054a5247a6f47eafdd",
        "epoch": 81,
        "update_index": 20655,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(fold: str, path: Path) -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "resolve_config.py"),
         "--fold", fold, "--output", str(path)], cwd=ROOT, check=True,
        capture_output=True, text=True)


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
    parser.add_argument("--output", default="results/FINALIZATION_VALIDATION.json")
    args = parser.parse_args()
    imported = Path(importlib.import_module("ebjd").__file__).resolve()
    if ROOT not in imported.parents:
        raise RuntimeError(f"ebjd imported outside paper workspace: {imported}")
    records = {}
    with tempfile.TemporaryDirectory(prefix="ebjd-paper-verify-") as directory:
        temporary = Path(directory)
        configs = {}
        for fold in ("hotel", "univ"):
            configs[fold] = temporary / f"{fold}.json"
            resolve(fold, configs[fold])
        for name, expected in EXPECTED.items():
            fold = expected.get("config_fold", name)
            checkpoint = ROOT / expected["checkpoint"]
            actual_hash = sha256(checkpoint)
            if actual_hash != expected["sha256"]:
                raise RuntimeError(f"{name} checkpoint SHA256 mismatch")
            config = load_config(configs[fold])
            model = build_model(config, "none")
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            model.load_state_dict(payload["model"], strict=True)
            loaded = model.state_dict()
            values_exact = all(
                torch.equal(loaded[key].cpu(), value.cpu())
                for key, value in payload["model"].items())
            if not values_exact:
                raise RuntimeError(f"{name} state values changed after strict load")
            if int(payload["epoch"]) != expected["epoch"]:
                raise RuntimeError(f"{name} epoch mismatch")
            if "update_index" in expected and int(payload["update_index"]) != expected["update_index"]:
                raise RuntimeError(f"{name} update mismatch")
            records[name] = {
                "status": "PASS", "checkpoint_sha256": actual_hash,
                "epoch": int(payload["epoch"]),
                "update_index": payload.get("update_index"),
                "state_tensors": len(model.state_dict()),
                "parameters": sum(value.numel() for value in model.parameters()),
                "strict_load": True, "state_values_exact": True,
            }
    result = {
        "schema": "ebjd-paper-finalization-validation-v1",
        "status": "PASS",
        "workspace": str(ROOT),
        "imported_ebjd": str(imported),
        "no_external_pythonpath_required": True,
        "checkpoints": records,
    }
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    _atomic_json(output, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
