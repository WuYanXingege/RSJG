#!/usr/bin/env python3
"""Create and optionally execute the bounded cleanup allowlist.

This script intentionally knows only the five completed primary unified runs
and local Python/pytest caches.  It never traverses symlinks and never touches
proposal/prediction banks, data, baselines, providers, or active CR outputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import time
from pathlib import Path


ROOT = Path("/media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG")
PAPER = ROOT / "RSJG_EBJD_Paper"
ASSETS = ROOT / "RSJG_EBJD_assets"
RUNS = (
    ASSETS / "experiments/zara1_unified_coupling_bridge_v1_20261028_214026",
    ASSETS / "experiments/zara2_unified_coupling_bridge_v1_20261029_094715",
    ASSETS / "experiments/eth_unified_coupling_bridge_v1_20261030_104912",
    ASSETS / "experiments/hotel_unified_coupling_bridge_v1_20261030_211601",
    ASSETS / "experiments/univ_unified_coupling_bridge_v1_20261031_111541",
)
CR_ROOT = ASSETS / "evaluations/eth_ucy_collision_metrics_20261102_103044"
KEEP_NAMES = {"best_eligible.pt", "best_jade.pt", "last.pt", "final.pt"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def allocated(path: Path) -> int:
    if path.is_dir() and not path.is_symlink():
        return sum(
            item.lstat().st_blocks * 512 for item in path.rglob("*")
            if not item.is_symlink())
    return path.lstat().st_blocks * 512


def disk() -> dict:
    usage = shutil.disk_usage(ROOT)
    return {"total": usage.total, "used": usage.used, "free": usage.free}


def open_paths() -> set[Path]:
    result = set()
    proc = Path("/proc")
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        for name in ("cwd", "exe"):
            try:
                result.add((entry / name).resolve(strict=True))
            except (FileNotFoundError, PermissionError, OSError):
                pass
        fd = entry / "fd"
        try:
            children = list(fd.iterdir())
        except (FileNotFoundError, PermissionError, OSError):
            continue
        for child in children:
            try:
                result.add(child.resolve(strict=True))
            except (FileNotFoundError, PermissionError, OSError):
                pass
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--maintenance-dir", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    out = Path(args.maintenance_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    before = disk()
    opened = open_paths()
    protected = {
        "active_or_stale_cr_root": str(CR_ROOT),
        "paper_workspace": str(PAPER),
        "persistent_assets": str(ASSETS),
        "git_common_dir_host": str(ROOT / "RSJG_JDV2_clean/.git"),
        "environment": str(ROOT / ".conda/rsjg"),
        "cleanup_dirs_untouched": [
            str(PAPER / "docs/cleanup/20261022_143701"),
            str(PAPER / "docs/cleanup/20261022_164559"),
        ],
        "cr_status": str(CR_ROOT / "BACKGROUND_STATUS.json"),
        "cr_log": str(CR_ROOT / "BACKGROUND_PIPELINE.log"),
        "cr_checkpoints": [
            str(run / "training/best_eligible.pt")
            for run in RUNS if (run / "training/best_eligible.pt").is_file()
        ] + [
            str(run / "training/best_jade.pt")
            for run in RUNS if (run / "training/best_eligible.pt").is_file() is False
        ],
        "cr_banks": str(CR_ROOT / "banks"),
        "active_seed_repeat_registry": str(
            ASSETS / "experiments/hotel_zara2_training_repeats_20261101_113517"),
        "sdd_source_and_manifests": str(ASSETS / "data/sdd_trajnet_jmm"),
    }
    (out / "PROTECTED_PATHS.json").write_text(
        json.dumps(protected, indent=2, sort_keys=True) + "\n")

    rows = []
    for run in RUNS:
        training = run / "training"
        status_path, evaluation_path = (
            training / "STATUS.json", training / "FINAL_EVALUATION.json")
        if not status_path.is_file() or not evaluation_path.is_file():
            continue
        status = json.loads(status_path.read_text())
        evaluation = json.loads(evaluation_path.read_text())
        if status.get("status") != "COMPLETE" or evaluation.get("status") != "COMPLETE":
            continue
        selected = Path(evaluation["selected_checkpoint"]).resolve()
        if selected.parent != training.resolve() or not selected.is_file():
            continue
        if sha256(selected) != evaluation.get("selected_checkpoint_sha256"):
            raise ValueError(f"selected checkpoint hash mismatch: {selected}")
        for keep in KEEP_NAMES:
            candidate = training / keep
            if keep in {"best_jade.pt", "last.pt"} and not candidate.is_file():
                raise FileNotFoundError(f"required retained alias missing: {candidate}")
        for path in sorted(training.glob("epoch_*.pt")):
            resolved = path.resolve()
            if path.is_symlink() or resolved in opened:
                continue
            rows.append({
                "path": str(resolved), "kind": "intermediate_checkpoint",
                "logical_bytes": path.stat().st_size,
                "allocated_bytes": allocated(path), "sha256": sha256(path),
                "retained_selected_path": str(selected),
                "retained_selected_sha256": evaluation["selected_checkpoint_sha256"],
                "reason": "completed run; final evaluation persisted; epoch copy is not a retained alias",
                "action": "DELETE",
            })

    for path in sorted(PAPER.rglob("*")):
        if path.name not in {"__pycache__", ".pytest_cache"} or not path.is_dir():
            continue
        if path.is_symlink() or path in opened:
            continue
        if any(parent.name == "cleanup" for parent in path.parents):
            continue
        rows.append({
            "path": str(path.resolve()), "kind": "rebuildable_cache",
            "logical_bytes": sum(
                item.stat().st_size for item in path.rglob("*") if item.is_file()),
            "allocated_bytes": allocated(path), "sha256": "directory",
            "retained_selected_path": "", "retained_selected_sha256": "",
            "reason": "project-local Python/pytest cache; no open-file dependency",
            "action": "DELETE",
        })

    columns = (
        "path", "kind", "logical_bytes", "allocated_bytes", "sha256",
        "retained_selected_path", "retained_selected_sha256", "reason", "action")
    with (out / "DELETE_ALLOWLIST.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    receipts = []
    if args.execute:
        opened = open_paths()
        for row in rows:
            path = Path(row["path"])
            record = {**row, "deleted": False, "error": None}
            try:
                if not path.exists() or path.is_symlink():
                    raise RuntimeError("candidate missing or became a symlink")
                if path.resolve() in opened:
                    raise RuntimeError("candidate acquired an open-file dependency")
                if row["sha256"] != "directory" and sha256(path) != row["sha256"]:
                    raise RuntimeError("candidate hash changed after allowlisting")
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
                record["deleted"] = not path.exists()
            except Exception as error:  # preserve any candidate that changed
                record["error"] = repr(error)
            receipts.append(record)
    after = disk()
    receipt = {
        "schema": "ebjd-cleanup-receipt-v1", "executed": args.execute,
        "created_unix": time.time(), "items": receipts,
        "deleted_count": sum(bool(item["deleted"]) for item in receipts),
        "failed_count": sum(bool(item["error"]) for item in receipts),
        "deleted_allocated_bytes": sum(
            int(item["allocated_bytes"]) for item in receipts if item["deleted"]),
    }
    (out / "DELETION_RECEIPT.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    (out / "SPACE_BEFORE_AFTER.json").write_text(json.dumps({
        "before": before, "after": after,
        "filesystem_free_delta_bytes": after["free"] - before["free"],
        "allowlist_allocated_bytes": sum(int(row["allocated_bytes"]) for row in rows),
        "receipt_deleted_allocated_bytes": receipt["deleted_allocated_bytes"],
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "candidates": len(rows), "deleted": receipt["deleted_count"],
        "failed": receipt["failed_count"],
        "allocated_bytes": receipt["deleted_allocated_bytes"],
    }))


if __name__ == "__main__":
    main()
