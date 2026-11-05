#!/usr/bin/env python3
"""Freeze a user-approved Goal-pretrain best checkpoint with strict checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
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
    parser.add_argument("--legacy-root", required=True)
    parser.add_argument("--best", required=True)
    parser.add_argument("--last", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--planned-epochs", type=int, required=True)
    parser.add_argument("--completed-epochs", type=int, required=True)
    parser.add_argument("--abandoned-next-epoch-batches", type=int, default=0)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    legacy = Path(args.legacy_root).resolve()
    best_path, last_path = Path(args.best).resolve(), Path(args.last).resolve()
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    last = torch.load(last_path, map_location="cpu", weights_only=False)
    required = {
        "checkpoint_type", "epoch", "dataset", "test_set",
        "goal_architecture", "training_epochs_planned", "best_metric_name",
        "best_metrics", "best_metrics_epochs", "model_state_dict",
    }
    if not required.issubset(best):
        raise RuntimeError("Goal best checkpoint metadata is incomplete")
    if (best["checkpoint_type"] != "gdts_goal_pretrain" or
            best["dataset"] != "sdd" or best["test_set"] != "sdd"):
        raise RuntimeError("Goal checkpoint protocol identity mismatch")
    if int(last["epoch"]) != args.completed_epochs:
        raise RuntimeError("Goal last checkpoint is not the declared complete epoch")
    if int(best["training_epochs_planned"]) != args.planned_epochs:
        raise RuntimeError("Goal planned epoch identity mismatch")
    if int(best["best_metrics_epochs"]["goal_BCE"]) != int(best["epoch"]):
        raise RuntimeError("Goal best epoch metadata mismatch")
    sys.path.insert(0, str(legacy))
    from src.models.goal_pretrain import Goal_Pretrain
    model = Goal_Pretrain(
        SimpleNamespace(dataset="sdd", obs_length=8, pred_length=12),
        torch.device("cpu"), dataset=object())
    model.load_state_dict(best["model_state_dict"], strict=True)
    payload = {
        "schema": "ebjd-goal-pretrain-completion-v1",
        "stage_status": "COMPLETED",
        "completion_reason": "USER_APPROVED_EARLY_STOP_PLATEAU",
        "planned_epochs": args.planned_epochs,
        "completed_epochs": args.completed_epochs,
        "abandoned_next_epoch_batches": args.abandoned_next_epoch_batches,
        "training_seed": args.seed,
        "dataset": "sdd_jmm_trajnet",
        "selected_checkpoint": str(best_path),
        "selected_checkpoint_sha256": sha256(best_path),
        "selected_epoch": int(best["epoch"]),
        "validation_goal_BCE": float(best["best_metrics"]["goal_BCE"]),
        "last_checkpoint": str(last_path),
        "last_checkpoint_sha256": sha256(last_path),
        "strict_load": "PASS",
        "optimizer_updates_after_completion_boundary": 0,
    }
    atomic_json(Path(args.output), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
