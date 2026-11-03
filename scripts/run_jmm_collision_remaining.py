#!/usr/bin/env python3
"""Resumable single-GPU runner for the remaining frozen JMM collision banks."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np


PAPER = Path(__file__).resolve().parents[1]
ASSETS = PAPER.parent / "RSJG_EBJD_assets"
EVAL = ASSETS / "evaluations" / "eth_ucy_collision_metrics_20261102_103044"
PYTHON = PAPER.parent / ".conda" / "rsjg" / "bin" / "python"
SEEDS = range(2035, 2040)
EXPECTED = {"zara1": (705, 2356), "zara2": (998, 5910), "univ": (947, 24334)}


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def bank_is_complete(path: Path, fold: str, seed: int, kind: str) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=True) as bank:
            metadata = json.loads(str(bank["metadata"].item()))
            if metadata["seed"] != seed or metadata["fold"] != fold:
                return False
            if (metadata["scene_count"], metadata["agent_occurrence_count"]) != EXPECTED[fold]:
                return False
            needed = {"scene_ids", "agent_ids", "targets"}
            needed.add("predictions" if kind == "GDTS" else "predictions_G")
            if kind == "G_H":
                needed.add("predictions_H")
            return needed.issubset(bank.files) and len(bank["scene_ids"]) == EXPECTED[fold][0]
    except Exception:
        return False


def task(fold: str, seed: int, kind: str) -> tuple[Path, list[str]]:
    output_dir = EVAL / "banks" / fold / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    if kind == "G_H":
        output = output_dir / "G_H.npz"
        base = ASSETS / "experiments" / {
            "zara1": "zara1_unified_coupling_bridge_v1_20261028_214026",
            "zara2": "zara2_unified_coupling_bridge_v1_20261029_094715",
            "univ": "univ_unified_coupling_bridge_v1_20261031_111541",
        }[fold]
        config = base / "configs" / ("model_config.json" if fold == "zara1" else f"RESOLVED_{fold.upper()}_UNIFIED.json")
        checkpoint = base / "training" / "best_eligible.pt"
        checkpoint_sha = {
            "zara1": "062a497c7baa28bfe88cf7e559784ad7029e979ddf93f3e0f8b08baa4713214b",
            "zara2": "1dcdf54a2ea8081ee21480f1bc7a0499b89dff5a8ccc2c99357f9dd01243c3be",
            "univ": "39b5fce2b343516b36572df97289079ae00881ab0b380319bd5c10f3474957d3",
        }[fold]
        if fold == "zara1":
            if seed == 2035:
                bank = ASSETS / "experiments/zara1_frozen_endpoint_attribution_20261026/banks/gdts_proposal_B_seed2035.npz"
            else:
                bank = ASSETS / "experiments/zara1_native_prior_conditional_bridge_20261026/validation_banks" / f"gdts_proposal_B_seed{seed}.npz"
        else:
            bank = base / "proposal_banks" / "native_validation" / f"seed_{seed}" / f"gdts_proposal_B_seed{seed}.npz"
        command = [str(PYTHON), "scripts/export_ebjd_collision_bank.py", "--config", str(config),
                   "--checkpoint", str(checkpoint), "--checkpoint-sha256", checkpoint_sha,
                   "--validation-bank", str(bank), "--seed", str(seed), "--fold", fold,
                   "--output", str(output)]
        return output, command

    output = output_dir / "GDTS.accepted.npz"
    if fold == "zara1":
        pipeline = ASSETS / "zara1_full_pipeline_v1"
        command = [str(PYTHON), "scripts/export_gdts_collision_bank.py",
                   "--recipe", str(PAPER / "configs/asset_preparation/zara1_gdts.yaml"),
                   "--runtime-root", str(EVAL / "runtime/zara1"),
                   "--scene-cache-root", str(pipeline / "gdts_scene_cache"),
                   "--checkpoint", str(ASSETS / "checkpoints/zara1/matched_gdts_best.pt"),
                   "--checkpoint-sha256", "34af0f4335de77cf2b6ecd48b93262dfc4764e679bd52fd677392a85c22bf6d7",
                   "--manifest", str(ASSETS / "data/zara1_official/validation/validation_manifest.json"),
                   "--seed", str(seed), "--output", str(output)]
    elif fold == "zara2":
        pipeline = ASSETS / "experiments/formal_zara2_gt_bridge_h_20261028_100627"
        command = [str(PYTHON), "scripts/export_gdts_collision_bank.py",
                   "--recipe", str(PAPER / "configs/asset_preparation/zara2_gdts.yaml"),
                   "--runtime-root", str(EVAL / "runtime/zara2"),
                   "--scene-cache-root", str(pipeline / "gdts/scene_cache"),
                   "--checkpoint", str(pipeline / "gdts/runtime/output/zara2/runs/zara2_matched_gdts_seed3101_v1/saved_models/best_model.pt"),
                   "--checkpoint-sha256", "bba3e538775e57fac45fd693ae6308c84fd92d6bf8d5451ab337005b292b2d80",
                   "--manifest", str(pipeline / "data_manifest/validation/validation_manifest.json"),
                   "--seed", str(seed), "--output", str(output)]
    else:
        command = [str(PYTHON), "scripts/export_gdts_jdv2_cache_collision_bank.py",
                   "--legacy-root", str(PAPER.parent / "RSJG_JDV2_clean"),
                   "--config", str(PAPER.parent / "RSJG_JDV2_clean/configs/joint_dependency_v2/jdv2_stage_a_univ.yaml"),
                   "--checkpoint", str(ASSETS / "checkpoints/univ/matched_gdts_best.pt"),
                   "--checkpoint-sha256", "ebfbae9de25463497c7bcc20981022f0638bc0349c7eeeb38bfc235dfaf61a3e",
                   "--manifest", str(ASSETS / "data/univ_official/validation/validation_manifest.json"),
                   "--seed", str(seed), "--runtime-root", str(EVAL / "runtime/univ_jdv2"),
                   "--output", str(output)]
    return output, command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", default=str(EVAL / "BACKGROUND_STATUS.json"))
    parser.add_argument("--log", default=str(EVAL / "BACKGROUND_PIPELINE.log"))
    args = parser.parse_args()
    status_path, log_path = Path(args.status), Path(args.log)
    started = time.time()
    payload = {"schema": "ebjd-jmm-background-v1", "pid": os.getpid(), "status": "RUNNING",
               "started_unix": started, "current": None, "completed": [], "skipped": [],
               "failure": None, "log": str(log_path)}
    atomic_json(status_path, payload)
    # Finish the interrupted ZARA1 pair first, then switch folds exactly once.
    stages = [("zara1", s, k) for s in SEEDS for k in ("G_H", "GDTS")]
    stages += [("zara2", s, k) for s in SEEDS for k in ("G_H", "GDTS")]
    stages += [("univ", s, k) for s in SEEDS for k in ("G_H", "GDTS")]
    try:
        with log_path.open("a", encoding="utf-8", buffering=1) as log:
            print(f"[{time.strftime('%F %T')}] background pipeline pid={os.getpid()}", file=log)
            for fold, seed, kind in stages:
                output, command = task(fold, seed, kind)
                label = f"{fold}/seed_{seed}/{kind}"
                if bank_is_complete(output, fold, seed, kind):
                    payload["skipped"].append(label)
                    atomic_json(status_path, payload)
                    print(f"[{time.strftime('%F %T')}] SKIP verified {label}", file=log)
                    continue
                payload["current"] = label
                payload["current_output"] = str(output)
                payload["updated_unix"] = time.time()
                atomic_json(status_path, payload)
                print(f"[{time.strftime('%F %T')}] START {label}", file=log)
                subprocess.run(command, cwd=PAPER, stdout=log, stderr=subprocess.STDOUT, check=True)
                if not bank_is_complete(output, fold, seed, kind):
                    raise RuntimeError(f"post-run bank validation failed: {label}")
                payload["completed"].append(label)
                payload["updated_unix"] = time.time()
                atomic_json(status_path, payload)
                print(f"[{time.strftime('%F %T')}] DONE {label}", file=log)
            payload["status"] = "BANKS_COMPLETE"
            payload["current"] = "aggregate"
            atomic_json(status_path, payload)
            analysis = EVAL / "analysis"
            subprocess.run([str(PYTHON), "scripts/analyze_jmm_collision_banks.py",
                            "--bank-root", str(EVAL / "banks"),
                            "--formal-metrics", str(PAPER / "docs/results/ebjd_eth_ucy_unified_fivefold_20261101/PER_SEED_METRICS.csv"),
                            "--output", str(analysis)], cwd=PAPER, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
            payload["status"] = "COMPLETE"
            payload["current"] = None
            payload["finished_unix"] = time.time()
            payload["elapsed_seconds"] = payload["finished_unix"] - started
            atomic_json(status_path, payload)
            print(f"[{time.strftime('%F %T')}] COMPLETE", file=log)
    except BaseException as error:
        payload["status"] = "FAILED"
        payload["failure"] = f"{type(error).__name__}: {error}"
        payload["traceback"] = traceback.format_exc()
        payload["finished_unix"] = time.time()
        payload["elapsed_seconds"] = payload["finished_unix"] - started
        atomic_json(status_path, payload)
        raise


if __name__ == "__main__":
    main()
