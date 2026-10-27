#!/usr/bin/env python3
"""Evaluate a frozen legacy GDTS checkpoint on synchronized scene windows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

import torch
import yaml


METRICS = ("minADE@K", "minFDE@K", "JADE", "JFDE")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
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
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--scene-cache-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--parity-receipt", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=("valid", "test"), default="valid")
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    args = parser.parse_args()
    recipe = yaml.safe_load(Path(args.recipe).read_text(encoding="utf-8"))
    legacy = Path(recipe["legacy_gdts"]["root"]).resolve()
    runtime = Path(args.runtime_root).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    manifest = Path(args.manifest).resolve()
    parity_path = Path(args.parity_receipt).resolve()
    output = Path(args.output_dir).resolve()
    if sha256(checkpoint) != args.checkpoint_sha256:
        raise ValueError("matched GDTS checkpoint SHA256 mismatch")
    parity = json.loads(parity_path.read_text(encoding="utf-8"))
    if not parity.get("passed") or parity["manifest"]["sha256"] != sha256(manifest):
        raise ValueError("synchronized manifest parity is absent or mismatched")
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    statistics_payload = manifest_payload["statistics"]
    sys.path.insert(0, str(legacy))
    os.chdir(runtime)
    from src.models.model import GDTS
    from src.parser import check_and_add_additional_args, get_parser
    from src.trainer import trainer

    cli = [
        "--dataset", "eth5", "--test_set", "zara1", "--phase", "test",
        "--device", "cuda:0", "--batch_size", "64", "--num_workers", "0",
        "--down_factor", "8", "--skip_ts_window", "1",
        "--goal_model_type", "social", "--training_stage", "finetune",
        "--data_augmentation", "False", "--shuffle_train_batches", "False",
        "--shuffle_test_batches", "False", "--num_samples", "20",
        "--ddpm_step", "100", "--ddim_step", "20", "--trunk_stage_step", "30",
        "--reproducibility", "True", "--seed", str(recipe["seed"]),
        "--batch_cache_root", str(Path(args.scene_cache_root).resolve()),
        "--use_wandb", "False",
    ]
    parsed = get_parser().parse_args(cli)
    parsed = check_and_add_additional_args(parsed)
    parsed.model_dir = str(output / "runtime")
    parsed.save_dir = parsed.model_dir
    evaluator = trainer(parsed)
    parsed.goal_model_type = "independent"
    parsed.training_stage = "baseline"
    parsed.use_scene_latent = False
    parsed.use_dynamic_relation = False
    parsed.use_joint_energy = False
    parsed.use_dependency_corrector = False
    parsed.jdv2_active = False
    parsed.joint_goal_enabled = False
    parsed.best_metric = "auto"
    evaluator.args = parsed
    evaluator.net = GDTS(parsed, evaluator.device).to(evaluator.device)
    evaluator._pending_training_state = None
    evaluator._load_state_file(str(checkpoint), weights_only=True)
    evaluator.net.eval()

    per_seed = output / "per_seed"
    per_seed.mkdir(parents=True, exist_ok=True)
    runs = []
    for index, seed in enumerate(args.seeds, start=1):
        target = per_seed / f"seed_{seed}.json"
        if target.is_file():
            record = json.loads(target.read_text(encoding="utf-8"))
            if (
                record.get("status") == "complete"
                and record.get("seed") == seed
                and record.get("checkpoint_sha256") == args.checkpoint_sha256
                and record.get("manifest_sha256") == sha256(manifest)
            ):
                runs.append(record)
                print(f"[{index}/{len(args.seeds)}] seed={seed} reused", flush=True)
                continue
            raise RuntimeError(f"invalid existing per-seed result: {target}")
        started = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(evaluator.device)
        raw = evaluator._evaluate_epoch(
            0, mode=args.split, evaluation_seed=seed, metric_names=METRICS)
        record = {
            "status": "complete", "seed": seed,
            "checkpoint_sha256": args.checkpoint_sha256,
            "manifest_sha256": sha256(manifest),
            "worlds": 20, "ddim_steps": 20,
            "scene_count": int(statistics_payload["scenes"]),
            "agent_occurrence_count": int(statistics_payload["agents"]),
            "elapsed_seconds": time.perf_counter() - started,
            "peak_memory_bytes": (
                int(torch.cuda.max_memory_allocated(evaluator.device))
                if torch.cuda.is_available() else 0),
            "model_parameter_count": int(sum(
                parameter.numel() for parameter in evaluator.net.parameters())),
            "metrics": {
                "minADE": float(raw["valid_minADE@K"]),
                "minFDE": float(raw["valid_minFDE@K"]),
                "JADE": float(raw["valid_JADE"]),
                "JFDE": float(raw["valid_JFDE"]),
            },
        }
        atomic_json(target, record)
        runs.append(record)
        print(
            f"[{index}/{len(args.seeds)}] seed={seed} "
            f"minADE={record['metrics']['minADE']:.6f} "
            f"minFDE={record['metrics']['minFDE']:.6f} "
            f"JADE={record['metrics']['JADE']:.6f} "
            f"JFDE={record['metrics']['JFDE']:.6f} "
            f"elapsed={record['elapsed_seconds']:.1f}s", flush=True)
        names = ("minADE", "minFDE", "JADE", "JFDE")
        summary = {
            name: {
                "mean": float(sum(run["metrics"][name] for run in runs) / len(runs)),
                "sample_standard_deviation": (
                    float(statistics.stdev(run["metrics"][name] for run in runs))
                    if len(runs) > 1 else 0.0
                ),
            } for name in names
        }
        atomic_json(output / "results.json", {
            "schema": "ebjd-matched-gdts-validation-v1",
            "status": "complete" if len(runs) == len(args.seeds) else "running",
            "fold": "zara1",
            "split": "validation" if args.split == "valid" else "test",
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": args.checkpoint_sha256,
            "manifest": str(manifest), "manifest_sha256": sha256(manifest),
            "parity_receipt": str(parity_path),
            "parity_receipt_sha256": sha256(parity_path),
            "worlds": 20, "ddim_steps": 20,
            "scene_count": int(statistics_payload["scenes"]),
            "agent_occurrence_count": int(statistics_payload["agents"]),
            "evaluation_seeds": args.seeds, "runs": runs, "summary": summary,
            "coordinate_unit": "m", "per_agent_reordering": False,
        })


if __name__ == "__main__":
    main()
