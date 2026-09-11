#!/usr/bin/env python3
"""Evaluate one GDTS checkpoint with independently seeded inference runs.

This utility is intentionally explicit about its selection rule.  Every seed
evaluates the configured K trajectories once over the complete split.  The
summary then records the seed-level minima independently for minADE@K and
minFDE@K.  This is a best-of-seeds diagnostic, not a replacement for the
usual fixed-seed or repeated-run mean protocol.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True,
                  ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_checkpoint(run_dir: Path, value: str) -> Path:
    direct = Path(value).expanduser()
    candidates = []
    if direct.is_absolute():
        candidates.append(direct)
    elif value.isdigit():
        candidates.append(
            run_dir / "saved_models" / f"epoch_{int(value):03d}.pt"
        )
    elif value == "best":
        candidates.append(run_dir / "saved_models" / "best_model.pt")
    else:
        candidates.extend([Path.cwd() / direct, run_dir / direct])
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    searched = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(f"Checkpoint not found. Searched:\n{searched}")


def load_model_args(run_dir: Path, device_name: str, mode: str):
    import torch

    from src.parser import get_parser, check_and_add_additional_args

    config_path = run_dir / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Run config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Run config is not a mapping: {config_path}")

    parser = get_parser()
    parser.set_defaults(**config)
    model_args = parser.parse_args([])
    model_args.phase = "test"
    model_args.device = device_name
    model_args.use_wandb = False
    model_args.data_augmentation = False
    model_args.shuffle_test_batches = False
    model_args.reproducibility = True
    model_args.load_checkpoint = None
    model_args.pretrain_path = None
    model_args.force_reprocess = False
    model_args.num_joint_samples = None
    model_args.joint_goal_enabled = None
    model_args = check_and_add_additional_args(model_args)
    model_args.model_dir = str(run_dir)
    model_args.save_dir = str(run_dir)
    model_args.config = str(config_path)
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"Requested {device_name}, but CUDA is unavailable in this process"
        )
    if mode not in {"valid", "test"}:
        raise ValueError(f"Unsupported evaluation mode: {mode}")
    return model_args, config


def load_completed_records(output_dir: Path, expected_seeds: list[int]) -> list[dict]:
    records = []
    for seed in expected_seeds:
        result_path = output_dir / "per_seed" / f"seed_{seed}.json"
        if not result_path.is_file():
            continue
        with open(result_path, "r", encoding="utf-8") as handle:
            record = json.load(handle)
        if record.get("seed") != seed or record.get("status") != "complete":
            raise ValueError(f"Invalid resumable result: {result_path}")
        records.append(record)
    return records


def write_csv(path: Path, records: list[dict]) -> None:
    if not records:
        return
    metric_names = sorted(records[0]["metrics"])
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["seed", "elapsed_seconds", *metric_names],
        )
        writer.writeheader()
        for record in sorted(records, key=lambda item: item["seed"]):
            writer.writerow({
                "seed": record["seed"],
                "elapsed_seconds": record["elapsed_seconds"],
                **record["metrics"],
            })
    os.replace(temporary, path)


def build_summary(records: list[dict], mode: str, expected_count: int) -> dict:
    ade_name = f"{mode}_minADE@K"
    fde_name = f"{mode}_minFDE@K"
    payload = {
        "status": "complete" if len(records) == expected_count else "running",
        "completed_seed_count": len(records),
        "expected_seed_count": expected_count,
        "selection_policy": (
            "Select the minimum seed-level minADE@K and minFDE@K "
            "independently across all requested deterministic inference seeds."
        ),
        "reporting_label": "best-of-seeds diagnostic",
        "comparison_warning": (
            "Do not present these minima as a fixed-seed result, a repeated-run "
            "mean, or a standard K-only metric; selecting across seeds is "
            "optimistically biased."
        ),
    }
    if not records:
        return payload
    missing = [
        name for name in (ade_name, fde_name)
        if name not in records[0]["metrics"]
    ]
    if missing:
        raise KeyError(f"Expected selection metrics are missing: {missing}")
    best_ade = min(records, key=lambda item: item["metrics"][ade_name])
    best_fde = min(records, key=lambda item: item["metrics"][fde_name])
    payload.update({
        "best_minADE@K": {
            "metric_name": ade_name,
            "seed": best_ade["seed"],
            "value": best_ade["metrics"][ade_name],
            "all_metrics_at_seed": best_ade["metrics"],
        },
        "best_minFDE@K": {
            "metric_name": fde_name,
            "seed": best_fde["seed"],
            "value": best_fde["metrics"][fde_name],
            "all_metrics_at_seed": best_fde["metrics"],
        },
    })
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a resumable, independently seeded GDTS metric sweep."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", default="best")
    parser.add_argument("--mode", choices=["valid", "test"], default="test")
    parser.add_argument("--seed-start", type=int, default=2025)
    parser.add_argument("--num-seeds", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--num-refinement-steps",
        type=int,
        default=None,
        help=(
            "Optional inference-only override. Use 0 for a paired "
            "no-refinement ablation without changing the training config."
        ),
    )
    parser.add_argument(
        "--energy-weight",
        type=float,
        default=None,
        help="Optional inference-only pair-energy weight override.",
    )
    parser.add_argument(
        "--joint-sampling-temperature",
        type=float,
        default=None,
        help="Optional inference-only joint-sampling temperature override.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.num_seeds < 1:
        parser.error("--num-seeds must be at least 1")
    if args.num_refinement_steps is not None and args.num_refinement_steps < 0:
        parser.error("--num-refinement-steps cannot be negative")
    if (args.energy_weight is not None and
            (not np.isfinite(args.energy_weight) or args.energy_weight < 0)):
        parser.error("--energy-weight must be finite and non-negative")
    if (args.joint_sampling_temperature is not None and
            (not np.isfinite(args.joint_sampling_temperature) or
             args.joint_sampling_temperature <= 0)):
        parser.error(
            "--joint-sampling-temperature must be finite and positive"
        )
    return args


def main() -> None:
    import torch

    from src.data_pre_process import Trajectory_Data_Pre_Process
    from src.trainer import trainer
    from src.utils import set_seed

    cli = parse_args()
    run_dir = cli.run_dir.expanduser().resolve()
    output_dir = cli.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "per_seed").mkdir(exist_ok=True)
    (output_dir / "raw_logs").mkdir(exist_ok=True)

    checkpoint = resolve_checkpoint(run_dir, cli.checkpoint)
    model_args, config = load_model_args(run_dir, cli.device, cli.mode)
    if cli.num_refinement_steps is not None:
        model_args.num_refinement_steps = cli.num_refinement_steps
    if cli.energy_weight is not None:
        model_args.energy_weight = cli.energy_weight
    if cli.joint_sampling_temperature is not None:
        model_args.joint_sampling_temperature = \
            cli.joint_sampling_temperature
    seeds = list(range(cli.seed_start, cli.seed_start + cli.num_seeds))
    evaluation_overrides = {
        "num_refinement_steps": cli.num_refinement_steps,
    }
    # Keep old manifests resumable: newer optional keys are recorded only
    # when the corresponding override is actually requested.
    if cli.energy_weight is not None:
        evaluation_overrides["energy_weight"] = cli.energy_weight
    if cli.joint_sampling_temperature is not None:
        evaluation_overrides["joint_sampling_temperature"] = \
            cli.joint_sampling_temperature
    identity = {
        "schema_version": 1,
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_argument": str(cli.checkpoint),
        "mode": cli.mode,
        "device": cli.device,
        "seeds": seeds,
        "num_samples_per_seed": int(model_args.num_samples),
        "joint_sampling_strategy": model_args.joint_sampling_strategy,
        "goal_candidate_prior": model_args.goal_candidate_prior,
        "evaluation_overrides": evaluation_overrides,
        "config": config,
    }
    manifest_path = output_dir / "manifest.json"
    if manifest_path.is_file():
        with open(manifest_path, "r", encoding="utf-8") as handle:
            existing = json.load(handle)
        mismatches = {
            key: (existing.get(key), value)
            for key, value in identity.items()
            if existing.get(key) != value
        }
        if mismatches:
            raise ValueError(
                "Output directory belongs to another sweep. Mismatches: "
                f"{mismatches}"
            )
    else:
        atomic_write_json(manifest_path, {
            **identity,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        })

    records = load_completed_records(output_dir, seeds)
    print(
        f"Seed sweep: {len(records)}/{len(seeds)} already complete; "
        f"checkpoint={checkpoint.name}; mode={cli.mode}; "
        f"K={model_args.num_samples}",
        flush=True,
    )

    # Reuse model and dataloaders across seeds.  RNG state is reset immediately
    # before every complete evaluation pass, so each recorded seed is standalone.
    set_seed(cli.seed_start, use_cuda=model_args.use_cuda)
    Trajectory_Data_Pre_Process(model_args)
    evaluator = trainer(model_args)
    checkpoint_epoch = evaluator._load_state_file(str(checkpoint))
    evaluator.net.eval()

    completed_seeds = {record["seed"] for record in records}
    for index, seed in enumerate(seeds, start=1):
        if seed in completed_seeds:
            print(f"[{index}/{len(seeds)}] seed={seed} reused", flush=True)
            continue
        raw_log_path = output_dir / "raw_logs" / f"seed_{seed}.log"
        started = time.time()
        with open(raw_log_path, "w", encoding="utf-8") as raw_log:
            with redirect_stdout(raw_log), redirect_stderr(raw_log):
                set_seed(seed, use_cuda=model_args.use_cuda)
                with torch.inference_mode():
                    metrics = evaluator._evaluate_epoch(
                        checkpoint_epoch, mode=cli.mode
                    )
        clean_metrics = {
            name: float(value) for name, value in metrics.items()
            if np.isfinite(float(value))
        }
        if len(clean_metrics) != len(metrics):
            raise ValueError(f"Non-finite metric produced for seed {seed}")
        record = {
            "status": "complete",
            "seed": seed,
            "checkpoint_epoch": int(checkpoint_epoch),
            "elapsed_seconds": time.time() - started,
            "metrics": clean_metrics,
            "raw_log": str(raw_log_path),
        }
        atomic_write_json(
            output_dir / "per_seed" / f"seed_{seed}.json", record
        )
        records.append(record)
        records.sort(key=lambda item: item["seed"])
        write_csv(output_dir / "seed_metrics.csv", records)
        summary = build_summary(records, cli.mode, len(seeds))
        summary.update({
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": identity["checkpoint_sha256"],
            "checkpoint_epoch": int(checkpoint_epoch),
            "seeds": seeds,
            "num_samples_per_seed": int(model_args.num_samples),
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        })
        atomic_write_json(output_dir / "summary.json", summary)
        ade = clean_metrics[f"{cli.mode}_minADE@K"]
        fde = clean_metrics[f"{cli.mode}_minFDE@K"]
        print(
            f"[{index}/{len(seeds)}] seed={seed} "
            f"minADE@K={ade:.5f} minFDE@K={fde:.5f} "
            f"elapsed={record['elapsed_seconds']:.1f}s",
            flush=True,
        )

    summary = build_summary(records, cli.mode, len(seeds))
    summary.update({
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": identity["checkpoint_sha256"],
        "checkpoint_epoch": int(checkpoint_epoch),
        "seeds": seeds,
        "num_samples_per_seed": int(model_args.num_samples),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    atomic_write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
