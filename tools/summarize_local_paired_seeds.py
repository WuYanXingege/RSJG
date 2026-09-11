#!/usr/bin/env python3
"""Aggregate two local GDTS seed sweeps with paired inference statistics."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics

from scipy.stats import t as student_t


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--baseline-label", required=True)
    parser.add_argument("--candidate-label", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--metrics", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must be unique")
    return args


def read_json(path: Path) -> dict:
    with open(path.expanduser().resolve(), "r", encoding="utf-8") as handle:
        return json.load(handle)


def describe(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def paired_inference(differences: list[float]) -> dict:
    count = len(differences)
    mean_difference = statistics.mean(differences)
    if count < 2:
        return {
            "mean_difference": mean_difference,
            "degrees_of_freedom": 0,
            "t_statistic": 0.0,
            "two_sided_p_value": 1.0,
            "mean_difference_95_ci": [mean_difference, mean_difference],
        }
    standard_deviation = statistics.stdev(differences)
    if standard_deviation == 0.0:
        statistic = 0.0 if mean_difference == 0.0 else math.copysign(
            math.inf, mean_difference
        )
        p_value = 1.0 if mean_difference == 0.0 else 0.0
        margin = 0.0
    else:
        standard_error = standard_deviation / math.sqrt(count)
        statistic = mean_difference / standard_error
        p_value = float(2.0 * student_t.sf(abs(statistic), count - 1))
        margin = float(student_t.ppf(0.975, count - 1) * standard_error)
    return {
        "mean_difference": mean_difference,
        "degrees_of_freedom": count - 1,
        "t_statistic": statistic,
        "two_sided_p_value": p_value,
        "mean_difference_95_ci": [
            mean_difference - margin,
            mean_difference + margin,
        ],
    }


def atomic_write(path: Path, payload: dict) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    baseline_dir = args.baseline_dir.expanduser().resolve()
    candidate_dir = args.candidate_dir.expanduser().resolve()
    baseline_manifest = read_json(baseline_dir / "manifest.json")
    candidate_manifest = read_json(candidate_dir / "manifest.json")
    protocol_fields = ("mode", "num_samples_per_seed", "seeds")
    for field in protocol_fields:
        if baseline_manifest.get(field) != candidate_manifest.get(field):
            raise ValueError(f"Protocol mismatch in {field}")

    per_seed = []
    for seed in args.seeds:
        baseline_path = baseline_dir / "per_seed" / f"seed_{seed}.json"
        candidate_path = candidate_dir / "per_seed" / f"seed_{seed}.json"
        baseline_record = read_json(baseline_path)
        candidate_record = read_json(candidate_path)
        if baseline_record.get("status") != "complete" or \
                candidate_record.get("status") != "complete":
            raise ValueError(f"Incomplete paired seed: {seed}")
        baseline_metrics = baseline_record["metrics"]
        candidate_metrics = candidate_record["metrics"]
        missing = [
            metric for metric in args.metrics
            if metric not in baseline_metrics or metric not in candidate_metrics
        ]
        if missing:
            raise KeyError(f"Missing metrics at seed {seed}: {missing}")
        baseline_values = {
            metric: float(baseline_metrics[metric]) for metric in args.metrics
        }
        candidate_values = {
            metric: float(candidate_metrics[metric]) for metric in args.metrics
        }
        per_seed.append({
            "seed": seed,
            "baseline_file": str(baseline_path),
            "candidate_file": str(candidate_path),
            args.baseline_label: baseline_values,
            args.candidate_label: candidate_values,
            "candidate_minus_baseline": {
                metric: candidate_values[metric] - baseline_values[metric]
                for metric in args.metrics
            },
        })

    aggregate = {}
    for metric in args.metrics:
        baseline_values = [
            item[args.baseline_label][metric] for item in per_seed
        ]
        candidate_values = [
            item[args.candidate_label][metric] for item in per_seed
        ]
        differences = [
            item["candidate_minus_baseline"][metric] for item in per_seed
        ]
        aggregate[metric] = {
            args.baseline_label: describe(baseline_values),
            args.candidate_label: describe(candidate_values),
            "candidate_minus_baseline": describe(differences),
            "candidate_wins": sum(value < 0 for value in differences),
            "baseline_wins": sum(value > 0 for value in differences),
            "ties": sum(value == 0 for value in differences),
            "paired_t_summary": paired_inference(differences),
        }

    output = {
        "status": "complete",
        "reporting_protocol": (
            "Each fixed-K test seed is evaluated independently; all "
            "differences are candidate minus baseline on paired seeds."
        ),
        "baseline_label": args.baseline_label,
        "candidate_label": args.candidate_label,
        "baseline_manifest": str(baseline_dir / "manifest.json"),
        "candidate_manifest": str(candidate_dir / "manifest.json"),
        "seeds": args.seeds,
        "num_paired_seeds": len(args.seeds),
        "metrics": args.metrics,
        "per_seed": per_seed,
        "aggregate": aggregate,
    }
    atomic_write(args.output, output)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
