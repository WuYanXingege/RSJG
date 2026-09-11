#!/usr/bin/env python3
"""Summarize paired, fixed-K JMM evaluations across inference seeds."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics

from scipy.stats import t as student_t


METRICS = ("minADE@20", "minFDE@20", "minJADE@20", "minJFDE@20")
PROTOCOL_FIELDS = (
    "num_scenes",
    "num_agent_instances",
    "mean_pedestrians_per_scene",
    "protocol",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-template", required=True)
    parser.add_argument("--candidate-template", required=True)
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_metrics(template: str, seed: int) -> tuple[Path, dict]:
    path = Path(template.format(seed=seed)).expanduser().resolve()
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    missing = [name for name in (*METRICS, *PROTOCOL_FIELDS) if name not in payload]
    if missing:
        raise KeyError(f"Missing fields in {path}: {missing}")
    return path, payload


def describe(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def paired_inference(differences: list[float]) -> dict[str, float | list[float]]:
    """Two-sided paired t summary for candidate-minus-baseline differences."""
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
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Seeds must be unique")

    per_seed = []
    reference_protocol = None
    for seed in args.seeds:
        baseline_path, baseline = load_metrics(args.baseline_template, seed)
        candidate_path, candidate = load_metrics(args.candidate_template, seed)
        baseline_protocol = {name: baseline[name] for name in PROTOCOL_FIELDS}
        candidate_protocol = {name: candidate[name] for name in PROTOCOL_FIELDS}
        if baseline_protocol != candidate_protocol:
            raise ValueError(f"Protocol mismatch for seed {seed}")
        if reference_protocol is None:
            reference_protocol = baseline_protocol
        elif reference_protocol != baseline_protocol:
            raise ValueError(f"Protocol changed at seed {seed}")
        per_seed.append({
            "seed": seed,
            "baseline_metrics_file": str(baseline_path),
            "candidate_metrics_file": str(candidate_path),
            args.baseline_label: {name: float(baseline[name]) for name in METRICS},
            args.candidate_label: {name: float(candidate[name]) for name in METRICS},
            "candidate_minus_baseline": {
                name: float(candidate[name] - baseline[name]) for name in METRICS
            },
        })

    aggregates = {}
    for metric in METRICS:
        baseline_values = [item[args.baseline_label][metric] for item in per_seed]
        candidate_values = [item[args.candidate_label][metric] for item in per_seed]
        differences = [
            item["candidate_minus_baseline"][metric] for item in per_seed
        ]
        aggregates[metric] = {
            args.baseline_label: describe(baseline_values),
            args.candidate_label: describe(candidate_values),
            "candidate_minus_baseline": describe(differences),
            "paired_t_summary": paired_inference(differences),
            "candidate_wins": sum(value < 0 for value in differences),
            "baseline_wins": sum(value > 0 for value in differences),
            "ties": sum(value == 0 for value in differences),
        }

    output = {
        "status": "complete",
        "reporting_protocol": (
            "Each seed is scored independently at K=20; means and sample "
            "standard deviations are computed across paired seeds."
        ),
        "baseline_label": args.baseline_label,
        "candidate_label": args.candidate_label,
        "seeds": args.seeds,
        "num_paired_seeds": len(args.seeds),
        "protocol": reference_protocol,
        "per_seed": per_seed,
        "aggregate": aggregates,
    }
    atomic_write(args.output, output)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
