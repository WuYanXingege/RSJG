#!/usr/bin/env python3
"""Compare paired JMM runs on single- and multi-agent scene subsets."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import sys

import numpy as np
from scipy.stats import t as student_t


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.jmm_protocol import (  # noqa: E402
    JMM_ETH_EXPECTED_AGENT_INSTANCES,
    JMM_ETH_EXPECTED_WINDOWS,
    JMM_ETH_SEQUENCE,
    JMM_NUM_SAMPLES,
    JMM_OBS_LEN,
    JMM_PRED_LEN,
    _ids_in_order,
    _integer_column,
    _load_standard_table,
    _table_to_tensor,
)


METRICS = ("minADE@20", "minFDE@20", "minJADE@20", "minJFDE@20")
SPLITS = ("all", "single_agent", "multi_agent")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-template", required=True)
    parser.add_argument("--candidate-template", required=True)
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def describe(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def paired_inference(differences: list[float]) -> dict[str, object]:
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


def _scene_metrics(scene_dir: Path) -> tuple[int, dict[str, float]]:
    obs_table = _load_standard_table(scene_dir / "obs.txt")
    gt_table = _load_standard_table(scene_dir / "gt.txt")
    agent_ids = _ids_in_order(gt_table)
    if not np.array_equal(np.sort(agent_ids), np.sort(_ids_in_order(obs_table))):
        raise ValueError(f"obs/gt agent IDs differ in {scene_dir}")
    obs_frames = np.unique(_integer_column(obs_table[:, 0], "obs frame_id"))
    gt_frames = np.unique(_integer_column(gt_table[:, 0], "gt frame_id"))
    if obs_frames.size != JMM_OBS_LEN or gt_frames.size != JMM_PRED_LEN:
        raise ValueError(f"Unexpected frame count in {scene_dir}")
    _table_to_tensor(obs_table, obs_frames, agent_ids, source=scene_dir / "obs.txt")
    ground_truth = _table_to_tensor(
        gt_table, gt_frames, agent_ids, source=scene_dir / "gt.txt"
    )
    samples = []
    for sample_index in range(JMM_NUM_SAMPLES):
        sample_path = scene_dir / f"sample_{sample_index:03d}.txt"
        sample_table = _load_standard_table(sample_path)
        samples.append(
            _table_to_tensor(
                sample_table, gt_frames, agent_ids, source=sample_path
            )
        )
    predictions = np.stack(samples, axis=0)
    distances = np.linalg.norm(predictions - ground_truth[None], axis=-1)
    per_sample_agent_ade = distances.mean(axis=1)
    per_sample_agent_fde = distances[:, -1]
    return int(agent_ids.size), {
        "joint_ade": float(per_sample_agent_ade.mean(axis=1).min()),
        "joint_fde": float(per_sample_agent_fde.mean(axis=1).min()),
        "marginal_ade_sum": float(per_sample_agent_ade.min(axis=0).sum()),
        "marginal_fde_sum": float(per_sample_agent_fde.min(axis=0).sum()),
    }


def score_splits(trajectory_root: Path) -> dict[str, dict[str, float | int]]:
    sequence_root = trajectory_root.expanduser().resolve() / JMM_ETH_SEQUENCE
    scene_dirs = sorted(
        path for path in sequence_root.iterdir()
        if path.is_dir() and path.name.startswith("frame_")
    )
    if len(scene_dirs) != JMM_ETH_EXPECTED_WINDOWS:
        raise ValueError(
            f"Expected {JMM_ETH_EXPECTED_WINDOWS} scenes, found {len(scene_dirs)}"
        )
    accumulators = {
        split: {
            "joint_ade": [],
            "joint_fde": [],
            "marginal_ade_sum": 0.0,
            "marginal_fde_sum": 0.0,
            "num_scenes": 0,
            "num_agent_instances": 0,
        }
        for split in SPLITS
    }
    for scene_dir in scene_dirs:
        num_agents, values = _scene_metrics(scene_dir)
        selected_splits = (
            ("all", "single_agent") if num_agents == 1
            else ("all", "multi_agent")
        )
        for split in selected_splits:
            accumulator = accumulators[split]
            accumulator["joint_ade"].append(values["joint_ade"])
            accumulator["joint_fde"].append(values["joint_fde"])
            accumulator["marginal_ade_sum"] += values["marginal_ade_sum"]
            accumulator["marginal_fde_sum"] += values["marginal_fde_sum"]
            accumulator["num_scenes"] += 1
            accumulator["num_agent_instances"] += num_agents

    if accumulators["all"]["num_agent_instances"] != JMM_ETH_EXPECTED_AGENT_INSTANCES:
        raise ValueError("Official JMM agent-instance count does not match")
    results = {}
    for split, accumulator in accumulators.items():
        num_agents = accumulator["num_agent_instances"]
        num_scenes = accumulator["num_scenes"]
        results[split] = {
            "minADE@20": accumulator["marginal_ade_sum"] / num_agents,
            "minFDE@20": accumulator["marginal_fde_sum"] / num_agents,
            "minJADE@20": float(np.mean(accumulator["joint_ade"])),
            "minJFDE@20": float(np.mean(accumulator["joint_fde"])),
            "num_scenes": num_scenes,
            "num_agent_instances": num_agents,
            "mean_pedestrians_per_scene": num_agents / num_scenes,
        }
    return results


def main() -> None:
    args = parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Seeds must be unique")

    per_seed = []
    reference_counts = None
    for seed in args.seeds:
        baseline_root = Path(args.baseline_template.format(seed=seed))
        candidate_root = Path(args.candidate_template.format(seed=seed))
        baseline = score_splits(baseline_root)
        candidate = score_splits(candidate_root)
        counts = {
            split: {
                name: baseline[split][name]
                for name in (
                    "num_scenes",
                    "num_agent_instances",
                    "mean_pedestrians_per_scene",
                )
            }
            for split in SPLITS
        }
        candidate_counts = {
            split: {
                name: candidate[split][name]
                for name in counts[split]
            }
            for split in SPLITS
        }
        if counts != candidate_counts:
            raise ValueError(f"Baseline/candidate scene counts differ at seed {seed}")
        if reference_counts is None:
            reference_counts = counts
        elif counts != reference_counts:
            raise ValueError(f"Scene counts changed at seed {seed}")
        per_seed.append({
            "seed": seed,
            "splits": {
                split: {
                    args.baseline_label: {
                        name: baseline[split][name] for name in METRICS
                    },
                    args.candidate_label: {
                        name: candidate[split][name] for name in METRICS
                    },
                    "candidate_minus_baseline": {
                        name: candidate[split][name] - baseline[split][name]
                        for name in METRICS
                    },
                }
                for split in SPLITS
            },
        })

    aggregate = {}
    for split in SPLITS:
        aggregate[split] = {}
        for metric in METRICS:
            baseline_values = [
                item["splits"][split][args.baseline_label][metric]
                for item in per_seed
            ]
            candidate_values = [
                item["splits"][split][args.candidate_label][metric]
                for item in per_seed
            ]
            differences = [
                item["splits"][split]["candidate_minus_baseline"][metric]
                for item in per_seed
            ]
            aggregate[split][metric] = {
                args.baseline_label: describe(baseline_values),
                args.candidate_label: describe(candidate_values),
                "candidate_minus_baseline": describe(differences),
                "paired_t_summary": paired_inference(differences),
                "candidate_wins": sum(value < 0 for value in differences),
                "baseline_wins": sum(value > 0 for value in differences),
                "ties": sum(value == 0 for value in differences),
            }

    payload = {
        "status": "complete",
        "diagnostic_only": True,
        "reporting_protocol": (
            "Official JMM scenes are split by agent count; each seed remains "
            "K=20 and is scored independently before paired aggregation."
        ),
        "baseline_label": args.baseline_label,
        "candidate_label": args.candidate_label,
        "seeds": args.seeds,
        "scene_counts": reference_counts,
        "per_seed": per_seed,
        "aggregate": aggregate,
    }
    atomic_write(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
