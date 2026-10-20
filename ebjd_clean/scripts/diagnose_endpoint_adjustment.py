#!/usr/bin/env python3
"""CPU-only GT-blind endpoint-set adjustment diagnostic on frozen banks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from ebjd.metrics import MetricAccumulator


METRICS = ("minADE", "minFDE", "JADE", "JFDE")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.bank_root).resolve()
    per_seed = []
    maximum_bridge_error = 0.0
    all_permutations = True
    bank_hashes = {}
    time = np.arange(1, 13, dtype=np.float64) / 12
    for seed in range(2035, 2040):
        path = root / f"seed_{seed}.npz"
        bank_hashes[str(seed)] = sha256(path)
        bank = np.load(path, allow_pickle=True)
        accumulator = MetricAccumulator()
        strata = {"N=1": MetricAccumulator(), "N>=2": MetricAccumulator()}
        for index in range(len(bank["scene_ids"])):
            prediction = np.asarray(bank["predictions"][index], dtype=np.float64)
            future = np.asarray(bank["future"][index], dtype=np.float64)
            observed = np.asarray(bank["observed"][index], dtype=np.float64)
            native = np.asarray(bank["native_goals"][index], dtype=np.float64)
            worlds, agents = prediction.shape[:2]
            if worlds != 20 or native.shape != (agents, 20, 2):
                raise ValueError("endpoint adjustment requires paired P20 banks")
            adjusted_goal = np.empty((worlds, agents, 2), dtype=np.float64)
            for agent in range(agents):
                distance = np.linalg.norm(
                    prediction[:, agent, -1, None] - native[agent][None], axis=-1)
                rows, columns = linear_sum_assignment(distance)
                permutation = np.empty(20, dtype=np.int64)
                permutation[rows] = columns
                all_permutations = all_permutations and np.array_equal(
                    np.sort(permutation), np.arange(20))
                adjusted_goal[:, agent] = native[agent, permutation]
            delta = adjusted_goal - prediction[..., -1, :]
            adjusted = prediction + time[None, None, :, None] * delta[:, :, None]
            origin = observed[:, -1]
            old_line = (origin[None, :, None]
                        + time[None, None, :, None]
                        * (prediction[..., -1, :] - origin[None])[:, :, None])
            new_line = (origin[None, :, None]
                        + time[None, None, :, None]
                        * (adjusted_goal - origin[None])[:, :, None])
            maximum_bridge_error = max(maximum_bridge_error, float(np.max(np.abs(
                (prediction - old_line) - (adjusted - new_line)))))
            pred_tensor = torch.from_numpy(adjusted).float()[None]
            future_tensor = torch.from_numpy(future).float()[None]
            valid = torch.ones(1, agents, dtype=torch.bool)
            accumulator.update(pred_tensor, future_tensor, valid)
            strata["N=1" if agents == 1 else "N>=2"].update(
                pred_tensor, future_tensor, valid)
        per_seed.append({
            "seed": seed, "metrics": accumulator.as_dict(),
            "strata": {name: value.as_dict() for name, value in strata.items()
                       if value.scene_count},
        })
    mean = {metric: float(statistics.fmean(
        row["metrics"][metric] for row in per_seed)) for metric in METRICS}
    standard_deviation = {metric: float(statistics.stdev(
        row["metrics"][metric] for row in per_seed)) for metric in METRICS}
    receipt = {
        "schema": "ebjd-endpoint-set-adjustment-diagnostic-v1",
        "status": "PASS" if all_permutations and maximum_bridge_error <= 1e-5 else "FAIL",
        "matching_uses_future_ground_truth": False,
        "matching": "per-agent Hungarian between old predicted endpoints and native P20",
        "candidate_set_preserved_exactly": all_permutations,
        "maximum_bridge_residual_absolute_error": maximum_bridge_error,
        "bank_sha256_by_seed": bank_hashes,
        "per_seed": per_seed, "mean": mean,
        "sample_standard_deviation": standard_deviation,
        "interpretation_boundary": (
            "minFDE recovery is constructed by endpoint replacement; ADE and joint metrics "
            "test whether the old bridge remains compatible with the reassigned endpoints."),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps({"status": receipt["status"], "mean": mean}, sort_keys=True))
    if receipt["status"] != "PASS":
        raise RuntimeError("endpoint adjustment invariants failed")


if __name__ == "__main__":
    main()
