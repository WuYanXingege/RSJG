#!/usr/bin/env python3
"""Create compact, reproducible tables from endpoint calibration audits."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from ebjd.diagnostic_bank import atomic_json, sha256_file


METRICS = ("minADE", "minFDE", "JADE", "JFDE")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True,
                        help="label=/path/to/calibration.json")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    metric_rows, strata_rows, k_rows = [], [], []
    bindings = {}
    statuses = {}
    for specification in args.input:
        label, raw_path = specification.split("=", 1)
        path = Path(raw_path).resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        statuses[label] = payload["status"]
        bindings[label] = {
            "result": str(path), "result_sha256": sha256_file(path),
            "bank_manifest": payload["bank_manifest"],
            "bank_manifest_sha256": payload["bank_manifest_sha256"],
            "checkpoint": payload["bank_binding"]["checkpoint"],
            "checkpoint_sha256": payload["bank_binding"]["checkpoint_sha256"],
            "external_binding_verification": payload["external_binding_verification"],
        }
        for result in payload["results"]:
            metrics = result["metrics"]
            coverage = result["coverage"]
            metric_rows.append({
                "epoch": label, "gamma": result["factor"],
                **{key: metrics[key] for key in METRICS},
                **coverage,
                "candidate_mean_max_abs_m": result["invariants"][
                    "candidate_mean_max_abs_m"],
                "bridge_residual_max_abs_m": result["invariants"][
                    "bridge_residual_max_abs_m"],
                "radius_ratio_abs_error": result["invariants"][
                    "endpoint_rms_radius_ratio_abs_error"],
                "all_four_improve_vs_original": result["comparison"][
                    "all_four_improve_vs_original"],
                "all_four_reach_matched_GDTS": result["comparison"][
                    "all_four_reach_matched_GDTS"],
            })
            for stratum, values in result["strata"].items():
                strata_rows.append({
                    "epoch": label, "gamma": result["factor"],
                    "stratum": stratum, "scene_count": values["scene_count"],
                    "agent_count": values["agent_count"],
                    **{key: values[key] for key in METRICS},
                })
            for values in result["k_curve"]:
                k_rows.append({
                    "epoch": label, "gamma": result["factor"], "K": values["K"],
                    **{key: values[key] for key in METRICS},
                })
    _write_csv(output / "FULL_METRICS.csv", metric_rows)
    _write_csv(output / "STRATA.csv", strata_rows)
    _write_csv(output / "K_CURVES.csv", k_rows)
    overall = "PARTIAL_RECOVERY" if all(
        value == "PARTIAL_RECOVERY" for value in statuses.values()) else "MIXED"
    atomic_json({
        "schema": "ebjd-endpoint-calibration-summary-v1",
        "status": overall, "optimizer_updates": 0,
        "epochs": list(statuses), "per_epoch_status": statuses,
        "bindings": bindings,
        "tables": {
            "full_metrics": str((output / "FULL_METRICS.csv").resolve()),
            "strata": str((output / "STRATA.csv").resolve()),
            "k_curves": str((output / "K_CURVES.csv").resolve()),
        },
    }, output / "SUMMARY.json")


if __name__ == "__main__":
    main()
