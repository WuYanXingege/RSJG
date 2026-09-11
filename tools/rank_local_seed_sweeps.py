#!/usr/bin/env python3
"""Rank completed local seed sweeps without cherry-picking a single seed."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import numpy as np


def parse_candidate(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("candidate must be LABEL=DIRECTORY")
    label, directory = value.split("=", 1)
    if not label or not directory:
        raise argparse.ArgumentTypeError("candidate must be LABEL=DIRECTORY")
    return label, Path(directory).expanduser().resolve()


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def load_records(directory: Path, requested_seeds: list[int] | None) -> list[dict]:
    per_seed = directory / "per_seed"
    paths = sorted(per_seed.glob("seed_*.json"))
    records = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            record = json.load(handle)
        if record.get("status") != "complete":
            continue
        if requested_seeds is None or record["seed"] in requested_seeds:
            records.append(record)
    records.sort(key=lambda item: item["seed"])
    if requested_seeds is not None:
        found = [record["seed"] for record in records]
        if found != sorted(requested_seeds):
            raise ValueError(
                f"{directory}: requested seeds {sorted(requested_seeds)}, found {found}"
            )
    if not records:
        raise ValueError(f"No completed seed records in {directory}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate", action="append", type=parse_candidate, required=True
    )
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["test_minADE@K", "test_minFDE@K", "test_JADE", "test_JFDE"],
    )
    parser.add_argument("--primary-metric", default="test_JADE")
    parser.add_argument("--secondary-metric", default="test_JFDE")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if len({label for label, _ in args.candidate}) != len(args.candidate):
        parser.error("candidate labels must be unique")
    for metric in (args.primary_metric, args.secondary_metric):
        if metric not in args.metrics:
            parser.error(f"ranking metric {metric!r} is absent from --metrics")

    candidates = {}
    for label, directory in args.candidate:
        records = load_records(directory, args.seeds)
        aggregate = {}
        for metric in args.metrics:
            try:
                values = np.asarray(
                    [record["metrics"][metric] for record in records], dtype=float
                )
            except KeyError as exc:
                raise KeyError(f"{directory} is missing metric {metric}") from exc
            aggregate[metric] = {
                "mean": float(values.mean()),
                "sample_std": (
                    float(values.std(ddof=1)) if len(values) > 1 else None
                ),
                "min": float(values.min()),
                "max": float(values.max()),
            }
        candidates[label] = {
            "directory": str(directory),
            "checkpoint_epoch": int(records[0]["checkpoint_epoch"]),
            "seed_count": len(records),
            "seeds": [record["seed"] for record in records],
            "aggregate": aggregate,
        }

    ranking = sorted(
        candidates,
        key=lambda label: (
            candidates[label]["aggregate"][args.primary_metric]["mean"],
            candidates[label]["aggregate"][args.secondary_metric]["mean"],
            label,
        ),
    )
    payload = {
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_policy": (
            f"Lowest mean {args.primary_metric}; {args.secondary_metric} is the "
            "tie-breaker. No best-of-seeds selection is used."
        ),
        "diagnostic_warning": (
            "This local UNIV setup reuses byte-identical validation/test sources; "
            "use the ranking for diagnostics, not as an untouched final test."
        ),
        "metrics": args.metrics,
        "ranking": ranking,
        "selected": {
            "label": ranking[0],
            **candidates[ranking[0]],
        },
        "candidates": candidates,
    }
    atomic_write_json(args.output.expanduser().resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
