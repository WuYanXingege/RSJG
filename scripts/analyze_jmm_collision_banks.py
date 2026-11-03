#!/usr/bin/env python3
"""Aggregate frozen GDTS/G/H full-trajectory banks with official JMM CR."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from ebjd.collision_metrics import ground_truth_collision_rate, scene_collision_statistics


FOLDS = ("eth", "hotel", "univ", "zara1", "zara2")
SEEDS = (2035, 2036, 2037, 2038, 2039)
METHODS = ("GDTS", "G", "H")
METRICS = ("CRJADE", "CRmean")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_bank(path: Path, method: str):
    with np.load(path, allow_pickle=True) as data:
        metadata = json.loads(str(data["metadata"].item()))
        if method in {"G", "H"}:
            predictions = list(data[f"predictions_{method}"])
        else:
            predictions = list(data["predictions"])
        return {
            "metadata": metadata,
            "scene_ids": [str(item) for item in data["scene_ids"]],
            "agent_ids": list(data["agent_ids"]),
            "predictions": predictions,
            "targets": list(data["targets"]),
            "sha256": sha256(path),
            "path": str(path.resolve()),
        }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def subset_name(n: int) -> tuple[str, ...]:
    return ("all", "N1" if n == 1 else "Nge2")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank-root", required=True)
    parser.add_argument("--formal-metrics", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    bank_root, output = Path(args.bank_root), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    formal = {}
    with Path(args.formal_metrics).open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            formal[(row["fold"].lower(), int(row["seed"]), row["method"])] = {
                key: float(row[key]) for key in ("minADE", "minFDE", "JADE", "JFDE")}

    scene_rows, seed_rows, gt_scene_rows, sources = [], [], [], []
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    gt_seen: set[tuple[str, str]] = set()
    for fold in FOLDS:
        for seed in SEEDS:
            directory = bank_root / fold / f"seed_{seed}"
            gdts_options = sorted(directory.glob("GDTS.accepted.npz"))
            if not gdts_options:
                gdts_options = sorted(directory.glob("GDTS.exact.npz"))
            if not gdts_options:
                gdts_options = sorted(directory.glob("GDTS.stream.npz"))
            gh_path = directory / "G_H.npz"
            if not gdts_options or not gh_path.exists():
                raise FileNotFoundError(f"missing complete banks for {fold} seed {seed}")
            paths = {"GDTS": gdts_options[0], "G": gh_path, "H": gh_path}
            for method in METHODS:
                bank = load_bank(paths[method], method)
                if len(bank["scene_ids"]) != len(bank["predictions"]):
                    raise ValueError("bank scene arrays disagree")
                expected = formal[(fold, seed, method)]
                metadata_metrics = bank["metadata"].get("displacement_metrics")
                # G/H banks are verified below by direct recomputation.
                ade, fde, jade, jfde = [], [], [], []
                for scene_id, agent_ids, prediction, target in zip(
                        bank["scene_ids"], bank["agent_ids"], bank["predictions"], bank["targets"]):
                    prediction = np.asarray(prediction, dtype=np.float32)
                    target = np.asarray(target, dtype=np.float32)
                    if prediction.shape != (20, len(agent_ids), 12, 2):
                        raise ValueError(f"invalid prediction shape: {fold}/{seed}/{method}/{scene_id}")
                    distance = np.linalg.norm(prediction - target[None], axis=-1)
                    ade.extend(distance.mean(-1).min(0).tolist())
                    fde.extend(distance[..., -1].min(0).tolist())
                    jade.append(float(distance.mean((1, 2)).min()))
                    jfde.append(float(distance[..., -1].mean(1).min()))
                    stats = scene_collision_statistics(
                        torch.from_numpy(prediction), torch.from_numpy(target))
                    n = len(agent_ids)
                    for subset in subset_name(n):
                        grouped[(fold, seed, method, subset)].append({
                            "CRJADE": stats.cr_jade, "CRmean": stats.cr_mean,
                            "jade_num": stats.collision_numerator_jade,
                            "jade_den": stats.collision_denominator_jade,
                            "mean_num": stats.collision_numerator_mean,
                            "mean_den": stats.collision_denominator_mean,
                        })
                    scene_rows.append({
                        "fold": fold.upper(), "seed": seed, "method": method,
                        "scene_id": scene_id, "valid_agents": n,
                        "jade_world_index": stats.jade_world_index,
                        "CRJADE_ratio": f"{stats.cr_jade:.12g}",
                        "CRmean_ratio": f"{stats.cr_mean:.12g}",
                        "CRJADE_collision_agents": stats.collision_numerator_jade,
                        "CRJADE_agent_denominator": stats.collision_denominator_jade,
                        "CRmean_collision_agent_worlds": stats.collision_numerator_mean,
                        "CRmean_agent_world_denominator": stats.collision_denominator_mean,
                        "bank_sha256": bank["sha256"],
                    })
                    gt_key = (fold, scene_id)
                    if gt_key not in gt_seen and method == "G" and seed == 2035:
                        rate, numerator, denominator = ground_truth_collision_rate(
                            torch.from_numpy(target))
                        gt_seen.add(gt_key)
                        gt_scene_rows.append({
                            "fold": fold.upper(), "scene_id": scene_id,
                            "valid_agents": n, "GT_CR_ratio": f"{rate:.12g}",
                            "collision_agents": numerator,
                            "agent_denominator": denominator,
                        })
                replay = {
                    "minADE": float(np.mean(ade)), "minFDE": float(np.mean(fde)),
                    "JADE": float(np.mean(jade)), "JFDE": float(np.mean(jfde)),
                }
                maximum = max(abs(replay[key] - expected[key]) for key in replay)
                if maximum > 2e-6:
                    raise RuntimeError(
                        f"formal metric replay mismatch {fold}/{seed}/{method}: {maximum} {replay} {expected}")
                if metadata_metrics is not None:
                    meta_max = max(abs(float(metadata_metrics[key]) - expected[key]) for key in replay)
                    if meta_max > 2e-6:
                        raise RuntimeError(f"bank receipt mismatch {fold}/{seed}/{method}: {meta_max}")
                sources.append({
                    "fold": fold, "seed": seed, "method": method,
                    "path": bank["path"], "sha256": bank["sha256"],
                    "checkpoint_sha256": bank["metadata"].get("checkpoint_sha256"),
                    "manifest_sha256": bank["metadata"].get("manifest_sha256"),
                    "scene_count": len(bank["scene_ids"]),
                    "agent_occurrence_count": sum(len(x) for x in bank["agent_ids"]),
                    "formal_replay_max_abs": maximum,
                    "elapsed_seconds": bank["metadata"].get("elapsed_seconds"),
                })

    for (fold, seed, method, subset), values in sorted(grouped.items()):
        row = {"fold": fold.upper(), "seed": seed, "method": method, "subset": subset,
               "scene_count": len(values)}
        for metric in METRICS:
            scores = [value[metric] for value in values]
            row[f"{metric}_ratio"] = f"{np.mean(scores):.12g}"
            row[f"{metric}_percent"] = f"{100*np.mean(scores):.12g}"
            prefix = "jade" if metric == "CRJADE" else "mean"
            row[f"{metric}_scene_rate_sum"] = f"{sum(scores):.12g}"
            row[f"{metric}_collision_numerator_audit"] = sum(v[f"{prefix}_num"] for v in values)
            row[f"{metric}_agent_denominator_audit"] = sum(v[f"{prefix}_den"] for v in values)
        seed_rows.append(row)

    summary_rows = []
    lookup = {(r["fold"].lower(), int(r["seed"]), r["method"], r["subset"]): r
              for r in seed_rows}
    for subset in ("all", "N1", "Nge2"):
        for scope in tuple(name.upper() for name in FOLDS) + ("MACRO",):
            for method in METHODS:
                for metric in METRICS:
                    values = []
                    contributing_folds = set()
                    for seed in SEEDS:
                        folds = FOLDS if scope == "MACRO" else (scope.lower(),)
                        per_fold = []
                        for fold in folds:
                            item = lookup.get((fold, seed, method, subset))
                            if item is not None:
                                per_fold.append(float(item[f"{metric}_ratio"]))
                                contributing_folds.add(fold)
                        if per_fold:
                            values.append(float(np.mean(per_fold)))
                    summary_rows.append({
                        "scope": scope, "subset": subset, "method": method,
                        "metric": metric, "contributing_folds": len(contributing_folds),
                        "available_seeds": len(values),
                        "mean_ratio": "" if not values else f"{np.mean(values):.12g}",
                        "sd_ddof1_ratio": "" if len(values) < 2 else f"{np.std(values, ddof=1):.12g}",
                        "mean_percent": "" if not values else f"{100*np.mean(values):.12g}",
                        "sd_ddof1_percent": "" if len(values) < 2 else f"{100*np.std(values, ddof=1):.12g}",
                    })

    gt_summary = []
    for fold in FOLDS:
        rows = [r for r in gt_scene_rows if r["fold"] == fold.upper()]
        for subset in ("all", "N1", "Nge2"):
            selected = rows if subset == "all" else [
                r for r in rows if (r["valid_agents"] == 1) == (subset == "N1")]
            rates = [float(r["GT_CR_ratio"]) for r in selected]
            gt_summary.append({
                "fold": fold.upper(), "subset": subset, "scene_count": len(selected),
                "GT_CR_ratio": None if not rates else float(np.mean(rates)),
                "GT_CR_percent": None if not rates else 100*float(np.mean(rates)),
            })
    write_csv(output / "PER_SCENE_COLLISION_METRICS.csv", scene_rows)
    write_csv(output / "PER_SEED_COLLISION_METRICS.csv", seed_rows)
    write_csv(output / "FIVEFOLD_COLLISION_SUMMARY.csv", summary_rows)
    write_csv(output / "GROUND_TRUTH_PER_SCENE.csv", gt_scene_rows)
    (output / "SOURCE_BANKS.json").write_text(
        json.dumps({"schema": "ebjd-jmm-source-banks-v1", "banks": sources}, indent=2) + "\n")
    (output / "GROUND_TRUTH_COLLISION_SUMMARY.json").write_text(
        json.dumps({"schema": "ebjd-jmm-gt-cr-v1", "summary": gt_summary}, indent=2) + "\n")
    (output / "COLLISION_RESULTS.json").write_text(json.dumps({
        "schema": "ebjd-jmm-collision-results-v1", "folds": list(FOLDS),
        "seeds": list(SEEDS), "methods": list(METHODS), "threshold_m": 0.2,
        "summary": summary_rows,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
