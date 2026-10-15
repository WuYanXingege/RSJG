#!/usr/bin/env python3
"""Freeze a fold result comparison and produce JSON/CSV/Chinese Markdown."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import tempfile
from pathlib import Path

import torch


METRICS = ("minADE", "minFDE", "JADE", "JFDE")


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(payload: dict, path: Path) -> None:
    atomic_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--selected-checkpoint", required=True)
    parser.add_argument("--ebjd-result", required=True)
    parser.add_argument("--gdts-result", required=True)
    parser.add_argument("--archive-dir", required=True)
    args = parser.parse_args()
    fold = args.fold.lower()
    prefix = fold.upper()
    run_dir = Path(args.run_dir).resolve()
    selected = Path(args.selected_checkpoint).resolve()
    archive = Path(args.archive_dir).resolve()
    selection = json.loads((run_dir / "SELECTION_STATUS.json").read_text())
    ebjd = json.loads(Path(args.ebjd_result).read_text())
    gdts = json.loads(Path(args.gdts_result).read_text())
    checkpoint = torch.load(selected, map_location="cpu", weights_only=False)
    if int(checkpoint["epoch"]) != int(
            (selection["best_eligible"] or selection["least_violation"])["epoch"]):
        raise RuntimeError("frozen checkpoint epoch differs from preregistered selector")
    final_epoch_summary = json.loads(
        (run_dir / "EPOCH_SUMMARY.jsonl").read_text().splitlines()[-1])
    if int(final_epoch_summary["epoch"]) != 100:
        # Kept deliberately strict; a partial run must never be finalized.
        raise RuntimeError(f"formal {prefix} training did not reach epoch 100")
    if int(checkpoint["resolved_config"]["training"]["epochs"]) != 100:
        raise RuntimeError("selected checkpoint does not retain the 100-epoch schedule")
    if ebjd["checkpoint_sha256"] != sha256(selected):
        raise RuntimeError("formal evaluation is not bound to the selected checkpoint")
    manifest_path = Path(checkpoint["resolved_config"]["paths"]["test_manifest"])
    manifest = json.loads(manifest_path.read_text())
    expected_scenes = int(manifest["statistics"]["scenes"])
    expected_agents = int(manifest["statistics"]["agents"])
    if (ebjd["scene_count"] != expected_scenes
            or ebjd["agent_occurrence_count"] != expected_agents):
        raise RuntimeError(f"formal evaluation has incomplete {prefix} coverage")
    if [item["seed"] for item in ebjd["per_seed"]] != list(range(2035, 2040)):
        raise RuntimeError("EBJD inference seeds are incomplete or reordered")
    gdts_by_seed = {int(item["seed"]): item for item in gdts["runs"]}
    ebjd_by_seed = {int(item["seed"]): item["metrics"] for item in ebjd["per_seed"]}
    rows = []
    for seed in range(2035, 2040):
        for metric in METRICS:
            gdts_key = f"{metric}@K" if metric in ("minADE", "minFDE") else metric
            baseline = float(gdts_by_seed[seed][gdts_key])
            candidate = float(ebjd_by_seed[seed][metric])
            rows.append({
                "seed": seed, "metric": metric, "GDTS": baseline,
                "EBJD": candidate, "EBJD_minus_GDTS": candidate - baseline,
                "relative_change_percent": 100 * (candidate / baseline - 1),
                "EBJD_better": candidate < baseline,
            })
    means = {}
    for metric in METRICS:
        metric_rows = [row for row in rows if row["metric"] == metric]
        baseline_values = [row["GDTS"] for row in metric_rows]
        candidate_values = [row["EBJD"] for row in metric_rows]
        differences = [row["EBJD_minus_GDTS"] for row in metric_rows]
        means[metric] = {
            "GDTS_mean": statistics.mean(baseline_values),
            "GDTS_sample_std": statistics.stdev(baseline_values),
            "EBJD_mean": statistics.mean(candidate_values),
            "EBJD_sample_std": statistics.stdev(candidate_values),
            "mean_difference": statistics.mean(differences),
            "relative_change_percent": 100 * (
                statistics.mean(candidate_values) / statistics.mean(baseline_values) - 1),
            "better_seed_count": sum(value < 0 for value in differences),
        }
    joint_gain = all(means[name]["mean_difference"] < 0 for name in ("JADE", "JFDE"))
    marginal_degradation = all(
        means[name]["mean_difference"] > 0 for name in ("minADE", "minFDE"))
    objective_passed = selection["best_eligible"] is not None
    result = {
        "schema": f"ebjd-{fold}-main-final-comparison-v1", "status": "COMPLETED",
        "fold": fold, "training_seed": 3101,
        "inference_seeds": list(range(2035, 2040)),
        "selected_checkpoint": str(selected),
        "selected_checkpoint_sha256": sha256(selected),
        "selected_epoch": int(checkpoint["epoch"]),
        "selection_basis": (
            "best_eligible" if objective_passed else "least_violation"),
        "selection": selection,
        "per_seed": rows, "summary": means,
        "conclusions": {
            f"joint_gain_reproduced_on_{fold}": joint_gain,
            f"marginal_degradation_reproduced_on_{fold}": marginal_degradation,
            "marginal_preserving_selector_objective_passed": objective_passed,
            "scope": (
                f"{prefix} fold, one independent training seed; five inference seeds "
                "measure inference randomness only"),
        },
        "protocol": {
            "worlds": ebjd["worlds"], "steps": ebjd["steps"],
            "unit": "m", "lower_is_better": True,
            "independent_test": False,
            "validation_test_identity": ebjd["validation_test_identity"],
            "cross_method_rng_claim": "same seed labels, not identical random streams",
        },
        "sources": {
            "ebjd_result": str(Path(args.ebjd_result).resolve()),
            "ebjd_result_sha256": sha256(args.ebjd_result),
            "gdts_result": str(Path(args.gdts_result).resolve()),
            "gdts_result_sha256": sha256(args.gdts_result),
        },
    }
    atomic_json(result, archive / f"{prefix}_FINAL_COMPARISON.json")

    csv_path = archive / f"{prefix}_FINAL_COMPARISON.csv"
    header = list(rows[0])
    from io import StringIO
    stream = StringIO()
    writer = csv.DictWriter(stream, fieldnames=header)
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(stream.getvalue(), csv_path)

    outcome = lambda value: "是" if value else "否"
    report = [
        f"# EBJD {prefix} seed 3101 正式训练与评测结果", "",
        f"- 状态：`COMPLETED`；训练 100 epoch，训练种子 3101。",
        f"- 正式 checkpoint：epoch {checkpoint['epoch']}，选择依据 "
        f"`{result['selection_basis']}`，SHA256 `{sha256(selected)}`。",
        "- 推理：2035–2039，20 worlds × 20 DDIM steps，batch=1、FP32、无 autocast。",
        "- validation/test 为官方镜像，因此 `independent_test=false`。", "",
        "| 指标 | GDTS 均值±样本标准差 | EBJD 均值±样本标准差 | EBJD 相对变化 | 改善种子数 |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric in METRICS:
        row = means[metric]
        report.append(
            f"| {metric} | {row['GDTS_mean']:.6f} ± {row['GDTS_sample_std']:.6f} "
            f"| {row['EBJD_mean']:.6f} ± {row['EBJD_sample_std']:.6f} "
            f"| {row['relative_change_percent']:+.2f}% | {row['better_seed_count']}/5 |")
    report.extend([
        "", "## 预注册问题", "",
        f"- 联合收益是否在 {prefix} 复现：**{outcome(joint_gain)}**。",
        f"- 边际退化是否在 {prefix} 复现：**{outcome(marginal_degradation)}**。",
        f"- 原 marginal-preserving 选模目标是否通过：**{outcome(objective_passed)}**。",
        "", f"结论只覆盖当前已完成的折；{prefix} 仍只有一个独立训练种子，"
        "五个推理种子的离散度不能作为多训练种子显著性证据。", "",
    ])
    atomic_text("\n".join(report), archive / f"{prefix}_FINAL_REPORT_ZH.md")
    print(json.dumps({
        "status": "COMPLETED", "archive": str(archive),
        "selected_epoch": int(checkpoint["epoch"]),
        "selected_checkpoint_sha256": sha256(selected),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
