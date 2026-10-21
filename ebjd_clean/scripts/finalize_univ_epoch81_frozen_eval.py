#!/usr/bin/env python3
"""Validate and archive the frozen UNIV epoch81 five-seed evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import statistics
from pathlib import Path


METRICS = ("minADE", "minFDE", "JADE", "JFDE")
SEEDS = tuple(range(2035, 2040))
CONTROL_SHA = "e9fd76be585fef0d9cf6ab82422e5a5775392030c59283054a5247a6f47eafdd"
CANDIDATE_SHA = "8d695b9e494674610cc51a6719ee6227363ddc158846cb3309d1fabe32624a0e"
MANIFEST_SHA = "46c52540aac6f0313865f01f53d7c735d91cb8116fdbee39cb65f0cf545c7d8e"


def read(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value)
    os.replace(temporary, path)


def metric_table(payload: dict) -> dict:
    by_seed = {int(row["seed"]): row["metrics"] for row in payload["per_seed"]}
    if tuple(sorted(by_seed)) != SEEDS:
        raise ValueError("formal result does not contain exactly seeds 2035-2039")
    return {
        metric: {
            "mean": float(payload["mean"][metric]),
            "sample_standard_deviation": float(
                payload["sample_standard_deviation"][metric]),
            "per_seed": [float(by_seed[seed][metric]) for seed in SEEDS],
        }
        for metric in METRICS
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--control-result", type=Path, required=True)
    parser.add_argument("--candidate-raw-result", type=Path, required=True)
    parser.add_argument("--candidate-fixed-result", type=Path, required=True)
    parser.add_argument("--parent-result", type=Path, required=True)
    parser.add_argument("--gdts-comparison", type=Path, required=True)
    parser.add_argument("--control-config", type=Path, required=True)
    parser.add_argument("--candidate-config", type=Path, required=True)
    args = parser.parse_args()

    payloads = {
        "parent_epoch73_raw": read(args.parent_result),
        "control_epoch81_raw": read(args.control_result),
        "candidate_epoch81_raw": read(args.candidate_raw_result),
        "candidate_epoch81_fixed": read(args.candidate_fixed_result),
    }
    tables = {name: metric_table(value) for name, value in payloads.items()}
    gdts_source = read(args.gdts_comparison)
    gdts = {}
    for metric in METRICS:
        rows = sorted(
            (row for row in gdts_source["per_seed"] if row["metric"] == metric),
            key=lambda row: row["seed"])
        values = [float(row["GDTS"]) for row in rows]
        if [int(row["seed"]) for row in rows] != list(SEEDS):
            raise ValueError("matched GDTS seeds are incomplete")
        gdts[metric] = {
            "mean": statistics.mean(values),
            "sample_standard_deviation": statistics.stdev(values),
            "per_seed": values,
        }
    tables["matched_gdts"] = gdts

    receipt_dirs = {
        "control": args.asset_root / "receipts" / "control",
        "candidate": args.asset_root / "receipts" / "candidate",
    }
    receipt_checks = []
    for arm, directory in receipt_dirs.items():
        expected_sha = CONTROL_SHA if arm == "control" else CANDIDATE_SHA
        for seed in SEEDS:
            receipt = read(directory / f"seed_{seed}.json")
            receipt_checks.append(all([
                receipt["status"] == "COMPLETE",
                int(receipt["seed"]) == seed,
                receipt["checkpoint_sha256"] == expected_sha,
                receipt["manifest_sha256"] == MANIFEST_SHA,
                int(receipt["worlds"]) == 20,
                int(receipt["steps"]) == 20,
                float(receipt["ddim_eta"]) == 0.0,
                int(receipt["scene_count"]) == 947,
                int(receipt["agent_occurrence_count"]) == 24334,
                receipt["model_dtype"] == "fp32",
                not receipt["autocast_enabled"],
                receipt["fixed_record"] is None if arm == "control" else
                receipt["fixed_record"]["derived_from_same_forward_as_raw"] is True,
            ]))

    evaluation_identity = all(
        value["manifest_sha256"] == MANIFEST_SHA
        and value["scene_count"] == 947
        and value["agent_occurrence_count"] == 24334
        and value["worlds"] == 20 and value["steps"] == 20
        and value["ddim_eta"] == 0 and value["model_dtype"] == "fp32"
        and value["requested_inference_seeds"] == list(SEEDS)
        for name, value in payloads.items() if name != "parent_epoch73_raw")
    checkpoint_binding = all([
        payloads["control_epoch81_raw"]["checkpoint_sha256"] == CONTROL_SHA,
        payloads["candidate_epoch81_raw"]["checkpoint_sha256"] == CANDIDATE_SHA,
        payloads["candidate_epoch81_fixed"]["checkpoint_sha256"] == CANDIDATE_SHA,
        payloads["candidate_epoch81_fixed"]["derived_from_same_forward_as_raw"] is True,
        Path(payloads["candidate_epoch81_fixed"]["raw_output"]).resolve()
        == args.candidate_raw_result.resolve(),
    ])
    finite = all(
        math.isfinite(block[metric]["mean"])
        and math.isfinite(block[metric]["sample_standard_deviation"])
        for block in tables.values() for metric in METRICS)

    candidate = tables["candidate_epoch81_fixed"]
    parent = tables["parent_epoch73_raw"]
    control = tables["control_epoch81_raw"]
    raw = tables["candidate_epoch81_raw"]
    deltas = {
        baseline: {
            metric: candidate[metric]["mean"] - tables[baseline][metric]["mean"]
            for metric in METRICS}
        for baseline in ("parent_epoch73_raw", "control_epoch81_raw", "matched_gdts")
    }
    fixed_minus_raw = {
        metric: candidate[metric]["mean"] - raw[metric]["mean"]
        for metric in METRICS}
    paired_fixed_minus_control = {
        str(seed): {
            metric: candidate[metric]["per_seed"][index]
            - control[metric]["per_seed"][index]
            for metric in METRICS}
        for index, seed in enumerate(SEEDS)
    }
    improvement_counts = {
        metric: sum(value[metric] < 0 for value in paired_fixed_minus_control.values())
        for metric in METRICS}
    upgrade = all(
        deltas[baseline][metric] <= 0
        for baseline in ("parent_epoch73_raw", "control_epoch81_raw")
        for metric in ("minADE", "minFDE")) and all(
        deltas[baseline][metric] < 0
        for baseline in ("parent_epoch73_raw", "control_epoch81_raw")
        for metric in ("JADE", "JFDE"))
    original = all(
        deltas["matched_gdts"][metric] <= 0
        for metric in ("minADE", "minFDE")) and all(
        deltas["matched_gdts"][metric] < 0
        for metric in ("JADE", "JFDE"))

    ledger_path = args.asset_root / "GPU_LEDGER.jsonl"
    ledger = [json.loads(line) for line in ledger_path.read_text().splitlines()
              if line.strip()]
    gpu_seconds = sum(float(row["elapsed_seconds"]) for row in ledger)
    checks = {
        "ten_seed_receipts_complete": all(receipt_checks) and len(receipt_checks) == 10,
        "formal_evaluation_identity": evaluation_identity,
        "checkpoint_binding": checkpoint_binding,
        "finite_metrics": finite,
        "gpu_budget_not_exceeded": gpu_seconds <= 7200.0,
    }
    status = "COMPLETE" if all(checks.values()) else "INVALID"
    result = {
        "schema": "ebjd-univ-epoch81-frozen-eval-results-v1",
        "status": status, "training_seed": 3101,
        "inference_seeds": list(SEEDS), "independent_test": False,
        "metrics": tables,
        "candidate_fixed_minus": deltas,
        "candidate_fixed_minus_candidate_raw": fixed_minus_raw,
        "candidate_fixed_minus_control_per_seed": paired_fixed_minus_control,
        "candidate_fixed_improvement_seed_count_vs_control": improvement_counts,
        "upgrade_contract": "PASS" if upgrade else "FAIL",
        "original_gdts_contract": "PASS" if original else "FAIL",
        "old_univ_advantage_retained": bool(upgrade),
        "checks": checks, "gpu_ledger": ledger,
        "actual_gpu_wall_seconds": gpu_seconds,
        "gpu_budget_seconds": 7200.0,
        "checkpoint_sha256": {"control": CONTROL_SHA, "candidate": CANDIDATE_SHA},
    }

    args.archive.mkdir(parents=True, exist_ok=True)
    for source, target in (
        (args.control_config, "CONTROL_CONFIG.json"),
        (args.candidate_config, "CANDIDATE_CONFIG.json"),
        (args.asset_root / "FROZEN_EVAL_PLAN.json", "FROZEN_EVAL_PLAN.json"),
        (args.asset_root / "CPU_PREFLIGHT_RECEIPT.json", "CPU_PREFLIGHT_RECEIPT.json"),
        (ledger_path, "GPU_LEDGER.jsonl"),
    ):
        shutil.copy2(source, args.archive / target)
    atomic_text(args.archive / "RESULTS.json", json.dumps(
        result, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

    def cell(name: str, metric: str) -> str:
        block = tables[name][metric]
        return f'{block["mean"]:.6f} ± {block["sample_standard_deviation"]:.6f}'

    labels = (
        ("parent_epoch73_raw", "原 UNIV epoch73 Raw"),
        ("control_epoch81_raw", "Control epoch81 Raw"),
        ("candidate_epoch81_raw", "Candidate epoch81 Raw（辅助）"),
        ("candidate_epoch81_fixed", "Candidate epoch81 Fixed-anchor（主候选）"),
        ("matched_gdts", "matched GDTS"),
    )
    lines = ["# UNIV epoch81 冻结补评测", "",
             f"状态：**{status}**。单训练 checkpoint；2035–2039 是推理重复；`independent_test=false`。", "",
             "| 正式五种子结果 | minADE ↓ | minFDE ↓ | JADE ↓ | JFDE ↓ |",
             "|---|---:|---:|---:|---:|"]
    for name, label in labels:
        lines.append(f"| {label} | " + " | ".join(
            cell(name, metric) for metric in METRICS) + " |")
    lines += ["", f"- 原升级合同：**{'PASS' if upgrade else 'FAIL'}**。",
              f"- 相对 matched GDTS 原合同：**{'PASS' if original else 'FAIL'}**。",
              f"- 是否保住原 UNIV epoch73 优势：**{'是' if upgrade else '否'}**。",
              f"- Fixed-anchor − Candidate Raw：`{fixed_minus_raw}`。",
              f"- Fixed-anchor 相对 Control 各指标改善 seed 数：`{improvement_counts}`。", "",
              "## Fixed-anchor − Control 逐 seed 差值", "",
              "| seed | minADE | minFDE | JADE | JFDE |", "|---:|---:|---:|---:|---:|"]
    for seed in SEEDS:
        row = paired_fixed_minus_control[str(seed)]
        lines.append(f"| {seed} | " + " | ".join(
            f"{row[metric]:+.6f}" for metric in METRICS) + " |")
    recommendation = (
        "UNIV 迁移候选通过；保留原 epoch73，同时保留 Candidate epoch81 作为候选，不启动下一轮实验。"
        if upgrade else
        "保留原 UNIV epoch73；停止本迁移分支，不续训、不调参、不延长 epoch。")
    lines += ["", f"本轮新增 GPU 墙钟：**{gpu_seconds:.2f} / 7200 秒**。",
              "", f"唯一建议：**{recommendation}**", ""]
    atomic_text(args.archive / "REPORT_ZH.md", "\n".join(lines))
    print(json.dumps({"status": status, "upgrade": upgrade,
                      "original": original, "gpu_seconds": gpu_seconds},
                     ensure_ascii=False, sort_keys=True))
    if status != "COMPLETE":
        raise RuntimeError("frozen evaluation audit invalid")


if __name__ == "__main__":
    main()
