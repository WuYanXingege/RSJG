#!/usr/bin/env python3
"""Validate and archive the frozen UNIV future-social intervention."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import statistics
import tempfile
from pathlib import Path


SEEDS = (2035, 2036, 2037, 2038, 2039)
METRICS = ("minADE", "minFDE", "JADE", "JFDE")


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def compact_receipts(
    directory: Path, mode: str, checkpoint_sha: str, config_sha: str,
    manifest_sha: str, ledger_by_seed: dict[int, dict] | None = None,
) -> list[dict]:
    rows = []
    for seed in SEEDS:
        receipt_path = directory / f"seed_{seed}.json"
        receipt = load(receipt_path)
        expected_schema = (
            "ebjd-evaluation-seed-receipt-v2" if mode == "self_only"
            else "ebjd-evaluation-seed-receipt-v1")
        expected = {
            "schema": expected_schema,
            "status": "COMPLETE",
            "checkpoint_sha256": checkpoint_sha,
            "config_sha256": config_sha,
            "manifest_sha256": manifest_sha,
            "seed": seed,
            "worlds": 20,
            "steps": 20,
            "ddim_eta": 0.0,
            "scene_count": 947,
            "agent_occurrence_count": 24334,
        }
        if mode == "self_only":
            expected["future_social_mode"] = mode
        for key, value in expected.items():
            if receipt.get(key) != value:
                raise ValueError(
                    f"{mode} seed {seed} receipt mismatch for {key}: "
                    f"{receipt.get(key)!r} != {value!r}")
        record = receipt.get("raw_record")
        if not isinstance(record, dict) or record.get("seed") != seed:
            raise ValueError(f"{mode} seed {seed} lacks its complete raw record")
        metrics = record.get("metrics", {})
        if any(not isinstance(metrics.get(name), (int, float)) for name in METRICS):
            raise ValueError(f"{mode} seed {seed} lacks finite scalar metrics")
        row = {
            "status": "COMPLETE",
            "mode": mode,
            "checkpoint_sha256": checkpoint_sha,
            "config_sha256": config_sha,
            "manifest_sha256": manifest_sha,
            "evaluator_sha256": receipt["evaluator_sha256"],
            "seed": seed,
            "worlds": 20,
            "ddim_steps": 20,
            "ddim_eta": 0.0,
            "scene_count": metrics["scene_count"],
            "agent_occurrence_count": metrics["agent_count"],
            "metrics": {name: float(metrics[name]) for name in METRICS},
            "inference_elapsed_seconds": float(
                receipt["inference_elapsed_seconds"]),
        }
        if ledger_by_seed is not None:
            ledger = ledger_by_seed.get(seed)
            if ledger is None or ledger.get("status") != "COMPLETE":
                raise ValueError(f"self_only seed {seed} lacks a complete GPU ledger row")
            row["gpu_wall_seconds"] = float(ledger["elapsed_seconds"])
        rows.append(row)
    return rows


def aggregate(rows: list[dict]) -> dict:
    result = {}
    for name in METRICS:
        values = [row["metrics"][name] for row in rows]
        result[name] = {
            "mean": float(statistics.fmean(values)),
            "sample_standard_deviation": float(statistics.stdev(values)),
            "per_seed": values,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-receipts", required=True)
    parser.add_argument("--self-receipts", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--tests-passed", type=int, required=True)
    args = parser.parse_args()

    archive = Path(args.archive)
    checkpoint_sha = sha256(args.checkpoint)
    config_sha = sha256(args.config)
    manifest_sha = sha256(args.manifest)
    ledger_rows = [json.loads(line) for line in Path(args.ledger).read_text().splitlines()
                   if line.strip()]
    ledger_by_seed = {int(row["seed"]): row for row in ledger_rows
                      if row.get("mode") == "self_only"}
    if len(ledger_by_seed) != len(SEEDS):
        raise ValueError("GPU ledger does not contain exactly five SELF_ONLY seeds")

    full_rows = compact_receipts(
        Path(args.full_receipts), "full", checkpoint_sha, config_sha, manifest_sha)
    self_rows = compact_receipts(
        Path(args.self_receipts), "self_only", checkpoint_sha, config_sha,
        manifest_sha, ledger_by_seed)
    full = aggregate(full_rows)
    self_only = aggregate(self_rows)
    deltas = {
        str(seed): {
            name: full_rows[index]["metrics"][name]
            - self_rows[index]["metrics"][name]
            for name in METRICS}
        for index, seed in enumerate(SEEDS)}
    full_better_counts = {
        name: sum(deltas[str(seed)][name] < 0 for seed in SEEDS)
        for name in METRICS}

    joint_mean_better = {
        name: full[name]["mean"] < self_only[name]["mean"]
        for name in ("JADE", "JFDE")}
    if all(joint_mean_better.values()) and all(
            full_better_counts[name] >= 4 for name in ("JADE", "JFDE")):
        mechanism = "SUPPORT"
    elif not any(joint_mean_better.values()):
        mechanism = "NOT_SUPPORTED"
    else:
        mechanism = "MIXED"
    marginal_tradeoff = any(
        full[name]["mean"] > self_only[name]["mean"]
        for name in ("minADE", "minFDE"))
    gpu_seconds = float(sum(float(row["elapsed_seconds"]) for row in ledger_rows))
    if gpu_seconds > 5400:
        raise ValueError(f"GPU budget exceeded: {gpu_seconds}")

    results = {
        "schema": "ebjd-univ-future-social-self-only-results-v1",
        "status": "COMPLETE",
        "training_seed": 3101,
        "inference_seeds": list(SEEDS),
        "checkpoint_sha256": checkpoint_sha,
        "config_sha256": config_sha,
        "manifest_sha256": manifest_sha,
        "worlds": 20,
        "ddim_steps": 20,
        "ddim_eta": 0.0,
        "metrics": {"FULL": full, "SELF_ONLY": self_only},
        "full_minus_self_only_per_seed": deltas,
        "full_better_seed_count": full_better_counts,
        "mechanism_judgment": mechanism,
        "marginal_metric_tradeoff": marginal_tradeoff,
        "actual_gpu_wall_seconds": gpu_seconds,
        "gpu_budget_seconds": 5400,
        "implementation_commit": args.implementation_commit,
        "cpu_tests_passed": args.tests_passed,
        "evidence_boundary": (
            "Frozen inference intervention on one training checkpoint; not a "
            "from-scratch structural ablation or significance test."),
        "candidate_transfer_status_unchanged": "FAIL",
    }

    def cell(mode: str, name: str) -> str:
        metric = results["metrics"][mode][name]
        return f'{metric["mean"]:.6f} ± {metric["sample_standard_deviation"]:.6f}'

    rows = "\n".join(
        "| {seed} | {minADE:+.6f} | {minFDE:+.6f} | {JADE:+.6f} | {JFDE:+.6f} |".format(
            seed=seed, **deltas[str(seed)]) for seed in SEEDS)
    tradeoff = "存在边缘指标权衡。" if marginal_tradeoff else "未观察到边缘指标权衡。"
    report = f"""# UNIV Future Social Attention SELF_ONLY 冻结机制验证

状态：**COMPLETE**。实验变量仅为六个 FutureBlock 的 Social Attention 连接范围；FULL 与 SELF_ONLY 使用同一 Control epoch81 checkpoint。SELF_ONLY 保留自连接、Q/K/V、输出投影、归一化、条件调制和残差，只屏蔽跨行人连接。历史 Social Attention 及其他推理路径不变。

| 五推理种子均值 ± 样本标准差 | minADE ↓ | minFDE ↓ | JADE ↓ | JFDE ↓ |
|---|---:|---:|---:|---:|
| FULL（复用已绑定结果） | {cell('FULL', 'minADE')} | {cell('FULL', 'minFDE')} | {cell('FULL', 'JADE')} | {cell('FULL', 'JFDE')} |
| SELF_ONLY | {cell('SELF_ONLY', 'minADE')} | {cell('SELF_ONLY', 'minFDE')} | {cell('SELF_ONLY', 'JADE')} | {cell('SELF_ONLY', 'JFDE')} |

## 配对差值：FULL − SELF_ONLY

负值表示 FULL 更好。

| seed | minADE | minFDE | JADE | JFDE |
|---:|---:|---:|---:|---:|
{rows}

FULL 优于 SELF_ONLY 的 seed 数：`{full_better_counts}`。

机制判断：**{mechanism}**。{tradeoff}

本轮新增 GPU 墙钟：**{gpu_seconds:.2f} / 5400 秒**。

证据边界：这是同一训练 checkpoint 的冻结推理干预，只能检验该模型是否实际利用未来跨人连接；不能替代从头训练的结构消融，也不构成统计显著性检验。既有 Candidate 迁移分支的 **FAIL** 状态不变。
"""

    archive.mkdir(parents=True, exist_ok=True)
    atomic_json(archive / "RESULTS.json", results)
    atomic_json(archive / "FULL_REUSED_PER_SEED.json", full_rows)
    atomic_json(archive / "SELF_ONLY_PER_SEED.json", self_rows)
    atomic_text(archive / "REPORT_ZH.md", report)
    shutil.copyfile(args.config, archive / "CONTROL_CONFIG.json")
    atomic_json(archive / "GPU_LEDGER.json", ledger_rows)
    print(json.dumps({
        "status": "COMPLETE", "mechanism_judgment": mechanism,
        "gpu_seconds": gpu_seconds,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
