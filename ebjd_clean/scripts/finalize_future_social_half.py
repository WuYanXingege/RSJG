#!/usr/bin/env python3
"""Validate and archive the frozen UNIV future-social gamma=0.5 run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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


def load(path: str | Path):
    return json.loads(Path(path).read_text())


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


def atomic_json(path: Path, value) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def validate_reused(
    path: Path, mode: str, checkpoint_sha: str, config_sha: str,
    manifest_sha: str,
) -> list[dict]:
    rows = load(path)
    if [row.get("seed") for row in rows] != list(SEEDS):
        raise ValueError(f"{mode} does not contain exact seeds 2035--2039")
    for row in rows:
        expected = {
            "status": "COMPLETE", "mode": mode,
            "checkpoint_sha256": checkpoint_sha,
            "config_sha256": config_sha,
            "manifest_sha256": manifest_sha,
            "worlds": 20, "ddim_steps": 20, "ddim_eta": 0.0,
            "scene_count": 947, "agent_occurrence_count": 24334,
        }
        for key, value in expected.items():
            if row.get(key) != value:
                raise ValueError(f"{mode} seed {row.get('seed')} mismatch: {key}")
    return rows


def validate_half(
    directory: Path, ledger_by_seed: dict[int, dict], checkpoint_sha: str,
    config_sha: str, manifest_sha: str,
) -> list[dict]:
    rows = []
    for seed in SEEDS:
        receipt = load(directory / f"seed_{seed}.json")
        expected = {
            "schema": "ebjd-evaluation-seed-receipt-v3",
            "status": "COMPLETE",
            "checkpoint_sha256": checkpoint_sha,
            "config_sha256": config_sha,
            "manifest_sha256": manifest_sha,
            "future_social_mode": "half", "future_social_gamma": 0.5,
            "seed": seed, "worlds": 20, "steps": 20, "ddim_eta": 0.0,
            "scene_count": 947, "agent_occurrence_count": 24334,
        }
        for key, value in expected.items():
            if receipt.get(key) != value:
                raise ValueError(f"HALF seed {seed} mismatch: {key}")
        record = receipt.get("raw_record")
        metrics = None if not isinstance(record, dict) else record.get("metrics")
        if record.get("seed") != seed or any(
                not math.isfinite(float(metrics.get(name, math.nan)))
                for name in METRICS):
            raise ValueError(f"HALF seed {seed} has invalid metrics")
        ledger = ledger_by_seed.get(seed)
        if ledger is None or ledger.get("status") != "COMPLETE":
            raise ValueError(f"HALF seed {seed} lacks complete GPU ledger entry")
        rows.append({
            "status": "COMPLETE", "mode": "half", "gamma": 0.5,
            "checkpoint_sha256": checkpoint_sha,
            "config_sha256": config_sha,
            "manifest_sha256": manifest_sha,
            "evaluator_sha256": receipt["evaluator_sha256"],
            "seed": seed, "worlds": 20, "ddim_steps": 20,
            "ddim_eta": 0.0, "scene_count": metrics["scene_count"],
            "agent_occurrence_count": metrics["agent_count"],
            "metrics": {name: float(metrics[name]) for name in METRICS},
            "inference_elapsed_seconds": float(
                receipt["inference_elapsed_seconds"]),
            "gpu_wall_seconds": float(ledger["elapsed_seconds"]),
        })
    return rows


def aggregate(rows: list[dict]) -> dict:
    result = {}
    for name in METRICS:
        values = [float(row["metrics"][name]) for row in rows]
        result[name] = {
            "mean": float(statistics.fmean(values)),
            "sample_standard_deviation": float(statistics.stdev(values)),
            "per_seed": values,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-per-seed", required=True)
    parser.add_argument("--self-per-seed", required=True)
    parser.add_argument("--half-receipts", required=True)
    parser.add_argument("--prior-epoch81-results", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--tests-passed", type=int, required=True)
    args = parser.parse_args()

    archive = Path(args.archive)
    checkpoint_sha, config_sha, manifest_sha = (
        sha256(args.checkpoint), sha256(args.config), sha256(args.manifest))
    ledger_rows = [json.loads(line) for line in Path(args.ledger).read_text().splitlines()
                   if line.strip()]
    ledger_by_seed = {int(row["seed"]): row for row in ledger_rows}
    if len(ledger_rows) != len(SEEDS) or len(ledger_by_seed) != len(SEEDS):
        raise ValueError("GPU ledger must contain exactly five formal calls")

    full_rows = validate_reused(
        Path(args.full_per_seed), "full", checkpoint_sha, config_sha, manifest_sha)
    self_rows = validate_reused(
        Path(args.self_per_seed), "self_only", checkpoint_sha, config_sha,
        manifest_sha)
    half_rows = validate_half(
        Path(args.half_receipts), ledger_by_seed, checkpoint_sha, config_sha,
        manifest_sha)
    modes = {
        "FULL": aggregate(full_rows), "HALF": aggregate(half_rows),
        "SELF_ONLY": aggregate(self_rows),
    }

    half_minus_full = {
        str(seed): {name: half_rows[index]["metrics"][name]
                    - full_rows[index]["metrics"][name] for name in METRICS}
        for index, seed in enumerate(SEEDS)}
    half_minus_self = {
        str(seed): {name: half_rows[index]["metrics"][name]
                    - self_rows[index]["metrics"][name] for name in METRICS}
        for index, seed in enumerate(SEEDS)}
    half_better_full = {
        name: sum(half_minus_full[str(seed)][name] < 0 for seed in SEEDS)
        for name in METRICS}
    joint_retention = {
        name: ((modes["SELF_ONLY"][name]["mean"] - modes["HALF"][name]["mean"])
               / (modes["SELF_ONLY"][name]["mean"] - modes["FULL"][name]["mean"]))
        for name in ("JADE", "JFDE")}
    marginal_recovery = {
        name: ((modes["FULL"][name]["mean"] - modes["HALF"][name]["mean"])
               / (modes["FULL"][name]["mean"] - modes["SELF_ONLY"][name]["mean"]))
        for name in ("minADE", "minFDE")}

    balanced = (
        all(modes["HALF"][name]["mean"] < modes["FULL"][name]["mean"]
            for name in ("minADE", "minFDE"))
        and all(half_better_full[name] >= 4 for name in ("minADE", "minFDE"))
        and all(joint_retention[name] >= 0.80 for name in ("JADE", "JFDE")))
    classification = "BALANCED_CANDIDATE" if balanced else "NO_CLEAR_GAIN"

    prior = load(args.prior_epoch81_results)
    gdts = prior["metrics"]["matched_gdts"]
    half_minus_gdts = {
        name: modes["HALF"][name]["mean"] - float(gdts[name]["mean"])
        for name in METRICS}
    gpu_seconds = float(sum(float(row["elapsed_seconds"]) for row in ledger_rows))
    if gpu_seconds > 2700:
        raise ValueError(f"GPU budget exceeded: {gpu_seconds}")

    results = {
        "schema": "ebjd-univ-future-social-half-results-v1",
        "status": "COMPLETE", "classification": classification,
        "training_seed": 3101, "inference_seeds": list(SEEDS),
        "checkpoint_sha256": checkpoint_sha, "config_sha256": config_sha,
        "manifest_sha256": manifest_sha, "gamma": 0.5,
        "worlds": 20, "ddim_steps": 20, "ddim_eta": 0.0,
        "metrics": modes,
        "half_minus_full_per_seed": half_minus_full,
        "half_minus_self_only_per_seed": half_minus_self,
        "half_better_than_full_seed_count": half_better_full,
        "joint_benefit_retention": joint_retention,
        "marginal_loss_recovery": marginal_recovery,
        "matched_gdts": gdts,
        "half_minus_matched_gdts": half_minus_gdts,
        "actual_gpu_wall_seconds": gpu_seconds, "gpu_budget_seconds": 2700,
        "implementation_commit": args.implementation_commit,
        "cpu_tests_passed": args.tests_passed,
        "candidate_transfer_status_unchanged": "FAIL",
        "evidence_boundary": (
            "Single-checkpoint frozen inference screening; not a significance "
            "test, retrained structural ablation, or formal model replacement."),
    }

    def cell(mode: str, name: str) -> str:
        item = modes[mode][name]
        return f'{item["mean"]:.6f} ± {item["sample_standard_deviation"]:.6f}'

    paired_rows = "\n".join(
        "| {seed} | {fa:+.6f} | {ff:+.6f} | {fj:+.6f} | {fJ:+.6f} | "
        "{sa:+.6f} | {sf:+.6f} | {sj:+.6f} | {sJ:+.6f} |".format(
            seed=seed,
            fa=half_minus_full[str(seed)]["minADE"],
            ff=half_minus_full[str(seed)]["minFDE"],
            fj=half_minus_full[str(seed)]["JADE"],
            fJ=half_minus_full[str(seed)]["JFDE"],
            sa=half_minus_self[str(seed)]["minADE"],
            sf=half_minus_self[str(seed)]["minFDE"],
            sj=half_minus_self[str(seed)]["JADE"],
            sJ=half_minus_self[str(seed)]["JFDE"])
        for seed in SEEDS)
    report = f"""# UNIV Future Social Attention gamma=0.5 冻结实验

状态：**COMPLETE**。唯一新增变量为六个 FutureBlock 内部固定 `gamma=0.5`：`A_self + 0.5 × (A_full − A_self)`。混合位于 attention 输出投影之后、残差之前；两路读取同一层同一输入并共享参数。FULL 和 SELF_ONLY 直接复用已绑定结果。

| 五推理种子均值 ± 样本标准差 | minADE ↓ | minFDE ↓ | JADE ↓ | JFDE ↓ |
|---|---:|---:|---:|---:|
| FULL | {cell('FULL','minADE')} | {cell('FULL','minFDE')} | {cell('FULL','JADE')} | {cell('FULL','JFDE')} |
| HALF (`gamma=0.5`) | {cell('HALF','minADE')} | {cell('HALF','minFDE')} | {cell('HALF','JADE')} | {cell('HALF','JFDE')} |
| SELF_ONLY | {cell('SELF_ONLY','minADE')} | {cell('SELF_ONLY','minFDE')} | {cell('SELF_ONLY','JADE')} | {cell('SELF_ONLY','JFDE')} |

联合收益保留率：JADE `{joint_retention['JADE']:.6f}`，JFDE `{joint_retention['JFDE']:.6f}`。

边缘损失恢复率：minADE `{marginal_recovery['minADE']:.6f}`，minFDE `{marginal_recovery['minFDE']:.6f}`。

HALF 优于 FULL 的 seed 数：`{half_better_full}`。

## 逐 seed 配对差值

前四列为 `HALF − FULL`，后四列为 `HALF − SELF_ONLY`；负值表示前者更低。

| seed | ΔF minADE | ΔF minFDE | ΔF JADE | ΔF JFDE | ΔS minADE | ΔS minFDE | ΔS JADE | ΔS JFDE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{paired_rows}

相对 matched GDTS 的 `HALF − GDTS`：`{half_minus_gdts}`；负值表示 HALF 更好。

候选判断：**{classification}**。

本轮新增 GPU 墙钟：**{gpu_seconds:.2f} / 2700 秒**。

证据边界：这是单训练 checkpoint 的冻结推理筛选，不是统计显著性检验、从头训练的结构消融或正式模型替换。既有 Candidate 迁移分支的 **FAIL** 结论不变。
"""

    archive.mkdir(parents=True, exist_ok=True)
    atomic_json(archive / "RESULTS.json", results)
    atomic_json(archive / "HALF_PER_SEED.json", half_rows)
    atomic_json(archive / "GPU_LEDGER.json", ledger_rows)
    atomic_text(archive / "REPORT_ZH.md", report)
    shutil.copyfile(args.config, archive / "CONTROL_CONFIG.json")
    print(json.dumps({
        "status": "COMPLETE", "classification": classification,
        "gpu_seconds": gpu_seconds,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
