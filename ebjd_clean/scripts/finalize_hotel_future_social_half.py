#!/usr/bin/env python3
"""Validate and archive HOTEL epoch45 future-social gamma=0.5 inference."""

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


def full_rows(path: Path) -> list[dict]:
    if sha256(path) != "5e0f13ab4cb272aebe9ea19e6f0307a9eb3213301256c2f6c826cf449bab800f":
        raise ValueError("FULL source result SHA256 mismatch")
    payload = load(path)
    if (payload.get("split") != "test" or payload.get("worlds") != 20
            or payload.get("steps") != 20
            or [row.get("seed") for row in payload.get("per_seed", [])]
            != list(SEEDS)):
        raise ValueError("FULL source protocol mismatch")
    rows = []
    for source in payload["per_seed"]:
        metrics = source["metrics"]
        if metrics.get("scene_count") != 445 or metrics.get("agent_count") != 1197:
            raise ValueError("FULL coverage mismatch")
        rows.append({
            "status": "COMPLETE", "mode": "full", "seed": source["seed"],
            "worlds": 20, "ddim_steps": 20, "ddim_eta": 0.0,
            "scene_count": 445, "agent_occurrence_count": 1197,
            "metrics": {name: float(metrics[name]) for name in METRICS},
        })
    return rows


def gdts_rows(path: Path) -> list[dict]:
    if sha256(path) != "df9aa35cd26296558c864a2bcda792a8411bbc83a530376c45d3429a768eae6c":
        raise ValueError("matched GDTS result SHA256 mismatch")
    payload = load(path)
    if payload.get("status") != "COMPLETED" or payload.get("protocol") != "E_joint":
        raise ValueError("matched GDTS protocol mismatch")
    if payload.get("evaluation_seeds") != list(SEEDS):
        raise ValueError("matched GDTS seed mismatch")
    rows = []
    for source in payload["runs"]:
        metrics = source["metrics"]
        counts = source["counts"]
        if counts.get("minADE@K") != 1197 or counts.get("JADE") != 445:
            raise ValueError("matched GDTS coverage mismatch")
        rows.append({
            "status": "COMPLETE", "mode": "matched_gdts",
            "seed": int(source["evaluation_seed"]), "worlds": 20,
            "ddim_steps": 20, "scene_count": 445,
            "agent_occurrence_count": 1197,
            "metrics": {
                "minADE": float(metrics["minADE@K"]),
                "minFDE": float(metrics["minFDE@K"]),
                "JADE": float(metrics["JADE"]),
                "JFDE": float(metrics["JFDE"]),
            },
        })
    return rows


def half_rows(
    directory: Path, ledger_by_seed: dict[int, dict], checkpoint_sha: str,
    config_sha: str, manifest_sha: str,
) -> list[dict]:
    rows = []
    for seed in SEEDS:
        receipt = load(directory / f"seed_{seed}.json")
        expected = {
            "schema": "ebjd-evaluation-seed-receipt-v3",
            "status": "COMPLETE", "checkpoint_sha256": checkpoint_sha,
            "config_sha256": config_sha, "manifest_sha256": manifest_sha,
            "future_social_mode": "half", "future_social_gamma": 0.5,
            "seed": seed, "worlds": 20, "steps": 20, "ddim_eta": 0.0,
            "scene_count": 445, "agent_occurrence_count": 1197,
        }
        for key, value in expected.items():
            if receipt.get(key) != value:
                raise ValueError(f"HALF seed {seed} mismatch: {key}")
        record = receipt.get("raw_record")
        metrics = None if not isinstance(record, dict) else record.get("metrics")
        if record.get("seed") != seed or any(
                not math.isfinite(float(metrics.get(name, math.nan)))
                for name in METRICS):
            raise ValueError(f"HALF seed {seed} metrics invalid")
        ledger = ledger_by_seed.get(seed)
        if ledger is None or ledger.get("status") != "COMPLETE":
            raise ValueError(f"HALF seed {seed} lacks GPU ledger entry")
        rows.append({
            "status": "COMPLETE", "mode": "half", "gamma": 0.5,
            "checkpoint_sha256": checkpoint_sha,
            "config_sha256": config_sha, "manifest_sha256": manifest_sha,
            "evaluator_sha256": receipt["evaluator_sha256"], "seed": seed,
            "worlds": 20, "ddim_steps": 20, "ddim_eta": 0.0,
            "scene_count": metrics["scene_count"],
            "agent_occurrence_count": metrics["agent_count"],
            "metrics": {name: float(metrics[name]) for name in METRICS},
            "inference_elapsed_seconds": float(receipt["inference_elapsed_seconds"]),
            "gpu_wall_seconds": float(ledger["elapsed_seconds"]),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-result", required=True)
    parser.add_argument("--gdts-result", required=True)
    parser.add_argument("--half-receipts", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--tests-passed", type=int, required=True)
    args = parser.parse_args()

    checkpoint_sha, config_sha, manifest_sha = (
        sha256(args.checkpoint), sha256(args.config), sha256(args.manifest))
    if checkpoint_sha != "425988097793b639eb868fcc9ff28e41a3022c08598d13f396383da89268e1b1":
        raise ValueError("HOTEL epoch45 checkpoint SHA256 mismatch")
    if config_sha != "a3800984c4876bec69caa62a587ab2db609d2dce1c7670995ddee0ecb6440278":
        raise ValueError("HOTEL frozen config SHA256 mismatch")
    if manifest_sha != "84c95cb431caf7214d1b85e8e7dbfeaf48c9fe301b1cda7f3201c3e8cf199c78":
        raise ValueError("HOTEL test manifest SHA256 mismatch")

    ledger_rows = [json.loads(line) for line in Path(args.ledger).read_text().splitlines()
                   if line.strip()]
    ledger_by_seed = {int(row["seed"]): row for row in ledger_rows}
    if len(ledger_rows) != 5 or len(ledger_by_seed) != 5:
        raise ValueError("GPU ledger must contain exactly five formal HALF calls")
    full = full_rows(Path(args.full_result))
    half = half_rows(
        Path(args.half_receipts), ledger_by_seed, checkpoint_sha, config_sha,
        manifest_sha)
    gdts = gdts_rows(Path(args.gdts_result))
    modes = {"FULL": aggregate(full), "HALF": aggregate(half),
             "matched_GDTS": aggregate(gdts)}
    delta = {
        str(seed): {name: half[index]["metrics"][name] - full[index]["metrics"][name]
                    for name in METRICS}
        for index, seed in enumerate(SEEDS)}
    improved = {
        name: sum(delta[str(seed)][name] < 0 for seed in SEEDS)
        for name in METRICS}
    cross_fold_checks = {
        "half_minADE_mean_below_full": (
            modes["HALF"]["minADE"]["mean"] < modes["FULL"]["minADE"]["mean"]),
        "half_minFDE_mean_below_full": (
            modes["HALF"]["minFDE"]["mean"] < modes["FULL"]["minFDE"]["mean"]),
        "minADE_at_least_4_of_5": improved["minADE"] >= 4,
        "minFDE_at_least_4_of_5": improved["minFDE"] >= 4,
        "half_JADE_below_gdts": (
            modes["HALF"]["JADE"]["mean"] < modes["matched_GDTS"]["JADE"]["mean"]),
        "half_JFDE_below_gdts": (
            modes["HALF"]["JFDE"]["mean"] < modes["matched_GDTS"]["JFDE"]["mean"]),
    }
    cross_fold = "SUPPORT" if all(cross_fold_checks.values()) else "NOT_SUPPORTED"
    contract_checks = {
        name: modes["HALF"][name]["mean"] <= modes["matched_GDTS"][name]["mean"]
        for name in METRICS}
    original_contract = "PASS" if all(contract_checks.values()) else "FAIL"
    gpu_seconds = float(sum(float(row["elapsed_seconds"]) for row in ledger_rows))
    if gpu_seconds > 1800:
        raise ValueError(f"GPU budget exceeded: {gpu_seconds}")

    results = {
        "schema": "ebjd-hotel-future-social-half-results-v1",
        "status": "COMPLETE", "cross_fold_direction": cross_fold,
        "original_hotel_performance_contract": original_contract,
        "cross_fold_checks": cross_fold_checks,
        "original_contract_all_four_no_worse_than_matched_gdts": contract_checks,
        "training_seed": 3101, "inference_seeds": list(SEEDS),
        "checkpoint_sha256": checkpoint_sha, "config_sha256": config_sha,
        "manifest_sha256": manifest_sha, "gamma": 0.5,
        "worlds": 20, "ddim_steps": 20, "ddim_eta": 0.0,
        "metrics": modes, "half_minus_full_per_seed": delta,
        "half_better_than_full_seed_count": improved,
        "half_minus_gdts_mean": {
            name: modes["HALF"][name]["mean"]
            - modes["matched_GDTS"][name]["mean"] for name in METRICS},
        "actual_gpu_wall_seconds": gpu_seconds, "gpu_budget_seconds": 1800,
        "implementation_commit": args.implementation_commit,
        "cpu_tests_passed": args.tests_passed,
        "independent_test": False,
        "evidence_boundary": (
            "Five inference repeats of one training checkpoint on mirrored HOTEL "
            "validation/test windows; not training-seed significance or retraining ablation."),
    }

    def cell(mode: str, name: str) -> str:
        item = modes[mode][name]
        return f'{item["mean"]:.6f} ± {item["sample_standard_deviation"]:.6f}'

    delta_rows = "\n".join(
        "| {seed} | {minADE:+.6f} | {minFDE:+.6f} | {JADE:+.6f} | {JFDE:+.6f} |".format(
            seed=seed, **delta[str(seed)]) for seed in SEEDS)
    failed_cross = [key for key, value in cross_fold_checks.items() if not value]
    failed_contract = [key for key, value in contract_checks.items() if not value]
    report = f"""# HOTEL Future Social Attention gamma=0.5 冻结推理

状态：**COMPLETE**。唯一新增变量为六个 FutureBlock 内部固定 `gamma=0.5`：`A_self + 0.5 × (A_full − A_self)`。使用原 HOTEL epoch45 checkpoint、Raw 输出和 test manifest；FULL 与 matched GDTS 直接复用已绑定五种子结果。

| 五推理种子均值 ± 样本标准差 | minADE ↓ | minFDE ↓ | JADE ↓ | JFDE ↓ |
|---|---:|---:|---:|---:|
| FULL | {cell('FULL','minADE')} | {cell('FULL','minFDE')} | {cell('FULL','JADE')} | {cell('FULL','JFDE')} |
| HALF | {cell('HALF','minADE')} | {cell('HALF','minFDE')} | {cell('HALF','JADE')} | {cell('HALF','JFDE')} |
| matched GDTS | {cell('matched_GDTS','minADE')} | {cell('matched_GDTS','minFDE')} | {cell('matched_GDTS','JADE')} | {cell('matched_GDTS','JFDE')} |

## HALF − FULL 逐 seed 差值

负值表示 HALF 更低。

| seed | minADE | minFDE | JADE | JFDE |
|---:|---:|---:|---:|---:|
{delta_rows}

HALF 优于 FULL 的 seed 数：`{improved}`。

跨折方向：**{cross_fold}**。未满足项：`{failed_cross}`。

原 HOTEL 性能合同（相对 matched GDTS 四项均不退化，零容差）：**{original_contract}**。未满足项：`{failed_contract}`。

本轮新增 GPU 墙钟：**{gpu_seconds:.2f} / 1800 秒**。

证据边界：这是单训练 checkpoint 的五次推理随机重复，且 HOTEL validation/test 为镜像窗口；不是五训练种子、统计显著性检验或重新训练的结构消融。原 epoch45 模型及正式结果保持不变。
"""

    archive = Path(args.archive)
    archive.mkdir(parents=True, exist_ok=True)
    atomic_json(archive / "RESULTS.json", results)
    atomic_json(archive / "FULL_REUSED_PER_SEED.json", full)
    atomic_json(archive / "HALF_PER_SEED.json", half)
    atomic_json(archive / "GDTS_REUSED_PER_SEED.json", gdts)
    atomic_json(archive / "GPU_LEDGER.json", ledger_rows)
    atomic_text(archive / "REPORT_ZH.md", report)
    shutil.copyfile(args.config, archive / "HOTEL_CONFIG.json")
    print(json.dumps({"status": "COMPLETE", "cross_fold": cross_fold,
                      "contract": original_contract, "gpu_seconds": gpu_seconds},
                     sort_keys=True))


if __name__ == "__main__":
    main()
