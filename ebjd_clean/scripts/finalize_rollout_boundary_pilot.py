#!/usr/bin/env python3
"""Create compact receipts and the Chinese report for the rollout boundary pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import statistics
from pathlib import Path

import numpy as np


METRICS = ("minADE", "minFDE", "JADE", "JFDE")
SEEDS = tuple(range(2035, 2040))


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def vectors(payload: dict) -> dict[str, list[float]]:
    if "per_seed" in payload:
        return {metric: [float(row["metrics"][metric])
                         for row in payload["per_seed"]] for metric in METRICS}
    mapping = {"minADE": "minADE@K", "minFDE": "minFDE@K",
               "JADE": "JADE", "JFDE": "JFDE"}
    return {metric: [float(row[mapping[metric]]) for row in payload["runs"]]
            for metric in METRICS}


def summarize(payload: dict) -> dict:
    values = vectors(payload)
    return {metric: {
        "mean": float(statistics.fmean(values[metric])),
        "sample_standard_deviation": float(statistics.stdev(values[metric])),
        "per_seed": values[metric],
    } for metric in METRICS}


def comparison(left: dict, right: dict) -> dict:
    lv, rv = vectors(left), vectors(right)
    result = {}
    for metric in METRICS:
        delta = [a - b for a, b in zip(lv[metric], rv[metric], strict=True)]
        result[metric] = {
            "mean_delta": float(statistics.fmean(delta)),
            "per_seed_delta": [float(value) for value in delta],
            "improved_seed_count": int(sum(value < 0 for value in delta)),
            "unchanged_seed_count": int(sum(value == 0 for value in delta)),
            "worsened_seed_count": int(sum(value > 0 for value in delta)),
        }
    return result


def strata(payload: dict) -> dict:
    output = {}
    for name in ("N=1", "N>=2"):
        rows = [row["strata"][name] for row in payload["per_seed"]]
        output[name] = {
            "scene_count": int(rows[0]["scene_count"]),
            "agent_occurrence_count": int(rows[0]["agent_count"]),
            "metrics": {metric: {
                "mean": float(statistics.fmean(float(row[metric]) for row in rows)),
                "sample_standard_deviation": float(statistics.stdev(
                    float(row[metric]) for row in rows)),
            } for metric in METRICS},
        }
    return output


def endpoint_seed(path: Path) -> dict:
    bank = np.load(path, allow_pickle=True)
    oracle, single, centered = [], [], []
    for index in range(len(bank["scene_ids"])):
        prediction = np.asarray(bank["predictions"][index], dtype=np.float64)
        future = np.asarray(bank["future"][index], dtype=np.float64)
        endpoints = prediction[..., -1, :]
        target = future[:, -1, :]
        error = np.linalg.norm(endpoints - target[None], axis=-1)
        oracle.extend(error.min(0).tolist())
        single.extend(error.reshape(-1).tolist())
        centre = endpoints.mean(0)
        centered.extend(np.sqrt(np.mean(np.sum(
            (endpoints - centre[None]) ** 2, axis=-1), axis=0)).tolist())
    oracle_array = np.asarray(oracle)
    return {
        "agent_occurrence_count": int(len(oracle)),
        "endpoint_recall_0p5": float(np.mean(oracle_array <= 0.5)),
        "endpoint_recall_1p0": float(np.mean(oracle_array <= 1.0)),
        "endpoint_centered_RMS_relative_to_candidate_mean": float(np.mean(centered)),
        "average_single_candidate_FDE": float(np.mean(single)),
    }


def endpoint_group(root: Path) -> dict:
    per_seed = {str(seed): endpoint_seed(root / f"seed_{seed}.npz")
                for seed in SEEDS}
    names = [name for name in per_seed[str(SEEDS[0])]
             if name != "agent_occurrence_count"]
    return {"per_seed": per_seed, "aggregate": {name: {
        "mean": float(statistics.fmean(per_seed[str(seed)][name] for seed in SEEDS)),
        "sample_standard_deviation": float(statistics.stdev(
            per_seed[str(seed)][name] for seed in SEEDS)),
    } for name in names}}


def read_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def training_summary(run: Path) -> dict:
    epochs = read_lines(run / "EPOCH_SUMMARY.jsonl")
    return {
        "epochs": [int(row["epoch"]) for row in epochs],
        "updates": int(sum(row["updates"] for row in epochs)),
        "wall_seconds": float(sum(row["wall_seconds"] for row in epochs)),
        "rollout_updates": int(sum(row["rollout_updates"] for row in epochs)),
        "projection_executions": int(sum(row.get("projection_executions", 0)
                                         for row in epochs)),
        "projection_active_updates": int(sum(row["projection_active_updates"]
                                             for row in epochs)),
        "peak_cuda_allocated_bytes": int(max(
            row["peak_cuda_allocated_bytes"] for row in epochs)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-dir", required=True)
    parser.add_argument("--shared-run", required=True)
    parser.add_argument("--on-run", required=True)
    parser.add_argument("--off-run", required=True)
    parser.add_argument("--epoch10-result", required=True)
    parser.add_argument("--gdts-result", required=True)
    parser.add_argument("--epoch20-result", required=True)
    parser.add_argument("--on-result", required=True)
    parser.add_argument("--off-result", required=True)
    parser.add_argument("--epoch20-banks", required=True)
    parser.add_argument("--on-banks", required=True)
    parser.add_argument("--off-banks", required=True)
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    protocol = Path(args.protocol_dir).resolve()
    shared, on, off = (Path(args.shared_run).resolve(),
                       Path(args.on_run).resolve(), Path(args.off_run).resolve())
    payloads = {
        "epoch10": read_json(args.epoch10_result),
        "epoch20": read_json(args.epoch20_result),
        "off_epoch25": read_json(args.off_result),
        "on_epoch25": read_json(args.on_result),
        "matched_gdts": read_json(args.gdts_result),
    }
    expected_manifest = (
        "03f9b6bef89d43137b31782bbf0eb5b16d19f8ddfe378e2c7c62e700e128397b")
    for name in ("epoch10", "epoch20", "off_epoch25", "on_epoch25"):
        if payloads[name]["manifest_sha256"] != expected_manifest:
            raise ValueError(f"{name} manifest identity mismatch")
    expected_epoch10 = (
        "b339ab96c288fe0757d2bce75f6408fa80dfde3b030a1109af68b4a760aa4f12")
    if payloads["epoch10"]["checkpoint_sha256"] != expected_epoch10:
        raise ValueError("reused epoch-10 result identity mismatch")

    on_log, off_log = read_lines(on / "train.jsonl"), read_lines(off / "train.jsonl")
    fields = (
        "epoch", "update", "logical_scene_agent_identity_sha256",
        "augmentation_sha256", "base_random_draw_sha256",
        "learning_rate", "unet_learning_rate")
    mismatches = []
    if len(on_log) != len(off_log):
        mismatches.append({"field": "record_count", "on": len(on_log),
                           "off": len(off_log)})
    for index, (left, right) in enumerate(zip(on_log, off_log)):
        for field in fields:
            if left[field] != right[field]:
                mismatches.append({"index": index, "field": field,
                                   "on": left[field], "off": right[field]})
                if len(mismatches) >= 20:
                    break
        if len(mismatches) >= 20:
            break
    on_rollout = [row for row in on_log if row["rollout_update"]]
    pairing = {
        "schema": "ebjd-eth-rollout-boundary-pairing-v1",
        "status": "PASS" if not mismatches else "FAIL",
        "fields_checked": list(fields), "updates_per_arm": len(on_log),
        "mismatches": mismatches,
        "shared_epoch20_checkpoint_sha256": sha256(shared / "epoch_020.pt"),
        "on_origin_sha256": read_json(on / "MIGRATION_RECEIPT.json")[
            "origin_checkpoint_sha256"],
        "off_origin_sha256": read_json(off / "OBJECTIVE_FORK_RECEIPT.json")[
            "origin_checkpoint_sha256"],
        "on_rollout_updates": int(sum(row["rollout_update"] for row in on_log)),
        "off_rollout_updates": int(sum(row["rollout_update"] for row in off_log)),
        "on_projection_executions": int(sum(row["projection_executed"] for row in on_log)),
        "off_projection_executions": int(sum(row["projection_executed"] for row in off_log)),
        "on_rollout_weight_range": [
            float(min(row["active_rollout_weight"] for row in on_rollout)),
            float(max(row["active_rollout_weight"] for row in on_rollout))],
    }
    if len(on_log) != len(off_log):
        pairing["status"] = "FAIL"
    if pairing["on_origin_sha256"] != pairing["shared_epoch20_checkpoint_sha256"] or \
            pairing["off_origin_sha256"] != pairing["shared_epoch20_checkpoint_sha256"]:
        pairing["status"] = "FAIL"
    if pairing["off_rollout_updates"] or pairing["off_projection_executions"]:
        pairing["status"] = "FAIL"

    summaries = {name: summarize(payload) for name, payload in payloads.items()}
    comparisons = {
        "on_minus_off": comparison(payloads["on_epoch25"], payloads["off_epoch25"]),
        "on_minus_epoch20": comparison(payloads["on_epoch25"], payloads["epoch20"]),
        "off_minus_epoch20": comparison(payloads["off_epoch25"], payloads["epoch20"]),
    }
    strata_results = {name: strata(payloads[name]) for name in
                       ("epoch20", "off_epoch25", "on_epoch25")}
    endpoint = {
        "epoch20": endpoint_group(Path(args.epoch20_banks)),
        "off_epoch25": endpoint_group(Path(args.off_banks)),
        "on_epoch25": endpoint_group(Path(args.on_banks)),
        "RMS_definition": "RMS endpoint radius relative to each agent's 20-candidate mean",
    }
    training = {
        "shared_11_20": training_summary(shared),
        "off_21_25": training_summary(off),
        "on_21_25": training_summary(on),
    }
    checkpoints = {
        "epoch20": sha256(shared / "epoch_020.pt"),
        "off_epoch25": sha256(off / "epoch_025.pt"),
        "on_epoch25": sha256(on / "epoch_025.pt"),
    }
    on_off = comparisons["on_minus_off"]
    on_e20 = comparisons["on_minus_epoch20"]
    if all(on_off[m]["mean_delta"] < 0 for m in METRICS) and \
            all(on_e20[m]["mean_delta"] <= 0 for m in METRICS):
        rollout_judgment = "SUPPORTED: ON优于OFF且维持或超过epoch20"
    elif all(on_off[m]["mean_delta"] > 0 for m in METRICS) and \
            all(comparisons["off_minus_epoch20"][m]["mean_delta"] <= 0
                for m in METRICS):
        rollout_judgment = "CONFLICT: ON退化而OFF保持或改善"
    elif all(on_e20[m]["mean_delta"] > 0 for m in METRICS) and \
            all(comparisons["off_minus_epoch20"][m]["mean_delta"] > 0
                for m in METRICS):
        rollout_judgment = "COMMON_DEGRADATION: 两臂均退化"
    else:
        rollout_judgment = "MIXED: 四项方向混合，逐项报告"
    gdts = summaries["matched_gdts"]
    on_summary = summaries["on_epoch25"]
    contract_checks = {
        "minADE_not_worse": on_summary["minADE"]["mean"] <= gdts["minADE"]["mean"],
        "minFDE_not_worse": on_summary["minFDE"]["mean"] <= gdts["minFDE"]["mean"],
        "JADE_strictly_better": on_summary["JADE"]["mean"] < gdts["JADE"]["mean"],
        "JFDE_strictly_better": on_summary["JFDE"]["mean"] < gdts["JFDE"]["mean"],
    }
    results = {
        "schema": "ebjd-eth-rollout-boundary-results-v1",
        "status": "COMPLETED",
        "protocol": read_json(protocol / "PLAN.json"),
        "metrics": summaries, "comparisons": comparisons,
        "strata": strata_results, "endpoint_diagnostics": endpoint,
        "training": training, "pairing": pairing,
        "checkpoint_sha256": checkpoints,
        "preflight": read_json(args.preflight),
        "rollout_judgment": rollout_judgment,
        "original_contract": {
            "status": "PASS" if all(contract_checks.values()) else "FAIL",
            "checks": contract_checks,
        },
        "limitations": [
            "one training seed (3101)",
            "five inference seeds are not five training repeats",
            "validation/test mirror; independent_test=false",
            "epoch25 covers only the early rollout ramp",
            "ON/OFF tests the combined rollout-loss plus projection mechanism",
        ],
    }
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    for name in ("PLAN.json", "SHARED_CONFIG.json", "ON_CONFIG.json",
                 "OFF_CONFIG.json", "COMMANDS.md"):
        shutil.copy2(protocol / name, output / name)
    atomic_json(output / "PAIRING_RECEIPT.json", pairing)
    atomic_json(output / "RESULTS_AND_DIAGNOSTICS.json", results)
    atomic_json(output / "TEST_AND_QA_RECEIPT.json", {
        "preflight": read_json(args.preflight),
        "pairing_status": pairing["status"],
        "formal_results_status": "COMPLETED",
    })

    def cell(name: str, metric: str) -> str:
        value = summaries[name][metric]
        return f'{value["mean"]:.6f} ± {value["sample_standard_deviation"]:.6f}'

    rows = []
    labels = {"epoch10": "Candidate epoch 10", "epoch20": "共享 epoch 20",
              "off_epoch25": "OFF epoch 25", "on_epoch25": "ON epoch 25",
              "matched_gdts": "Matched GDTS"}
    for name in ("epoch10", "epoch20", "off_epoch25", "on_epoch25", "matched_gdts"):
        rows.append("| " + labels[name] + " | " + " | ".join(
            cell(name, metric) for metric in METRICS) + " |")
    report = f"""# EBJD ETH rollout 启用边界配对验证

> 状态：已完成。训练种子仅 3101；2035–2039 是推理种子；`independent_test=false`。

## 主要结果

| 对象 | minADE | minFDE | JADE | JFDE |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

## 结论

- rollout 判断：**{rollout_judgment}**。
- ON/OFF 配对：**{pairing['status']}**；两臂各 {pairing['updates_per_arm']} 次更新，ON rollout/projection 为 {pairing['on_rollout_updates']}/{pairing['on_projection_executions']}，OFF 为 {pairing['off_rollout_updates']}/{pairing['off_projection_executions']}。
- 原 matched-GDTS 合同：**{results['original_contract']['status']}**。
- ON−OFF、ON−epoch20、OFF−epoch20 的逐指标与逐 seed 差值、N=1/N≥2 分层和终点覆盖诊断见 `RESULTS_AND_DIAGNOSTICS.json`。
- centered RMS 定义为每名行人的 20 个候选终点相对其候选均值的 RMS 半径。

## 工程收据

- epoch20 SHA256：`{checkpoints['epoch20']}`
- OFF epoch25 SHA256：`{checkpoints['off_epoch25']}`
- ON epoch25 SHA256：`{checkpoints['on_epoch25']}`
- 共享阶段耗时：{training['shared_11_20']['wall_seconds']/3600:.3f} h；OFF：{training['off_21_25']['wall_seconds']/3600:.3f} h；ON：{training['on_21_25']['wall_seconds']/3600:.3f} h。
- 峰值显存（allocated）：共享 {training['shared_11_20']['peak_cuda_allocated_bytes']/2**30:.3f} GiB，OFF {training['off_21_25']['peak_cuda_allocated_bytes']/2**30:.3f} GiB，ON {training['on_21_25']['peak_cuda_allocated_bytes']/2**30:.3f} GiB。

## 解释边界

本实验仅检验 native-goal 模型上“rollout loss + marginal actual-step projection”整套机制的早期 ramp；不单独归因某个损失项，不证明两模块存在交互增益，也不把五个推理种子解释为五次独立训练。
"""
    Path(args.report).write_text(report, encoding="utf-8")
    print(json.dumps({
        "status": "COMPLETED", "pairing": pairing["status"],
        "rollout_judgment": rollout_judgment,
        "contract": results["original_contract"]["status"],
    }, sort_keys=True))
    if pairing["status"] != "PASS":
        raise RuntimeError("formal ON/OFF pairing failed")


if __name__ == "__main__":
    main()
