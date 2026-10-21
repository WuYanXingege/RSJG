#!/usr/bin/env python3
"""Audit and archive the fixed-epoch UNIV NO_TOKEN transfer experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import statistics
from pathlib import Path

import torch


METRICS = ("minADE", "minFDE", "JADE", "JFDE")
SEEDS = tuple(range(2035, 2040))


def read(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines()
            if line.strip()]


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


def metrics(payload: dict) -> dict:
    by_seed = {int(row["seed"]): row["metrics"] for row in payload["per_seed"]}
    return {name: {
        "mean": float(payload["mean"][name]),
        "sample_standard_deviation": float(payload["sample_standard_deviation"][name]),
        "per_seed": [float(by_seed[seed][name]) for seed in SEEDS],
    } for name in METRICS}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--control-run", type=Path, required=True)
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--control-result", type=Path, required=True)
    parser.add_argument("--candidate-raw-result", type=Path, required=True)
    parser.add_argument("--candidate-fixed-result", type=Path, required=True)
    parser.add_argument("--parent-result", type=Path, required=True)
    parser.add_argument("--gdts-comparison", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    plan = read(args.protocol / "PLAN.json")
    preflight = read(args.assets / "CUDA_PREFLIGHT_RECEIPT.json")
    sources = {
        "parent_raw": read(args.parent_result),
        "control_raw": read(args.control_result),
        "candidate_raw": read(args.candidate_raw_result),
        "candidate_fixed": read(args.candidate_fixed_result),
    }
    table = {name: metrics(value) for name, value in sources.items()}
    gdts_source = read(args.gdts_comparison)
    gdts = {}
    for name in METRICS:
        rows = [row for row in gdts_source["per_seed"] if row["metric"] == name]
        values = [float(row["GDTS"]) for row in sorted(rows, key=lambda x: x["seed"])]
        gdts[name] = {"mean": statistics.mean(values),
                      "sample_standard_deviation": statistics.stdev(values),
                      "per_seed": values}
    table["matched_GDTS"] = gdts

    train_rows = {"control": jsonl(args.control_run / "train.jsonl"),
                  "candidate": jsonl(args.candidate_run / "train.jsonl")}
    summaries = {"control": jsonl(args.control_run / "EPOCH_SUMMARY.jsonl"),
                 "candidate": jsonl(args.candidate_run / "EPOCH_SUMMARY.jsonl")}
    checkpoint_paths = {"control": args.control_run / "epoch_081.pt",
                        "candidate": args.candidate_run / "epoch_081.pt"}
    checkpoints, headers = {}, {}
    for arm, path in checkpoint_paths.items():
        checkpoints[arm] = sha256(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        headers[arm] = {"epoch": int(payload["epoch"]),
                        "update_index": int(payload["update_index"]),
                        "schema": payload["schema"],
                        "epoch_boundary_only": payload["epoch_boundary_only"]}
    paired_fields = ("logical_scene_agent_identity_sha256", "augmentation_sha256",
                     "base_random_draw_sha256", "learning_rate", "unet_learning_rate")
    paired_mismatches = sum(
        any(left[key] != right[key] for key in paired_fields)
        for left, right in zip(train_rows["control"], train_rows["candidate"], strict=True))
    all_rows = train_rows["control"] + train_rows["candidate"]
    finite = all(math.isfinite(float(value)) for row in all_rows
                 for value in row.values() if isinstance(value, (int, float)))
    formal_eval_identity = all(
        payload["manifest_sha256"] == plan["assets"]["heldout_manifest_sha256"]
        and payload["scene_count"] == 947
        and payload["agent_occurrence_count"] == 24334
        and payload["worlds"] == 20 and payload["steps"] == 20
        and payload["ddim_eta"] == 0 and payload["model_dtype"] == "fp32"
        for name, payload in sources.items() if name != "parent_raw")
    parent_mirror_identity = all([
        sources["parent_raw"]["scene_count"] == 947,
        sources["parent_raw"]["agent_occurrence_count"] == 24334,
        sources["parent_raw"]["worlds"] == 20,
        sources["parent_raw"]["steps"] == 20,
        sources["parent_raw"]["ddim_eta"] == 0,
        sources["parent_raw"]["model_dtype"] == "fp32",
    ])
    fixed_oracle_error = max(abs(
        row["metrics"]["minFDE"] - row["native_goal_oracle_FDE"])
        for row in sources["candidate_fixed"]["per_seed"])
    checks = {
        "preflight_pass": preflight["status"] == "PASS",
        "eight_epochs_each": all([row["epoch"] for row in summaries[arm]]
                                 == list(range(74, 82)) for arm in summaries),
        "2040_updates_each": all(len(train_rows[arm]) == 2040 for arm in train_rows),
        "terminal_epoch81_update20655": all(
            value["epoch"] == 81 and value["update_index"] == 20655
            and value["schema"] == "ebjd-checkpoint-v2"
            and value["epoch_boundary_only"] is True for value in headers.values()),
        "all_updates_paired": paired_mismatches == 0,
        "rollout_projection_off": all(
            row["rollout_update"] == 0 and row["projection_executed"] == 0
            for row in all_rows),
        "path_loss_off": all(
            row["fixed_anchor_path"] == 0
            and row["active_fixed_anchor_path_weight"] == 0 for row in all_rows),
        "finite_training": finite,
        "formal_eval_identity": formal_eval_identity,
        "parent_test_mirror_identity": parent_mirror_identity,
        "evaluation_checkpoint_binding": (
            sources["control_raw"]["checkpoint_sha256"] == checkpoints["control"]
            and sources["candidate_raw"]["checkpoint_sha256"] == checkpoints["candidate"]
            and sources["candidate_fixed"]["checkpoint_sha256"] == checkpoints["candidate"]),
        "candidate_raw_fixed_same_forward": bool(
            sources["candidate_fixed"].get("derived_from_same_forward_as_raw"))
            and Path(sources["candidate_fixed"]["raw_output"]).resolve()
            == args.candidate_raw_result.resolve(),
        "fixed_endpoint_is_native_oracle": fixed_oracle_error <= 1e-6,
        "gpu_budget_not_exceeded": True,
    }
    parent, control, candidate = (
        table["parent_raw"], table["control_raw"], table["candidate_fixed"])
    delta_parent = {name: candidate[name]["mean"] - parent[name]["mean"]
                    for name in METRICS}
    delta_control = {name: candidate[name]["mean"] - control[name]["mean"]
                     for name in METRICS}
    delta_gdts = {name: candidate[name]["mean"] - gdts[name]["mean"]
                  for name in METRICS}
    improved = {baseline: {name: sum(
        candidate[name]["per_seed"][i] < table[baseline][name]["per_seed"][i]
        for i in range(5)) for name in METRICS}
        for baseline in ("parent_raw", "control_raw", "matched_GDTS")}
    upgrade = all(delta[name] <= 0 for delta in (delta_parent, delta_control)
                  for name in ("minADE", "minFDE")) and all(
        delta[name] < 0 for delta in (delta_parent, delta_control)
        for name in ("JADE", "JFDE"))
    original = (delta_gdts["minADE"] <= 0 and delta_gdts["minFDE"] <= 0
                and delta_gdts["JADE"] < 0 and delta_gdts["JFDE"] < 0)
    bank_seconds = float(plan["assets"]["bank_export_elapsed_seconds"])
    training_seconds = sum(float(row["wall_seconds"])
                           for arm in summaries for row in summaries[arm])
    eval_seconds = sum(float((args.assets / name).read_text()) for name in (
        "control_evaluation_seconds.txt", "candidate_evaluation_seconds.txt"))
    failed_eval_path = args.assets / "failed_control_evaluation_seconds.txt"
    failed_evaluation_seconds = (
        float(failed_eval_path.read_text()) if failed_eval_path.exists() else 0.0)
    failed_preflight_path = args.assets / "CUDA_PREFLIGHT_RECEIPT_failed_9f0eb7c.json"
    failed_preflight_seconds = (
        float(read(failed_preflight_path)["wall_seconds"])
        if failed_preflight_path.exists() else 0.0)
    discarded_path = args.assets / "recovery_discarded_gpu_seconds.txt"
    discarded_training_seconds = (
        float(discarded_path.read_text()) if discarded_path.exists() else 0.0)
    total = (bank_seconds + float(preflight["wall_seconds"])
             + failed_preflight_seconds + training_seconds
             + discarded_training_seconds + failed_evaluation_seconds
             + eval_seconds)
    checks["gpu_budget_not_exceeded"] = total <= plan["protocol"]["gpu_budget_seconds"]
    status = "COMPLETE" if all(checks.values()) else "INVALID"
    result = {
        "schema": "ebjd-univ-no-token-transfer-results-v1", "status": status,
        "source_commit": plan["source_commit"], "training_seed": 3101,
        "inference_seeds": list(SEEDS), "independent_test": False,
        "metrics": table, "candidate_fixed_minus_parent_raw": delta_parent,
        "candidate_fixed_minus_control_raw": delta_control,
        "candidate_fixed_minus_matched_GDTS": delta_gdts,
        "candidate_improvement_seed_count": improved,
        "upgrade_contract": "PASS" if upgrade else "FAIL",
        "original_contract": "PASS" if original else "FAIL",
        "checks": checks, "paired_update_mismatches": paired_mismatches,
        "fixed_native_oracle_max_abs_error": fixed_oracle_error,
        "checkpoint_sha256": checkpoints, "checkpoint_headers": headers,
        "cost": {"bank_export_seconds": bank_seconds,
                 "preflight_seconds": float(preflight["wall_seconds"]),
                 "failed_preflight_seconds": failed_preflight_seconds,
                 "training_and_validation_seconds": training_seconds,
                 "discarded_recovery_training_seconds": discarded_training_seconds,
                 "failed_formal_evaluation_seconds": failed_evaluation_seconds,
                 "formal_evaluation_seconds": eval_seconds,
                 "total_gpu_wall_seconds": total,
                 "peak_cuda_allocated_bytes": max(
                     [int(preflight["peak_cuda_allocated_bytes"])]
                     + [int(row["peak_cuda_allocated_bytes"])
                        for arm in summaries for row in summaries[arm]])},
    }
    args.archive.mkdir(parents=True, exist_ok=True)
    for source, target in (
        (args.protocol / "PLAN.json", "PLAN.json"),
        (args.protocol / "CONTROL_CONFIG.json", "CONTROL_CONFIG.json"),
        (args.protocol / "CANDIDATE_CONFIG.json", "CANDIDATE_CONFIG.json"),
        (args.assets / "CUDA_PREFLIGHT_RECEIPT.json", "CUDA_PREFLIGHT_RECEIPT.json"),
    ):
        shutil.copy2(source, args.archive / target)
    atomic_text(args.archive / "RESULTS.json", json.dumps(result, indent=2, sort_keys=True) + "\n")

    def cell(block: dict, name: str) -> str:
        return f'{block[name]["mean"]:.4f} ± {block[name]["sample_standard_deviation"]:.4f}'
    labels = (("parent_raw", "Parent epoch73 Raw"),
              ("control_raw", "Control epoch81 Raw"),
              ("candidate_fixed", "Candidate epoch81 Fixed-anchor"),
              ("candidate_raw", "Candidate epoch81 Raw"),
              ("matched_GDTS", "Matched GDTS"))
    lines = ["# UNIV NO_TOKEN transfer：固定 epoch81 正式结果", "",
             f"- 完整性：**{status}**；upgrade contract：**{'PASS' if upgrade else 'FAIL'}**；原合同：**{'PASS' if original else 'FAIL'}**。",
             "- 单训练种子 3101；2035–2039 只度量推理随机性；validation/test 镜像，`independent_test=false`。", "",
             "| 输出 | minADE ↓ | minFDE ↓ | JADE ↓ | JFDE ↓ |",
             "|---|---:|---:|---:|---:|"]
    for key, label in labels:
        lines.append(f"| {label} | " + " | ".join(cell(table[key], name) for name in METRICS) + " |")
    retained = original
    recommendation = ("保留 NO_TOKEN Fixed-anchor 作为 UNIV 主候选，并进入独立训练种子复验。"
                      if upgrade and original else
                      "不升级 UNIV 主线；保留原 epoch73 模型，停止本迁移分支且不调参、不延长。")
    lines += ["", f"旧 UNIV 相对 GDTS 优势是否保留：**{'是' if retained else '否'}**。",
              f"Candidate Fixed−Parent Raw：{delta_parent}。",
              f"Candidate Fixed−Control Raw：{delta_control}。",
              f"实际 GPU 计费口径墙钟：{total:.1f} 秒（上限 {plan['protocol']['gpu_budget_seconds']} 秒）。",
              "", f"唯一建议：{recommendation}", ""]
    atomic_text(args.archive / "REPORT_ZH.md", "\n".join(lines))
    atomic_text(args.archive / "COMMANDS_AND_RECOVERY.md", "\n".join([
        "# 执行与恢复", "",
        "固定串行入口：", "",
        "```bash", "bash scripts/run_univ_no_token_transfer.sh", "```", "",
        "本轮正式顺序为 Control epoch74–81、Candidate epoch74–81、五种子评估和最终审计。",
        "检查点只允许从各臂最近完整 epoch 边界恢复；禁止采用中断 epoch 的 update 日志。", "",
    ]))
    print(json.dumps({"status": status, "upgrade": upgrade,
                      "original": original, "recommendation": recommendation},
                     ensure_ascii=False, sort_keys=True))
    if status != "COMPLETE":
        raise RuntimeError("UNIV transfer final audit invalid")


if __name__ == "__main__":
    main()
