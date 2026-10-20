#!/usr/bin/env python3
"""Finalize the paired three-epoch fixed-anchor path-loss pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
from pathlib import Path

import torch


METRICS = ("minADE", "minFDE", "JADE", "JFDE")
SEEDS = (2035, 2036, 2037, 2038, 2039)


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluation(payload: dict) -> dict:
    by_seed = {int(row["seed"]): row["metrics"] for row in payload["per_seed"]}
    return {metric: {
        "mean": float(payload["mean"][metric]),
        "sample_standard_deviation": float(payload["sample_standard_deviation"][metric]),
        "per_seed": [float(by_seed[seed][metric]) for seed in SEEDS],
    } for metric in METRICS}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--control-run", type=Path, required=True)
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--control-result", type=Path, required=True)
    parser.add_argument("--candidate-result", type=Path, required=True)
    parser.add_argument("--parent-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    plan = read(args.protocol / "PLAN.json")
    parent_results = read(args.parent_results)
    control_payload, candidate_payload = read(args.control_result), read(args.candidate_result)
    parent = parent_results["metrics"]["fixed_anchor"]
    gdts = parent_results["metrics"]["matched_GDTS"]
    control, candidate = evaluation(control_payload), evaluation(candidate_payload)
    training_rows = {
        "control": rows(args.control_run / "train.jsonl"),
        "candidate": rows(args.candidate_run / "train.jsonl"),
    }
    summaries = {
        "control": rows(args.control_run / "EPOCH_SUMMARY.jsonl"),
        "candidate": rows(args.candidate_run / "EPOCH_SUMMARY.jsonl"),
    }
    checkpoints = {}
    checkpoint_headers = {}
    for arm, run in (("control", args.control_run), ("candidate", args.candidate_run)):
        path = run / "epoch_028.pt"
        checkpoints[arm] = sha256(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        checkpoint_headers[arm] = {
            "epoch": int(payload["epoch"]), "update_index": int(payload["update_index"]),
            "schema": payload["schema"], "epoch_boundary_only": payload["epoch_boundary_only"],
            "path_weight": payload["resolved_config"]["objectives"][
                "fixed_anchor_path_weight"],
        }
    pair_fields = (
        "logical_scene_agent_identity_sha256", "augmentation_sha256",
        "base_random_draw_sha256", "learning_rate", "unet_learning_rate")
    control_rows, candidate_rows = training_rows["control"], training_rows["candidate"]
    paired_mismatches = sum(
        any(left[field] != right[field] for field in pair_fields)
        for left, right in zip(control_rows, candidate_rows))
    all_records = control_rows + candidate_rows
    finite = all(math.isfinite(float(value)) for row in all_records for value in row.values()
                 if isinstance(value, (int, float)))
    checks = {
        "three_epochs_each": all([row["epoch"] for row in summaries[arm]] == [26, 27, 28]
                                 for arm in summaries),
        "771_updates_each": all(len(training_rows[arm]) == 771 for arm in training_rows),
        "terminal_update_7196": all(
            checkpoint_headers[arm]["update_index"] == 7196
            and checkpoint_headers[arm]["epoch"] == 28 for arm in checkpoint_headers),
        "paired_inputs_all_updates": paired_mismatches == 0
        and len(control_rows) == len(candidate_rows),
        "control_weight_zero": all(row["active_fixed_anchor_path_weight"] == 0
                                   and row["fixed_anchor_path"] == 0 for row in control_rows),
        "candidate_weight_point_one": all(
            row["active_fixed_anchor_path_weight"] == 0.1
            and row["fixed_anchor_path"] > 0 for row in candidate_rows),
        "rollout_projection_off": all(
            row["rollout_update"] == 0 and row["projection_executed"] == 0
            for row in all_records),
        "all_training_values_finite": finite,
        "formal_eval_identity": all(
            payload["manifest_sha256"] == plan.get("manifest_sha256", payload["manifest_sha256"])
            and payload["worlds"] == 20 and payload["steps"] == 20
            and payload["ddim_eta"] == 0 and payload["endpoint_projection_radius"] == 0
            for payload in (control_payload, candidate_payload)),
        "eval_checkpoint_binding": (
            control_payload["checkpoint_sha256"] == checkpoints["control"]
            and candidate_payload["checkpoint_sha256"] == checkpoints["candidate"]),
    }
    minfde_spread = max(
        abs(block["minFDE"]["per_seed"][index] - parent["minFDE"]["per_seed"][index])
        for block in (control, candidate) for index in range(5))
    checks["rho0_minfde_preserved"] = minfde_spread <= 1e-6
    preflight = read(args.assets / "CUDA_PREFLIGHT_RECEIPT.json")
    checks["cuda_preflight_pass"] = preflight["status"] == "PASS"
    delta_control = {metric: candidate[metric]["mean"] - control[metric]["mean"]
                     for metric in METRICS}
    delta_parent = {metric: candidate[metric]["mean"] - parent[metric]["mean"]
                    for metric in METRICS}
    delta_gdts = {metric: candidate[metric]["mean"] - gdts[metric]["mean"]
                  for metric in METRICS}
    improvement_count = {metric: sum(
        candidate[metric]["per_seed"][i] < control[metric]["per_seed"][i]
        for i in range(5)) for metric in METRICS}
    candidate_pass = (
        delta_control["minADE"] < 0 and delta_parent["minADE"] < 0
        and delta_control["JADE"] <= 0 and delta_parent["JADE"] <= 0
        and delta_control["JFDE"] <= 0 and delta_parent["JFDE"] <= 0)
    original_checks = {
        "minADE_not_worse": delta_gdts["minADE"] <= 0,
        "minFDE_not_worse": delta_gdts["minFDE"] <= 0,
        "JADE_strictly_better": delta_gdts["JADE"] < 0,
        "JFDE_strictly_better": delta_gdts["JFDE"] < 0,
    }
    training_seconds = sum(row["wall_seconds"] for arm in summaries for row in summaries[arm])
    validation_seconds = sum(row["validation_seconds"] for arm in summaries for row in summaries[arm])
    evaluation_seconds = sum(float((args.assets / f"{arm}_evaluation_seconds.txt").read_text())
                             for arm in ("control", "candidate"))
    peak = max(int(row["peak_cuda_allocated_bytes"])
               for arm in summaries for row in summaries[arm])
    receipt = {
        "schema": "ebjd-no-token-fixed-anchor-path-pilot-results-v1",
        "status": "COMPLETE" if all(checks.values()) else "INVALID",
        "source_commit": plan["source_commit"], "training_seed": 3101,
        "inference_seeds": list(SEEDS), "independent_test": False,
        "metrics": {"parent_fixed_anchor": parent, "control": control,
                    "candidate": candidate, "matched_GDTS": gdts},
        "candidate_minus_control": delta_control,
        "candidate_minus_parent": delta_parent,
        "candidate_minus_matched_GDTS": delta_gdts,
        "candidate_improves_control_seed_count": improvement_count,
        "candidate_screen": "PASS" if candidate_pass else "FAIL",
        "original_contract": "PASS" if all(original_checks.values()) else "FAIL",
        "original_contract_checks": original_checks,
        "checks": checks, "paired_update_mismatches": paired_mismatches,
        "rho0_minfde_max_abs_delta": minfde_spread,
        "checkpoint_sha256": checkpoints, "checkpoint_headers": checkpoint_headers,
        "cost": {
            "preflight_seconds": preflight["wall_seconds"],
            "training_and_validation_wall_seconds": training_seconds,
            "validation_wall_seconds_included": validation_seconds,
            "formal_evaluation_wall_seconds": evaluation_seconds,
            "total_gpu_wall_seconds": preflight["wall_seconds"] + training_seconds + evaluation_seconds,
            "peak_cuda_allocated_bytes": peak,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)

    def cell(block: dict, metric: str) -> str:
        return f'{block[metric]["mean"]:.4f} ± {block[metric]["sample_standard_deviation"]:.4f}'
    report = f"""# NO_TOKEN 固定终点路径监督短试验

## 结论

- 配对与完整性：**{'PASS' if all(checks.values()) else 'FAIL'}**；两臂各771 updates，固定epoch28，逐update输入/RNG/LR一致，rollout/projection为0。
- 候选标准：**{'PASS' if candidate_pass else 'FAIL'}**。
- 原合同：**{'PASS' if all(original_checks.values()) else 'FAIL'}**。
- 单训练seed3101；五个推理seed不是独立训练重复，`independent_test=false`。

| 输出 | minADE ↓ | minFDE ↓ | JADE ↓ | JFDE ↓ |
|---|---:|---:|---:|---:|
| 父NO_TOKEN epoch25 Fixed-anchor | {cell(parent, 'minADE')} | {cell(parent, 'minFDE')} | {cell(parent, 'JADE')} | {cell(parent, 'JFDE')} |
| Control epoch28 Fixed-anchor | {cell(control, 'minADE')} | {cell(control, 'minFDE')} | {cell(control, 'JADE')} | {cell(control, 'JFDE')} |
| Candidate epoch28 Fixed-anchor | {cell(candidate, 'minADE')} | {cell(candidate, 'minFDE')} | {cell(candidate, 'JADE')} | {cell(candidate, 'JFDE')} |
| matched GDTS | {cell(gdts, 'minADE')} | {cell(gdts, 'minFDE')} | {cell(gdts, 'JADE')} | {cell(gdts, 'JFDE')} |

Candidate−Control：{', '.join(f'{m} {delta_control[m]:+.6f}' for m in METRICS)}。逐seed改善：{', '.join(f'{m} {improvement_count[m]}/5' for m in METRICS)}。

Candidate−父模型：{', '.join(f'{m} {delta_parent[m]:+.6f}' for m in METRICS)}。Candidate−GDTS：{', '.join(f'{m} {delta_gdts[m]:+.6f}' for m in METRICS)}。

ρ=0 minFDE最大浮点差 `{minfde_spread:.3e}`，不解释为训练收益。GPU总墙钟 `{receipt['cost']['total_gpu_wall_seconds']:.1f}` 秒，峰值 `{peak / 2**30:.2f}` GiB；其中正式训练/验证 `{training_seconds:.1f}` 秒、五种子评估 `{evaluation_seconds:.1f}` 秒、预检 `{preflight['wall_seconds']:.1f}` 秒。

唯一下一步：{'保留λ=0.10路径监督进入独立训练seed复验。' if candidate_pass else '停止该λ=0.10路径监督分支，不调权重、不延长训练。'}
"""
    args.report.write_text(report)
    print(json.dumps({"status": receipt["status"], "candidate": receipt["candidate_screen"],
                      "contract": receipt["original_contract"]}, sort_keys=True))


if __name__ == "__main__":
    main()
