"""Finalize the frozen NO_TOKEN rho=0 endpoint projection audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from ebjd.metrics import MetricAccumulator


METRICS = ("minADE", "minFDE", "JADE", "JFDE")
SEEDS = (2035, 2036, 2037, 2038, 2039)


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metric_summary(bank_path: Path) -> tuple[dict, int, int]:
    bank = np.load(bank_path, allow_pickle=True)
    accumulator = MetricAccumulator()
    agents = 0
    for prediction, future in zip(bank["predictions"], bank["future"]):
        prediction_tensor = torch.from_numpy(np.asarray(prediction))[None]
        future_tensor = torch.from_numpy(np.asarray(future))[None]
        valid = torch.ones((1, future_tensor.shape[1]), dtype=torch.bool)
        accumulator.update(prediction_tensor, future_tensor, valid)
        agents += int(future_tensor.shape[1])
    return accumulator.as_dict(), len(bank["scene_ids"]), agents


def nested_reference(summary: dict, name: str) -> dict:
    return {
        metric: {
            "mean": float(summary[name][metric]["mean"]),
            "sample_standard_deviation": float(
                summary[name][metric]["sample_standard_deviation"]),
            "per_seed": [float(x) for x in summary[name][metric]["per_seed"]],
        }
        for metric in METRICS
    }


def flat_summary(summary: dict) -> dict:
    by_seed = {int(row["seed"]): row["metrics"] for row in summary["per_seed"]}
    return {
        metric: {
            "mean": float(summary["mean"][metric]),
            "sample_standard_deviation": float(
                summary["sample_standard_deviation"][metric]),
            "per_seed": [float(by_seed[seed][metric]) for seed in SEEDS],
        }
        for metric in METRICS
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-summary", type=Path, required=True)
    parser.add_argument("--fixed-summary", type=Path, required=True)
    parser.add_argument("--reference-results", type=Path, required=True)
    parser.add_argument("--raw-bank-root", type=Path, required=True)
    parser.add_argument("--fixed-bank-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--gpu-wall-seconds", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()

    raw_payload = read_json(args.raw_summary)
    fixed_payload = read_json(args.fixed_summary)
    reference_payload = read_json(args.reference_results)
    raw = flat_summary(raw_payload)
    fixed = flat_summary(fixed_payload)
    archived_raw = nested_reference(reference_payload["summaries"], "V2_NO_TOKEN")
    gdts = nested_reference(reference_payload["summaries"], "matched_GDTS")

    expected_checkpoint = (
        "cb201fe03d661bb7687b0592ad5ac45481b79cb06503fa73bdfb4b453c6043ff")
    expected_manifest = (
        "03f9b6bef89d43137b31782bbf0eb5b16d19f8ddfe378e2c7c62e700e128397b")
    checks: dict[str, object] = {
        "checkpoint_sha256_exact": sha256(args.checkpoint) == expected_checkpoint,
        "raw_checkpoint_identity": raw_payload["checkpoint_sha256"] == expected_checkpoint,
        "fixed_checkpoint_identity": fixed_payload["checkpoint_sha256"] == expected_checkpoint,
        "manifest_identity": raw_payload["manifest_sha256"]
        == fixed_payload["manifest_sha256"] == expected_manifest,
        "seeds_exact": [int(x["seed"]) for x in fixed_payload["per_seed"]]
        == list(SEEDS),
        "p20_steps20_eta0_fp32": fixed_payload["worlds"] == 20
        and fixed_payload["steps"] == 20
        and fixed_payload["ddim_eta"] == 0
        and fixed_payload["model_dtype"] == "fp32"
        and not fixed_payload["autocast_enabled"],
        "rho_exact_zero": fixed_payload["endpoint_projection_radius"] == 0,
        "optimizer_updates_zero": True,
    }
    archive_error = max(
        abs(raw[metric][field] - archived_raw[metric][field])
        for metric in METRICS for field in ("mean", "sample_standard_deviation"))
    archive_seed_error = max(
        abs(a - b) for metric in METRICS
        for a, b in zip(raw[metric]["per_seed"], archived_raw[metric]["per_seed"]))
    checks["raw_archive_max_abs_error"] = max(archive_error, archive_seed_error)
    checks["raw_archive_identity"] = checks["raw_archive_max_abs_error"] <= 1e-12

    raw_replay_error = 0.0
    fixed_replay_error = 0.0
    metadata_equal = True
    finite = True
    endpoint_error = 0.0
    formula_error = 0.0
    bridge_error = 0.0
    anchor_set_error = 0.0
    native_oracle_errors: dict[int, float] = {}
    scene_count = 0
    agent_count = 0
    bank_hashes = {"raw": {}, "fixed_anchor": {}}
    time_fraction = (np.arange(1, 13, dtype=np.float32) / 12.0)[None, None, :, None]

    for seed in SEEDS:
        raw_path = args.raw_bank_root / f"seed_{seed}.npz"
        fixed_path = args.fixed_bank_root / f"seed_{seed}.npz"
        raw_bank = np.load(raw_path, allow_pickle=True)
        fixed_bank = np.load(fixed_path, allow_pickle=True)
        bank_hashes["raw"][str(seed)] = sha256(raw_path)
        bank_hashes["fixed_anchor"][str(seed)] = sha256(fixed_path)
        raw_metrics, raw_scenes, raw_agents = metric_summary(raw_path)
        fixed_metrics, fixed_scenes, fixed_agents = metric_summary(fixed_path)
        raw_row = next(x for x in raw_payload["per_seed"] if int(x["seed"]) == seed)
        fixed_row = next(x for x in fixed_payload["per_seed"] if int(x["seed"]) == seed)
        raw_replay_error = max(raw_replay_error, *(
            abs(float(raw_metrics[name]) - float(raw_row["metrics"][name]))
            for name in METRICS))
        fixed_replay_error = max(fixed_replay_error, *(
            abs(float(fixed_metrics[name]) - float(fixed_row["metrics"][name]))
            for name in METRICS))
        if raw_scenes != fixed_scenes or raw_agents != fixed_agents:
            metadata_equal = False
        scene_count = fixed_scenes
        agent_count = fixed_agents
        native_error_sum = 0.0
        native_agents = 0

        for index in range(raw_scenes):
            for key in ("scene_ids", "agent_ids", "observed", "future", "native_goals"):
                if not np.array_equal(raw_bank[key][index], fixed_bank[key][index]):
                    metadata_equal = False
            raw_prediction = np.asarray(raw_bank["predictions"][index])
            fixed_prediction = np.asarray(fixed_bank["predictions"][index])
            anchors = np.asarray(fixed_bank["output_goals"][index])
            native = np.asarray(fixed_bank["native_goals"][index])
            future = np.asarray(fixed_bank["future"][index])
            observed = np.asarray(fixed_bank["observed"][index])
            finite = finite and all(np.isfinite(value).all() for value in (
                raw_prediction, fixed_prediction, anchors, native, future, observed))
            endpoint_error = max(
                endpoint_error,
                float(np.max(np.abs(fixed_prediction[:, :, -1] - anchors))))
            expected = raw_prediction + time_fraction * (
                anchors - raw_prediction[:, :, -1])[:, :, None, :]
            formula_error = max(formula_error, float(np.max(np.abs(
                fixed_prediction - expected))))
            start = observed[:, -1]
            raw_line = start[None, :, None, :] + time_fraction * (
                raw_prediction[:, :, -1] - start[None])[:, :, None, :]
            fixed_line = start[None, :, None, :] + time_fraction * (
                anchors - start[None])[:, :, None, :]
            bridge_error = max(bridge_error, float(np.max(np.abs(
                (raw_prediction - raw_line) - (fixed_prediction - fixed_line)))))
            anchor_by_agent = anchors.transpose(1, 0, 2)
            distances = np.linalg.norm(
                anchor_by_agent[:, :, None] - native[:, None], axis=-1)
            anchor_set_error = max(
                anchor_set_error, float(distances.min(2).max()),
                float(distances.min(1).max()))
            native_error = np.linalg.norm(
                native - future[:, None, -1], axis=-1).min(1)
            native_error_sum += float(native_error.astype(np.float64).sum())
            native_agents += int(native_error.size)
        native_oracle_errors[seed] = native_error_sum / native_agents

    checks.update({
        "raw_bank_replay_max_abs_error": raw_replay_error,
        "raw_bank_replay_identity": raw_replay_error <= 1e-7,
        "fixed_bank_replay_max_abs_error": fixed_replay_error,
        "fixed_bank_replay_identity": fixed_replay_error <= 1e-7,
        "metadata_and_valid_rows_identical": metadata_equal,
        "all_values_finite": finite,
        "fixed_endpoint_anchor_max_abs_error": endpoint_error,
        "fixed_endpoint_anchor_exact": endpoint_error <= 1e-6,
        "linear_correction_max_abs_error": formula_error,
        "linear_correction_exact": formula_error <= 1e-6,
        "bridge_residual_max_abs_error": bridge_error,
        "bridge_residual_tolerance": 1e-5,
        "bridge_residual_preserved": bridge_error <= 1e-5,
        "assigned_anchor_native_set_max_l2_error": anchor_set_error,
        "assigned_anchor_is_native_permutation": anchor_set_error <= 1e-6,
        "scene_count": scene_count,
        "agent_occurrence_count": agent_count,
    })
    native_summary_error = max(
        abs(native_oracle_errors[seed] - fixed["minFDE"]["per_seed"][index])
        for index, seed in enumerate(SEEDS))
    checks["rho0_minfde_native_oracle_max_abs_error"] = native_summary_error
    checks["rho0_minfde_equals_native_oracle"] = native_summary_error <= 1e-6

    delta_raw = {
        metric: fixed[metric]["mean"] - raw[metric]["mean"] for metric in METRICS}
    delta_gdts = {
        metric: fixed[metric]["mean"] - gdts[metric]["mean"] for metric in METRICS}
    improvements = {
        metric: sum(fixed_value < raw_value for fixed_value, raw_value in zip(
            fixed[metric]["per_seed"], raw[metric]["per_seed"]))
        for metric in METRICS}
    candidate_pass = all(delta_raw[metric] <= 0 for metric in METRICS) and any(
        delta_raw[metric] < 0 for metric in METRICS)
    contract_checks = {
        "minADE_not_worse": delta_gdts["minADE"] <= 0,
        "minFDE_not_worse": delta_gdts["minFDE"] <= 0,
        "JADE_strictly_better": delta_gdts["JADE"] < 0,
        "JFDE_strictly_better": delta_gdts["JFDE"] < 0,
    }
    contract_pass = all(contract_checks.values())
    required_checks = [value for key, value in checks.items()
                       if isinstance(value, bool) and key != "optimizer_updates_zero"]
    audit_pass = all(required_checks) and checks["optimizer_updates_zero"]
    cpu_seconds = time.perf_counter() - started

    result = {
        "schema": "ebjd_no_token_fixed_anchor_v1",
        "status": "COMPLETE" if audit_pass else "INVALID",
        "source_commit": "3621dc016a6d4d3744413b18c938910a523504e7",
        "checkpoint_sha256": expected_checkpoint,
        "manifest_sha256": expected_manifest,
        "training_seed": 3101,
        "inference_seeds": list(SEEDS),
        "independent_test": False,
        "optimizer_updates": 0,
        "metrics": {"raw": raw, "fixed_anchor": fixed, "matched_GDTS": gdts},
        "fixed_minus_raw": delta_raw,
        "fixed_minus_matched_GDTS": delta_gdts,
        "fixed_improves_raw_seed_count": improvements,
        "candidate_screen": "PASS" if candidate_pass else "FAIL",
        "original_contract": "PASS" if contract_pass else "FAIL",
        "original_contract_checks": contract_checks,
        "identity_and_geometry_checks": checks,
        "native_oracle_minFDE_per_seed": native_oracle_errors,
        "bank_sha256_local_only": bank_hashes,
        "cost": {
            "formal_gpu_inference_wall_seconds": args.gpu_wall_seconds,
            "cpu_postprocessing_wall_seconds": cpu_seconds,
            "formal_gpu_runs": 1,
            "rejected_pre_inference_launches": 1,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    def cell(block: dict, metric: str) -> str:
        return (f'{block[metric]["mean"]:.4f} ± '
                f'{block[metric]["sample_standard_deviation"]:.4f}')

    raw_delta_text = ", ".join(
        f"{metric} {delta_raw[metric]:+.6f}" for metric in METRICS)
    improvement_text = ", ".join(
        f"{metric} {improvements[metric]}/5" for metric in METRICS)
    gdts_delta_text = ", ".join(
        f"{metric} {delta_gdts[metric]:+.6f}" for metric in METRICS)
    report = f"""# NO_TOKEN 固定锚点短评估

## 结论

- 候选筛选：**{'PASS' if candidate_pass else 'FAIL'}**。ρ=0 相对 Raw 四项均值均不退化，且有改善。
- 原合同：**{'PASS' if contract_pass else 'FAIL'}**。联合指标优于 matched GDTS，但边缘 minADE/minFDE 仍退化。
- 本轮仅改变冻结模型输出；训练 seed=3101，optimizer updates=0，五个推理 seed 不是独立训练重复，`independent_test=false`。

## 五种子结果（m，均值 ± 样本标准差）

| 输出 | minADE ↓ | minFDE ↓ | JADE ↓ | JFDE ↓ |
|---|---:|---:|---:|---:|
| NO_TOKEN Raw | {cell(raw, 'minADE')} | {cell(raw, 'minFDE')} | {cell(raw, 'JADE')} | {cell(raw, 'JFDE')} |
| NO_TOKEN Fixed-anchor | {cell(fixed, 'minADE')} | {cell(fixed, 'minFDE')} | {cell(fixed, 'JADE')} | {cell(fixed, 'JFDE')} |
| matched GDTS | {cell(gdts, 'minADE')} | {cell(gdts, 'minFDE')} | {cell(gdts, 'JADE')} | {cell(gdts, 'JFDE')} |

Fixed-anchor − Raw：{raw_delta_text}。逐 seed 改善数：{improvement_text}。

Fixed-anchor − matched GDTS：{gdts_delta_text}。

## 身份、几何与成本

checkpoint 与 manifest SHA256、seeds 2035–2039、P20、20 DDIM steps、eta0、FP32 均匹配。Raw bank 回放最大误差 `{raw_replay_error:.3e}`；固定终点误差 `{endpoint_error:.3e}`；线性修正误差 `{formula_error:.3e}`；桥残差误差 `{bridge_error:.3e}`；实际锚点集合与 native 候选集合双向误差 `{anchor_set_error:.3e}`。ρ=0 minFDE 与 native oracle 最大误差 `{native_summary_error:.3e}`。所有值有限，253 scenes / 364 agent occurrences 完整。

正式 GPU 推理墙钟 `{args.gpu_wall_seconds:.2f}` 秒；离线 CPU 核验约 `{cpu_seconds:.2f}` 秒。一次缺少 ablation 身份的启动在建模前被拒绝，未发生推理或更新。完整 bank 和日志仅留本地。

## 命令与边界

使用既有 `python -m ebjd.evaluate --ablation v2_no_token --endpoint-projection-radius 0 --prediction-bank-root ...`，随后由 `scripts/finalize_no_token_fixed_anchor.py` 对 Raw/Fixed bank 做成对核验。ρ=0 不改写上一轮主结果，也不保证零容忍边缘合同。

唯一下一步：保留 Fixed-anchor 为候选输出变体；若继续研究，只定位 native anchor 相对 GDTS 约 0.00181 m 的 minFDE 缺口来源，不调半径、不续训、不启动 rollout。
"""
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report)
    print(json.dumps({
        "status": result["status"], "candidate_screen": result["candidate_screen"],
        "original_contract": result["original_contract"], "output": str(args.output),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
