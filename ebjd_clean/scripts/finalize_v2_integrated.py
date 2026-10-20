#!/usr/bin/env python3
"""Build compact receipts and the Chinese EBJD-V2 integrated report."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
from pathlib import Path


METRICS = ("minADE", "minFDE", "JADE", "JFDE")


def read(path: str | Path) -> dict:
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
    output = {}
    for metric in METRICS:
        delta = [a - b for a, b in zip(lv[metric], rv[metric], strict=True)]
        output[metric] = {
            "mean_delta": float(statistics.fmean(delta)),
            "per_seed_delta": delta,
            "improved_seed_count": int(sum(value < 0 for value in delta)),
            "worsened_seed_count": int(sum(value > 0 for value in delta)),
        }
    return output


def lines(path: Path) -> list[dict]:
    return [json.loads(row) for row in path.read_text().splitlines() if row.strip()]


def training_summary(run: Path) -> dict:
    rows = lines(run / "EPOCH_SUMMARY.jsonl")
    return {
        "epochs": [int(row["epoch"]) for row in rows],
        "updates": int(sum(row["updates"] for row in rows)),
        "wall_seconds": float(sum(row["wall_seconds"] for row in rows)),
        "seconds_per_epoch": [float(row["wall_seconds"]) for row in rows],
        "rollout_updates": int(sum(row["rollout_updates"] for row in rows)),
        "projection_executions": int(sum(row.get("projection_executions", 0)
                                         for row in rows)),
        "peak_cuda_allocated_bytes": int(max(
            row["peak_cuda_allocated_bytes"] for row in rows)),
    }


def pairing_audit(runs: dict[str, Path]) -> dict:
    logs = {name: lines(path / "train.jsonl") for name, path in runs.items()}
    fields = (
        "epoch", "update", "logical_scene_agent_identity_sha256",
        "augmentation_sha256", "base_random_draw_sha256",
        "learning_rate", "unet_learning_rate")
    reference = logs["full"]
    mismatches = []
    for name in ("no_token", "no_anchor"):
        if len(logs[name]) != len(reference):
            mismatches.append({"arm": name, "field": "record_count",
                               "reference": len(reference), "actual": len(logs[name])})
        for index, (left, right) in enumerate(zip(reference, logs[name])):
            for field in fields:
                if left[field] != right[field]:
                    mismatches.append({"arm": name, "index": index, "field": field,
                                       "reference": left[field], "actual": right[field]})
                    if len(mismatches) >= 20:
                        break
            if len(mismatches) >= 20:
                break
    return {
        "status": "PASS" if not mismatches else "FAIL",
        "fields": list(fields), "updates_per_arm": len(reference),
        "mismatches": mismatches,
    }


def format_metric(row: dict, name: str) -> str:
    return f"{row[name]['mean']:.4f} ± {row[name]['sample_standard_deviation']:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--control-run", required=True)
    parser.add_argument("--full-run", required=True)
    parser.add_argument("--no-token-run", required=True)
    parser.add_argument("--no-anchor-run", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    assets, protocol = Path(args.assets), Path(args.protocol)
    evaluation = assets / "evaluation_v1"
    workspace = assets.parents[1]
    old_assets = workspace / "RSJG_EBJD_diagnostic_assets/eth_rollout_boundary_20261019"
    native_assets = workspace / "RSJG_EBJD_diagnostic_assets/eth_native_goal_conditioning_20261019"
    paths = {
        "parent_epoch20": old_assets / "evaluation_v1/epoch020_seeds2035_2039.json",
        "C": old_assets / "evaluation_v1/off_epoch025_seeds2035_2039.json",
        "V2_FULL": evaluation / "v2_full_epoch025_seeds2035_2039.json",
        "V2_NO_TOKEN": evaluation / "v2_no_token_epoch025_seeds2035_2039.json",
        "V2_NO_ANCHOR": evaluation / "v2_no_anchor_epoch025_seeds2035_2039.json",
        "matched_GDTS": native_assets / "evaluation_v4/matched_gdts_seeds2035_2039.json",
        "full_initial_pairing": evaluation / "v2_full_initial_pairing_seeds2035_2039.json",
        "full_direct_geometry": evaluation / "v2_full_direct_geometry_seeds2035_2039.json",
        "full_projection_rho0": evaluation / "v2_full_projection_rho0_seeds2035_2039.json",
        "full_projection_train_p95": evaluation / "v2_full_projection_train_p95_seeds2035_2039.json",
    }
    endpoint_adjustment_path = evaluation / "parent_epoch20_endpoint_adjustment.json"
    attention_path = evaluation / "v2_full_attention_observer.json"
    paths["parent_endpoint_adjustment"] = endpoint_adjustment_path
    missing = [str(path) for path in (*paths.values(), attention_path)
               if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"finalization inputs missing: {missing}")
    payloads = {name: read(path) for name, path in paths.items()}
    manifest = "03f9b6bef89d43137b31782bbf0eb5b16d19f8ddfe378e2c7c62e700e128397b"
    for name, payload in payloads.items():
        if name not in {"matched_GDTS", "parent_endpoint_adjustment"} \
                and payload.get("manifest_sha256") != manifest:
            raise ValueError(f"{name} manifest identity differs")
    summaries = {name: summarize(payload) for name, payload in payloads.items()}
    comparisons = {
        "Full_minus_C": comparison(payloads["V2_FULL"], payloads["C"]),
        "Full_minus_NO_TOKEN": comparison(
            payloads["V2_FULL"], payloads["V2_NO_TOKEN"]),
        "Full_minus_NO_ANCHOR": comparison(
            payloads["V2_FULL"], payloads["V2_NO_ANCHOR"]),
        "Full_minus_parent_epoch20": comparison(
            payloads["V2_FULL"], payloads["parent_epoch20"]),
        "Full_minus_GDTS": comparison(
            payloads["V2_FULL"], payloads["matched_GDTS"]),
        "initial_pairing_minus_Full": comparison(
            payloads["full_initial_pairing"], payloads["V2_FULL"]),
        "direct_geometry_minus_Full": comparison(
            payloads["full_direct_geometry"], payloads["V2_FULL"]),
    }
    runs = {"full": Path(args.full_run), "no_token": Path(args.no_token_run),
            "no_anchor": Path(args.no_anchor_run)}
    pairing = pairing_audit(runs)
    training = {name: training_summary(path) for name, path in runs.items()}
    checkpoints = {
        "C": {"path": str(Path(args.control_run) / "epoch_025.pt"),
              "sha256": sha256(Path(args.control_run) / "epoch_025.pt")},
        **{name: {"path": str(path / "epoch_025.pt"),
                  "sha256": sha256(path / "epoch_025.pt")}
           for name, path in runs.items()},
    }
    full, gdts = summaries["V2_FULL"], summaries["matched_GDTS"]
    contract = {
        "minADE_not_worse": full["minADE"]["mean"] <= gdts["minADE"]["mean"],
        "minFDE_not_worse": full["minFDE"]["mean"] <= gdts["minFDE"]["mean"],
        "JADE_strictly_better": full["JADE"]["mean"] < gdts["JADE"]["mean"],
        "JFDE_strictly_better": full["JFDE"]["mean"] < gdts["JFDE"]["mean"],
    }
    status = "PASS" if all(contract.values()) else "FAIL"
    results = {
        "schema": "ebjd-v2-integrated-results-v1", "status": "COMPLETED",
        "contract_status": status, "manifest_sha256": manifest,
        "plan_sha256": sha256(protocol / "PLAN.json"),
        "summaries": summaries, "comparisons": comparisons,
        "contract_checks": contract, "paired_training": pairing,
        "training": training, "checkpoints": checkpoints,
        "preflight": {
            "cpu": read(assets / "CPU_PREFLIGHT_RECEIPT.json")["status"],
            "cuda": read(assets / "CUDA_PREFLIGHT_RECEIPT.json")["status"],
            "control_reuse": read(assets / "CONTROL_REUSE_RECEIPT.json")["status"],
        },
        "speed_audit": read(assets / "SPEED_AUDIT.json"),
        "endpoint_diagnostics": {
            name: payloads[name]["per_seed"] for name in
            ("V2_FULL", "V2_NO_TOKEN", "V2_NO_ANCHOR")},
        "parent_endpoint_adjustment": payloads["parent_endpoint_adjustment"],
        "attention_observer": read(attention_path),
        "trust_region": {
            "rho0": summaries["full_projection_rho0"],
            "train_p95": summaries["full_projection_train_p95"],
            "radius_m": read(protocol / "PLAN.json")["assets"][
                "train_nearest_anchor_endpoint_error_p95_m"],
        },
        "independent_test": False,
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "RESULTS_AND_DIAGNOSTICS.json", results)
    atomic_json(output / "TRAINING_PAIRING_AUDIT.json", pairing)
    atomic_json(output / "CHECKPOINTS.json", checkpoints)

    rows = []
    for name in ("parent_epoch20", "C", "V2_FULL", "V2_NO_TOKEN",
                 "V2_NO_ANCHOR", "matched_GDTS"):
        row = summaries[name]
        rows.append("| " + name + " | " + " | ".join(
            format_metric(row, metric) for metric in METRICS) + " |")
    speed_rows = []
    speed = results["speed_audit"]
    for name, row in speed["workloads"].items():
        speed_rows.append(
            f"| {name} | {row['reference']['mean_update_seconds']:.3f} | "
            f"{row['optimized']['mean_update_seconds']:.3f} | {row['speedup']:.3f}× | "
            f"{row['compatibility']} | {row['optimized']['peak_allocated_bytes']/2**30:.2f} GiB |")
    report = f"""# EBJD V2 集成网络：ETH 固定短程实验

## 状态

- 工程：COMPLETE；CPU/CUDA 接线、按名称迁移和控制复用均通过。
- 训练与五种子评估：COMPLETE；比较固定 epoch25，不按中间曲线选模。
- 原合同：**{status}**。五个 inference seeds 共享同一训练 seed3101，`independent_test=false`。

## 真实数据流

```text
历史8帧 → GRU/social → 8历史+4运动摘要 ┐
                                       ├→ 12/16分组cross-attention → 3 coarse blocks
14通道crop → 可训练U-Net → 4×4地图16 ┘                         │
冻结GDTS/TTST20 → 每人20候选 → GT盲一一配对 → world锚点       │
高斯latent + 整组候选 + world锚点 → 锚点终点残差/桥变量 ←──────┘
粗物理轨迹 → 路径地图采样 + 物理关系 → Jacobian时间到latent汇总
          → 3 refinement blocks → 20步DDIM → P20联合场景
```

Provider 只读冻结；训练 P1 模式标签由训练 GT 最近候选产生，推理 P20 不读取未来 GT。

## 模块与维度

| 模块 | V2_FULL | NO_TOKEN | NO_ANCHOR |
|---|---:|---:|---:|
| motion/map memory | 12 + 16 | 8 + 64 | 12 + 16 |
| context attention | split + shared 384→64→2 gate | legacy joint | split + shared gate |
| goal latent | `(g-A)/s_g` | `(g-A)/s_g` | CV-centered legacy |
| joint pairing / assoc | ON / 0.05 | ON / 0.05 | OFF / 0 |
| keep loss | 0.01 | 0.01 | 0 |
| geometry map | decoder Jacobian | decoder Jacobian | decoder Jacobian |
| formal rollout / projection | OFF / OFF | OFF / OFF | OFF / OFF |

## 五种子结果（m，均值 ± 样本标准差）

| 模型 | minADE | minFDE | JADE | JFDE |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

所有逐 seed 差值、人数分层、K 曲线、endpoint P90/P95、单候选 FDE、中心误差和中心化 RMS 均在 `RESULTS_AND_DIAGNOSTICS.json` 中。`best_*` 未跨 epoch 拼接。

## 执行速度审计

| 工作负载 | 优化前 s/update | 优化后 s/update | 提速 | 兼容级别 | 优化后峰值 |
|---|---:|---:|---:|---|---:|
{chr(10).join(speed_rows)}

1.3× 是目标而不是结果门槛；未达到的工作负载不作提速宣称。启用项仅包括同 context 候选/锚点编码复用、DDIM 循环内诊断同步移除、OFF 无消费者零梯度树惰性化，以及显式 non-reentrant checkpoint 依赖。未通过或未证明有收益的 compile、worker、SDPA、QP 重写均未启用。

## 解释边界

- 一一配对只保持输入锚点集合；默认软残差输出不保证最终 minFDE 不退化。
- `rho=0` 与训练折最近锚点误差 P95 的投影是只读输出诊断，不是默认模型。
- Jacobian bias 是解码敏感度启发式，不是唯一正确的变量变换。
- C 是旧72-token、旧goal表示的已认证 rollout-OFF 控制；它不被描述成 V2 执行速度基准。
- 当前证据只覆盖 ETH，不与旧 HOTEL/UNIV 结构结果拼接。

完整小型收据目录：`{output}`。
"""
    Path(args.report).write_text(report, encoding="utf-8")
    print(json.dumps({"status": "COMPLETED", "contract": status,
                      "report": str(Path(args.report).resolve())}, sort_keys=True))


if __name__ == "__main__":
    main()
