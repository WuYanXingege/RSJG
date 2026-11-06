#!/usr/bin/env python3
"""Render the small EBJD input-pipeline benchmark receipt for Git archival."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def mean(rows, key):
    return statistics.fmean(float(row[key]) for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    results_path, audit_path = Path(args.results), Path(args.audit)
    results = json.loads(results_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    rows = []
    all_correct = True
    all_speedups = []
    for case in results["cases"]:
        correctness = case["correctness"]
        all_correct &= bool(correctness["finite"])
        all_correct &= bool(correctness["all_required_gradients_present"])
        for repeat, (legacy, candidate) in enumerate(zip(
            [row for row in case["performance"]
             if row["mode"] == "legacy_cpu_rigid"],
            [row for row in case["performance"]
             if row["mode"] == "compact_uint8_gpu_rigid_v1"],
            strict=True,
        ), start=1):
            speedup = legacy["seconds_per_update"] / candidate["seconds_per_update"]
            all_speedups.append(speedup)
            rows.append({
                "fold_case": case["name"], "repeat": repeat,
                "legacy_s_per_update": legacy["seconds_per_update"],
                "candidate_s_per_update": candidate["seconds_per_update"],
                "speedup": speedup,
                "legacy_data_wait_s_per_update": legacy["data_wait_seconds_per_update"],
                "candidate_data_wait_s_per_update": candidate["data_wait_seconds_per_update"],
                "legacy_peak_allocated_mib": legacy["peak_allocated_mib"],
                "candidate_peak_allocated_mib": candidate["peak_allocated_mib"],
                "legacy_rss_peak_mib": legacy["rss_peak_mib"],
                "candidate_rss_peak_mib": candidate["rss_peak_mib"],
            })
    accepted = (
        all_correct and all(speedup >= 1.03 for speedup in all_speedups))
    conclusion = (
        "经适配后有稳定收益" if accepted else
        "输入侧候选未满足跨场景稳定采用门槛")

    with (output / "RESULTS.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (output / "BENCHMARK_RESULTS.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "DATA_PATH_AUDIT.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# EBJD 输入管线适用性核对",
        "",
        f"结论：**{conclusion}**。本结论只覆盖实测的 HOTEL/UNIV 正式训练输入与当前硬件。",
        "",
        "## 数据路径核对",
        "",
        "|环节|统一 EBJD 实际行为|与 matched-GDTS 的关系|",
        "|---|---|---|",
        "|缓存|每场景观测/未来坐标、身份及每人 6×256×256 的 uint8 局部语义裁剪|不是 GDTS 的物理 batch 密集热图缓存|",
        "|历史热图|MSCE forward 中按观测坐标在 GPU 生成 8 通道热图|已经具备同类优化，无需迁移 GDTS 热图重建|",
        "|未来监督热图|map loss 在 GPU 从未来坐标生成|不存在 CPU 密集监督热图增强|",
        "|增强|同场景统一旋转/反射；地图、轨迹和 G 同步|增强策略不同，不引入 GDTS 的仿射/透视操作|",
        "|proposal|冻结 bank 按 scene/agent 身份读取，随同一刚体变换同步更新|候选编号及 world 配对不变|",
        "|可学习路径|MSCE、coupling、GCJD/FJAB、CTF 均保留梯度|不进入 no_grad|",
        "",
        "已适配的执行变化只有两项：collate/传输阶段保留语义图 uint8；原有刚体增强转至主进程 GPU。",
        "",
        "## 真实 optimizer update 性能",
        "",
        "|输入|重复|旧路径 s/update|候选 s/update|提速|旧/新等待 s/update|旧/新显存 MiB|",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"|{row['fold_case']}|{row['repeat']}|"
            f"{row['legacy_s_per_update']:.6f}|"
            f"{row['candidate_s_per_update']:.6f}|"
            f"{row['speedup']:.3f}×|"
            f"{row['legacy_data_wait_s_per_update']:.6f}/"
            f"{row['candidate_data_wait_s_per_update']:.6f}|"
            f"{row['legacy_peak_allocated_mib']:.1f}/"
            f"{row['candidate_peak_allocated_mib']:.1f}|")
    lines.extend([
        "",
        f"两折两次配对平均提速：**{statistics.fmean(all_speedups):.3f}×**；"
        f"最小配对提速：**{min(all_speedups):.3f}×**。",
        "",
        "这些数字包含真实 DataLoader、增强、传输、完整 forward/loss/backward、"
        "梯度累积和 optimizer step；不包含 20-world/20-step 推理。",
        "",
        "## 数值与梯度验收",
        "",
    ])
    for case in results["cases"]:
        c = case["correctness"]
        lines.extend([
            f"### {case['name']}", "",
            f"- 参数有限：{c['finite']}；MSCE/U-Net、coupling、GCJD/CTF 梯度均存在："
            f"{c['all_required_gradients_present']}。",
            f"- legacy 自重复 loss 最大相对差："
            f"{c['legacy_repeat_loss']['max_relative']:.3e}。",
            f"- candidate 对 legacy loss 最大相对差："
            f"{c['candidate_vs_legacy_loss']['max_relative']:.3e}。",
            f"- candidate 对 legacy Goal U-Net 参数相对 L2："
            f"{c['candidate_vs_legacy_parameter_relative_l2']['goal_unet']:.3e}。",
            f"- candidate 对 legacy 全模型参数相对 L2："
            f"{c['candidate_vs_legacy_parameter_relative_l2']['trajectory_and_coupling']:.3e}。",
            "",
        ])
    lines.extend([
        "## 适用范围与边界", "",
        "- 适用于当前 6×256×256 每人语义裁剪、四场景 logical batch、四次梯度累积的统一 EBJD。",
        "- 已实测较高人数 HOTEL 训练折与相对较低人数 UNIV 训练折；不能外推到任意地图尺寸或其他硬件。",
        "- GDTS 的主要收益来自移除 CPU 密集轨迹热图处理；EBJD 本来已经在 GPU 生成历史/未来热图，因此不能套用 GDTS 的提速倍数。",
        "- 20 worlds/20 DDIM steps 仅属于正式推理，不在 rollout-OFF 训练更新计时中。",
        "",
        "## 固定结论", "",
        ("候选可作为后续统一 EBJD 的可选执行配置 `compact_uint8_gpu_rigid_v1`；"
         "方法、损失、增强分布和 checkpoint 键空间不变。" if accepted else
         "保留 `legacy_cpu_rigid` 为正式默认；输入侧不是值得继续扩展的主要提速方向。"),
        "",
        f"代码提交：`{results['source_commit']}`。",
    ])
    (output / "REPORT_ZH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
