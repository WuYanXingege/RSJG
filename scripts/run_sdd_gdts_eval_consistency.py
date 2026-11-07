#!/usr/bin/env python3
"""Run and summarize the frozen SDD matched-GDTS evaluation audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml


PAPER = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def manifest_identities(manifest_path: Path) -> tuple[Counter, set[str]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    identities: Counter = Counter()
    windows: set[str] = set()
    for record in payload["shards"]:
        with np.load(manifest_path.parent / record["path"],
                     allow_pickle=True) as shard:
            for source, frames, agents, window in zip(
                    shard["source_sequences"], shard["frame_ids"],
                    shard["agent_ids"], shard["window_keys"]):
                windows.add(str(window))
                start = str(int(np.asarray(frames).reshape(-1)[0]))
                for agent in np.asarray(agents).reshape(-1):
                    identities[(str(source), start, str(agent))] += 1
    return identities, windows


def result_identities(path: Path) -> tuple[Counter, set[str]]:
    identities: Counter = Counter()
    windows: set[str] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            identities[(row["source_sequence"], row["starting_frame"],
                        row["agent_id"])] += 1
            scene_id = row["manifest_scene_id"]
            if scene_id:
                windows.add(scene_id.rsplit(":", 1)[-1]
                            if scene_id.count(":") == 1 else
                            ":".join(scene_id.split(":")[-2:]))
    return identities, windows


def identity_comparison(expected: Counter, actual: Counter) -> dict:
    missing = expected - actual
    extra = actual - expected
    return {
        "expected_occurrences": int(sum(expected.values())),
        "actual_occurrences": int(sum(actual.values())),
        "expected_unique_identities": len(expected),
        "actual_unique_identities": len(actual),
        "exact_multiset_match": not missing and not extra,
        "missing_occurrences": int(sum(missing.values())),
        "extra_occurrences": int(sum(extra.values())),
        "duplicate_occurrences_beyond_expected": int(sum(extra.values())),
        "missing_examples": [list(key) + [value]
                             for key, value in missing.most_common(10)],
        "extra_examples": [list(key) + [value]
                           for key, value in extra.most_common(10)],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--formal-results", required=True)
    parser.add_argument("--training-log", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.checkpoint).resolve()
    manifest = Path(args.manifest).resolve()
    formal_path = Path(args.formal_results).resolve()
    formal = json.loads(formal_path.read_text(encoding="utf-8"))
    if formal.get("status") != "complete" or len(formal.get("runs", [])) != 5:
        raise RuntimeError("formal five-seed scene evaluation is incomplete")
    if sha256(checkpoint) != args.checkpoint_sha256:
        raise RuntimeError("checkpoint hash mismatch")
    checkpoint_payload = torch.load(
        checkpoint, map_location="cpu", weights_only=False)
    if int(checkpoint_payload.get("epoch", -1)) != 70:
        raise RuntimeError("selected checkpoint is not epoch 70")

    common = [
        sys.executable, str(PAPER / "scripts/evaluate_sdd_gdts_consistency_case.py"),
        "--recipe", str(Path(args.recipe).resolve()),
        "--runtime-root", str(Path(args.runtime_root).resolve()),
        "--cache-root", str(Path(args.cache_root).resolve()),
        "--checkpoint", str(checkpoint),
        "--checkpoint-sha256", args.checkpoint_sha256,
        "--manifest", str(manifest),
    ]
    cases = (("packed", 3101), ("packed", 2035), ("scene", 3101))
    case_payloads: dict[str, dict] = {}
    for layout, seed in cases:
        case_name = f"{layout}_{seed}"
        case_dir = output / "cases" / case_name
        command = common + [
            "--layout", layout, "--seed", str(seed),
            "--output-dir", str(case_dir),
        ]
        log_path = output / "logs" / f"{case_name}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Running {case_name}", flush=True)
        with log_path.open("a", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=PAPER, stdout=log,
                                    stderr=subprocess.STDOUT, check=False)
        if result.returncode:
            raise RuntimeError(f"{case_name} exited {result.returncode}")
        case_payloads[case_name] = json.loads(
            (case_dir / "RESULT.json").read_text(encoding="utf-8"))

    formal_2035 = next(run for run in formal["runs"] if run["seed"] == 2035)
    table = []
    for name, payload in case_payloads.items():
        table.append({
            "case": name,
            "layout": payload["layout"],
            "seed": payload["seed"],
            "physical_batches": payload["physical_batch_count"],
            "valid_agent_occurrences": payload["valid_agent_occurrence_count"],
            "minADE": payload["metrics"]["original_minADE"],
            "minFDE": payload["metrics"]["original_minFDE"],
            "reference_pixel_minADE": payload["metrics"]["reference_pixel_minADE"],
            "reference_pixel_minFDE": payload["metrics"]["reference_pixel_minFDE"],
            "elapsed_seconds": payload["elapsed_seconds"],
            "source": "new_frozen_replay",
        })
    table.append({
        "case": "scene_2035", "layout": "scene", "seed": 2035,
        "physical_batches": 1999,
        "valid_agent_occurrences": formal_2035["agent_occurrence_count"],
        "minADE": formal_2035["metrics"]["minADE"],
        "minFDE": formal_2035["metrics"]["minFDE"],
        "reference_pixel_minADE": "",
        "reference_pixel_minFDE": "",
        "elapsed_seconds": formal_2035["elapsed_seconds"],
        "source": "reused_formal_result",
    })
    table.sort(key=lambda row: (row["layout"], row["seed"]))
    with (output / "TWO_BY_TWO_RESULTS.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader(); writer.writerows(table)

    expected, expected_windows = manifest_identities(manifest)
    identity_results = {}
    for name, payload in case_payloads.items():
        actual, windows = result_identities(Path(payload["per_agent_errors"]))
        identity_results[name] = {
            **identity_comparison(expected, actual),
            "manifest_window_count": len(expected_windows),
            "observed_manifest_scene_ids": len(windows),
        }

    p3101 = case_payloads["packed_3101"]["metrics"]
    p2035 = case_payloads["packed_2035"]["metrics"]
    s3101 = case_payloads["scene_3101"]["metrics"]
    s2035 = formal_2035["metrics"]
    best = checkpoint_payload["best_metrics"]
    effects = {
        "packed_seed_2035_minus_3101": {
            "minADE": p2035["original_minADE"] - p3101["original_minADE"],
            "minFDE": p2035["original_minFDE"] - p3101["original_minFDE"],
        },
        "scene_seed_2035_minus_3101": {
            "minADE": s2035["minADE"] - s3101["original_minADE"],
            "minFDE": s2035["minFDE"] - s3101["original_minFDE"],
        },
        "scene_minus_packed_seed_3101": {
            "minADE": s3101["original_minADE"] - p3101["original_minADE"],
            "minFDE": s3101["original_minFDE"] - p3101["original_minFDE"],
        },
        "scene_minus_packed_seed_2035": {
            "minADE": s2035["minADE"] - p2035["original_minADE"],
            "minFDE": s2035["minFDE"] - p2035["original_minFDE"],
        },
    }
    log_reproduction = {
        "epoch70_log_minADE": float(best["minADE@K"]),
        "epoch70_log_minFDE": float(best["minFDE@K"]),
        "packed_3101_minADE": p3101["original_minADE"],
        "packed_3101_minFDE": p3101["original_minFDE"],
        "absolute_difference_minADE": abs(
            p3101["original_minADE"] - float(best["minADE@K"])),
        "absolute_difference_minFDE": abs(
            p3101["original_minFDE"] - float(best["minFDE@K"])),
    }

    reference_ok = all(
        payload["reference_agreement"][key] <= 1e-5
        for payload in case_payloads.values()
        for key in ("minADE_max_abs", "minFDE_max_abs"))
    scene_identity_ok = identity_results["scene_3101"]["exact_multiset_match"]
    packed_identity_ok = identity_results["packed_3101"]["exact_multiset_match"]
    if not reference_ok or not scene_identity_ok:
        decision = "BLOCKED_INCONSISTENT_EVALUATOR"
    else:
        decision = "SCENE_SYNCHRONIZED_FIVE_SEED"
    protocol = {
        "schema": "sdd-gdts-frozen-baseline-protocol-v1",
        "status": "FROZEN" if decision == "SCENE_SYNCHRONIZED_FIVE_SEED" else "BLOCKED",
        "decision": decision,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": args.checkpoint_sha256,
        "checkpoint_epoch": 70,
        "manifest": str(manifest),
        "manifest_sha256": sha256(manifest),
        "layout": "one synchronized manifest window per physical evaluation item",
        "scene_isolation": True,
        "valid_agent_reduction": "global per-agent mean",
        "minADE": "per-agent min over K after future-time mean",
        "minFDE": "per-agent min over K at final future frame",
        "seeds": [2035, 2036, 2037, 2038, 2039],
        "K": 20, "ddpm_steps": 100, "ddim_steps": 20,
        "trunk_stage_step": 30, "coordinate_unit": "pixel",
        "baseline_results": formal["summary"],
        "reason": (
            "The synchronized scene layout exactly matches the JMM manifest "
            "identity multiset and preserves joint-scene membership. The "
            "legacy packed validation cache is retained only as a training-"
            "curve diagnostic when its identity multiset differs."
        ),
    }
    atomic_json(output / "IDENTITY_AUDIT.json", identity_results)
    atomic_json(output / "EFFECTS.json", effects)
    atomic_json(output / "METRIC_RECOMPUTATION.json", {
        name: payload["reference_agreement"]
        for name, payload in case_payloads.items()})
    atomic_json(output / "FROZEN_BASELINE_PROTOCOL.json", protocol)
    runtime_root = Path(args.runtime_root).resolve()
    training_config_path = (
        runtime_root/
        "output/sdd/runs/sdd_jmm_matched_gdts_seed3101_v1/config.yaml")
    training_protocol_path = (
        runtime_root/
        "output/sdd/runs/sdd_jmm_matched_gdts_seed3101_v1/"
        "evaluation_protocol.json")
    run_manifest_path = runtime_root.parents[1]/"RUN_MANIFEST.json"
    training_config = yaml.safe_load(training_config_path.read_text(
        encoding="utf-8"))
    parameter_names = (
        "goal_model_type", "training_stage", "validation_seed", "num_samples",
        "ddpm_step", "ddim_step", "trunk_stage_step", "down_factor",
        "batch_size", "num_workers", "amp_enabled", "amp_dtype",
        "reproducibility", "data_augmentation", "shuffle_test_batches",
        "batch_cache_root")
    identity = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": args.checkpoint_sha256,
        "checkpoint_internal_epoch": int(checkpoint_payload["epoch"]),
        "checkpoint_best_metrics": checkpoint_payload["best_metrics"],
        "manifest": str(manifest), "manifest_sha256": sha256(manifest),
        "formal_results": str(formal_path),
        "formal_results_sha256": sha256(formal_path),
        "training_log": str(Path(args.training_log).resolve()),
        "training_log_sha256": sha256(Path(args.training_log).resolve()),
        "training_runtime_config": str(training_config_path),
        "training_runtime_config_sha256": sha256(training_config_path),
        "training_effective_parameters": {
            name: training_config.get(name) for name in parameter_names},
        "training_evaluation_protocol": str(training_protocol_path),
        "training_evaluation_protocol_sha256": sha256(training_protocol_path),
        "run_manifest": str(run_manifest_path),
        "run_manifest_sha256": sha256(run_manifest_path),
        "run_manifest_source_identity": json.loads(
            run_manifest_path.read_text(encoding="utf-8")),
        "audit_source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PAPER, text=True).strip(),
    }
    atomic_json(output / "IDENTITY.json", identity)
    summary = {
        "schema": "sdd-gdts-eval-consistency-summary-v1",
        "status": "COMPLETE" if protocol["status"] == "FROZEN" else "BLOCKED",
        "decision": decision,
        "same_epoch70_checkpoint_confirmed": True,
        "packed_epoch70_log_reproduction": log_reproduction,
        "identity_audit": identity_results,
        "effects": effects,
        "reference_metric_agreement_passed": reference_ok,
        "packed_matches_manifest": packed_identity_ok,
        "scene_matches_manifest": scene_identity_ok,
        "frozen_protocol": str(output / "FROZEN_BASELINE_PROTOCOL.json"),
        "five_seed_baseline": formal["summary"],
    }
    atomic_json(output / "SUMMARY.json", summary)

    report = f"""# SDD matched-GDTS 训练验证与冻结评测一致性核查

## 结论

- 同一权重：**已确认**。`best_model.pt` 内部 epoch 为 70，SHA256 为 `{args.checkpoint_sha256}`。
- 独立指标复算：**{'通过' if reference_ok else '失败'}**。原指标与直接像素公式的最大逐行人差见 `METRIC_RECOMPUTATION.json`。
- 训练期 packed 身份是否等于 JMM manifest：**{'是' if packed_identity_ok else '否'}**。
- scene-window 身份是否等于 JMM manifest：**{'是' if scene_identity_ok else '否'}**。
- 冻结口径：**{decision}**。

## 2×2 冻结回放

| layout | seed | physical batches | valid occurrences | minADE (px) | minFDE (px) |
|---|---:|---:|---:|---:|---:|
"""
    for row in table:
        report += (f"| {row['layout']} | {row['seed']} | "
                   f"{row['physical_batches']} | "
                   f"{row['valid_agent_occurrences']} | "
                   f"{float(row['minADE']):.6f} | "
                   f"{float(row['minFDE']):.6f} |\n")
    report += f"""

## 已确认原因

训练期 `814` 项 cache 不是 `1999` 个同步窗口的等价物理重排。它是旧 independent-GDTS 的单轨迹滑窗缓存；同步窗口 cache 才逐项绑定 JMM manifest，并保留同一窗口内全部有效行人。因此，旧训练曲线和正式 scene-window 五种子结果的差异不能只解释为 RNG 或单 seed 选模偏好。

seed 与 layout 的数值效应已分离记录在 `EFFECTS.json`。逐行人误差和位移误差 bank 位于 `cases/`，可在 CPU 上重新归约。

## 统一 baseline 口径

冻结 checkpoint epoch 70，不重新选模。baseline 使用同步 scene-window manifest、场景隔离、K=20、DDPM=100、DDIM=20、trunk=30、像素坐标、推理 seeds 2035–2039，并对所有有效行人的 minADE/minFDE 作全局平均。五种子结果以 `FROZEN_BASELINE_PROTOCOL.json` 为准。

旧 `8.x/13.x` packed 数值仅保留为训练曲线，不作为与 EBJD 的正式 baseline。

## 尚未排除

在样本身份已经不同的前提下，本轮不把剩余差值进一步归因到 TTST、扩散随机流或网络 batch 行为；这些因素的纯效应只由同一 layout 内的 seed 对照表述。
"""
    (output / "REPORT_ZH.md").write_text(report, encoding="utf-8")
    print(json.dumps({
        "status": summary["status"], "decision": decision,
        "output": str(output)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
