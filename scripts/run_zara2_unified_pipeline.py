#!/usr/bin/env python3
"""Resumable ZARA2 native-bank -> unified-training -> report pipeline."""

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
import time
from datetime import datetime, timezone
from pathlib import Path


OLD_ROOT = Path(
    "/media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/"
    "RSJG_EBJD_assets/experiments/formal_zara2_gt_bridge_h_20261028_100627")
PAPER = Path(__file__).resolve().parents[1]
PYTHON = Path(
    "/media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/.conda/rsjg/bin/python")
RECIPE = PAPER / "configs/asset_preparation/zara2_gdts.yaml"
GDTS_SHA = "bba3e538775e57fac45fd693ae6308c84fd92d6bf8d5451ab337005b292b2d80"
TRAIN_SHA = "0bd18ec06fda1974ec6cf84b5e19d898ae00cbb38b6bdcd7e5dfe1100c3d70d2"
VALID_SHA = "de47ab89a5421dad62eb1846511d72f22e104fe1fc7771117676dcda806c581f"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class Pipeline:
    def __init__(self, root: Path, source_commit: str) -> None:
        self.root = root.resolve()
        self.source_commit = source_commit
        self.root.mkdir(parents=True, exist_ok=True)
        for name in ("configs", "assets_manifest", "proposal_banks", "logs",
                     "training", "evaluations", "reports"):
            (self.root / name).mkdir(exist_ok=True)
        self.state_path = self.root / "RUN_MANIFEST.json"
        self.budget_path = self.root / "GPU_BUDGET.json"
        self.state = (json.loads(self.state_path.read_text())
                      if self.state_path.exists() else {})
        self.budget = (json.loads(self.budget_path.read_text())
                       if self.budget_path.exists() else {
                           "limit_seconds": 57600.0, "consumed_seconds": 0.0,
                           "stages": []})
        self.config = self.root / "configs/RESOLVED_ZARA2_UNIFIED.json"
        self.gdts = OLD_ROOT / "gdts/matched_gdts_best.pt"
        self.train_manifest = OLD_ROOT / "data_manifest/train/train_manifest.json"
        self.validation_manifest = OLD_ROOT / "data_manifest/validation/validation_manifest.json"

    def update(self, stage: str, status: str, **extra) -> None:
        self.state.update({
            "schema": "ebjd-zara2-unified-pipeline-v1", "fold": "zara2",
            "experiment_root": str(self.root), "source_commit": self.source_commit,
            "stage": stage, "status": status,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(), **extra})
        atomic_json(self.state_path, self.state)

    def charge(self, stage: str, seconds: float, status: str) -> None:
        self.budget["consumed_seconds"] += seconds
        self.budget["remaining_seconds"] = max(
            0.0, self.budget["limit_seconds"] - self.budget["consumed_seconds"])
        self.budget["stages"].append({
            "stage": stage, "seconds": seconds, "status": status,
            "ended_at_utc": datetime.now(timezone.utc).isoformat()})
        atomic_json(self.budget_path, self.budget)

    def run(self, stage: str, command: list[str], log: Path) -> None:
        if self.budget["consumed_seconds"] >= self.budget["limit_seconds"]:
            raise RuntimeError("16-hour GPU budget is exhausted")
        self.update(stage, "RUNNING", log=str(log), command=command)
        started = time.perf_counter()
        outcome = "failed"
        try:
            with log.open("a", encoding="utf-8") as handle:
                result = subprocess.run(
                    command, cwd=PAPER, stdout=handle, stderr=subprocess.STDOUT,
                    check=False)
            if result.returncode:
                raise RuntimeError(f"{stage} exited with code {result.returncode}")
            outcome = "complete"
        finally:
            self.charge(stage, time.perf_counter() - started, outcome)

    def verify_assets(self) -> None:
        expected = {
            self.gdts: GDTS_SHA, self.train_manifest: TRAIN_SHA,
            self.validation_manifest: VALID_SHA,
            OLD_ROOT / "gdts/goal_unet_best.pt":
                "50175952e2a5dca2984320b3d7751dcf851e09264b324fcf50a511f423a014db",
        }
        for path, digest in expected.items():
            if not path.is_file() or sha256(path) != digest:
                raise RuntimeError(f"asset identity mismatch: {path}")
        old_state = json.loads((OLD_ROOT / "RUN_MANIFEST.json").read_text())
        if old_state.get("status") != "PAUSED_FOR_UNIFIED_MODEL_VALIDATION":
            raise RuntimeError("old ZARA2 pipeline is not frozen in the required paused state")
        atomic_json(self.root / "assets_manifest/ASSETS.json", {
            "schema": "ebjd-zara2-unified-assets-v1",
            "assets": {str(path): digest for path, digest in expected.items()},
            "old_pipeline_status": old_state.get("status")})

    def prepare_config(self) -> None:
        source = OLD_ROOT / "protocol/RESOLVED_ZARA2.json"
        config = json.loads(source.read_text(encoding="utf-8"))
        config["experiment"]["source_commit"] = self.source_commit
        config["experiment"]["implementation_base"] = self.source_commit[:7]
        config["paths"]["output_root"] = str(self.root / "training")
        config["training"]["epochs"] = 30
        config["training"]["lr_final_fraction"] = 0.1
        config["training"]["rollout_training_enabled"] = False
        config["training"]["paired_marginal_actual_step_projection_enabled"] = False
        atomic_json(self.config, config)
        zara1 = Path(
            "/media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/"
            "RSJG_EBJD_assets/experiments/zara1_unified_coupling_bridge_v1_20261028_214026/"
            "configs/model_config.json")
        atomic_json(self.root / "configs/ZARA1_ZARA2_REQUIRED_DIFF.json", {
            "schema": "ebjd-unified-crossfold-required-diff-v1",
            "zara1_config_sha256": sha256(zara1),
            "zara2_config_sha256": sha256(self.config),
            "allowed_difference_categories": [
                "fold_identity", "data_and_manifest_paths", "same_fold_assets",
                "training_fold_fitted_representation_scales", "output_identity"],
            "method_contract_unchanged": True})

    def export_banks(self) -> None:
        train_dir = self.root / "proposal_banks/native_train"
        train_bank = train_dir / "gdts_proposal_train_seed3101.npz"
        common = [str(PYTHON)]
        if not train_bank.exists():
            self.run("EXPORT_TRAIN_BANK", common + [
                "scripts/export_zara1_train_proposals.py", "--recipe", str(RECIPE),
                "--checkpoint", str(self.gdts), "--checkpoint-sha256", GDTS_SHA,
                "--manifest", str(self.train_manifest), "--manifest-sha256", TRAIN_SHA,
                "--scene-cache-root", str(OLD_ROOT / "gdts/scene_cache"),
                "--runtime-root", str(OLD_ROOT / "gdts/runtime"),
                "--output-dir", str(train_dir), "--seed", "3101"],
                self.root / "logs/export_train_bank.log")
        for seed in range(2035, 2040):
            output = self.root / f"proposal_banks/native_validation/seed_{seed}"
            bank = output / f"gdts_proposal_B_seed{seed}.npz"
            if bank.exists():
                continue
            self.run(f"EXPORT_VALIDATION_BANK_{seed}", common + [
                "scripts/export_zara1_endpoint_banks.py", "gdts", "--recipe", str(RECIPE),
                "--runtime-root", str(OLD_ROOT / "gdts/runtime"),
                "--scene-cache-root", str(OLD_ROOT / "gdts/scene_cache"),
                "--checkpoint", str(self.gdts), "--checkpoint-sha256", GDTS_SHA,
                "--manifest", str(self.validation_manifest),
                "--manifest-sha256", VALID_SHA, "--output-dir", str(output),
                "--seed", str(seed)], self.root / f"logs/export_validation_{seed}.log")

    def train(self) -> None:
        remaining = self.budget["limit_seconds"] - self.budget["consumed_seconds"]
        if remaining <= 7200:
            raise RuntimeError("insufficient budget after bank generation; final reserve protected")
        command = [
            str(PYTHON), "-u", "scripts/run_zara1_unified_coupling_bridge.py",
            "--config", str(self.config), "--fold", "zara2", "--compact-scenes",
            "--train-bank", str(self.root / "proposal_banks/native_train/gdts_proposal_train_seed3101.npz"),
            "--validation-bank-pattern", str(
                self.root / "proposal_banks/native_validation/seed_{seed}/gdts_proposal_B_seed{seed}.npz"),
            "--output", str(self.root / "training"),
            "--source-commit", self.source_commit,
            "--max-gpu-seconds", str(remaining),
            "--final-reserve-seconds", "7200", "--resume"]
        self.run("UNIFIED_TRAIN_AND_EVALUATE", command, self.root / "logs/console.log")

    def finalize(self) -> None:
        final_path = self.root / "training/FINAL_EVALUATION.json"
        if not final_path.exists():
            status = json.loads((self.root / "training/STATUS.json").read_text())
            self.update("FINALIZE", "BUDGET_PARTIAL", training_status=status)
            return
        final = json.loads(final_path.read_text())
        matched = json.loads((OLD_ROOT / "evaluation/matched_gdts/results.json").read_text())
        gdts = {int(row["seed"]): row["metrics"] for row in matched["runs"]}
        rows = []
        for run in final["runs"]:
            seed = int(run["seed"])
            row = {"seed": seed}
            for mode in ("G", "H"):
                for metric, value in run[mode].items():
                    row[f"{mode}_{metric}"] = value
            for metric, value in gdts[seed].items():
                row[f"GDTS_{metric}"] = value
                row[f"H_minus_GDTS_{metric}"] = run["H"][metric] - value
                row[f"H_minus_G_{metric}"] = run["H"][metric] - run["G"][metric]
            rows.append(row)
        csv_path = self.root / "reports/PER_SEED_METRICS.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        gdts_summary = {metric: {
            "mean": statistics.fmean(row["metrics"][metric] for row in matched["runs"]),
            "sample_standard_deviation": statistics.stdev(
                row["metrics"][metric] for row in matched["runs"]),
        } for metric in ("minADE", "minFDE", "JADE", "JFDE")}
        result = {
            "schema": "ebjd-zara2-unified-final-summary-v1",
            "selected_checkpoint": final["selected_checkpoint"],
            "selected_checkpoint_sha256": final["selected_checkpoint_sha256"],
            "optimizer_updates": final["optimizer_updates"],
            "G": final["summary"]["G"], "H": final["summary"]["H"],
            "matched_GDTS": gdts_summary,
            "performance_contract_pass": all(
                final["summary"]["H"][metric]["mean"] <= gdts_summary[metric]["mean"]
                for metric in gdts_summary),
            "pairing_mechanism_mean_joint_improvement": all(
                final["summary"]["H"][metric]["mean"] < final["summary"]["G"][metric]["mean"]
                for metric in ("JADE", "JFDE")),
            "five_inference_seeds_one_training_seed": True,
        }
        atomic_json(self.root / "reports/FINAL_RESULT_SUMMARY.json", result)
        selected = Path(final["selected_checkpoint"])
        checkpoint_manifest = {
            "selected": str(selected), "selected_sha256": sha256(selected),
            "selection_identity": final["selection_identity"],
            "all_epoch_checkpoints": [{
                "epoch": epoch, "path": str(self.root / f"training/epoch_{epoch:03d}.pt"),
                "sha256": sha256(self.root / f"training/epoch_{epoch:03d}.pt")}
                for epoch in range(1, 31)]}
        atomic_json(self.root / "reports/CHECKPOINT_MANIFEST.json", checkpoint_manifest)
        report = [
            "# ZARA2 统一配对条件桥正式跨折复验", "",
            f"- 状态：COMPLETE", f"- 源码：`{self.source_commit}`",
            f"- 选定权重：`{final['selection_identity']}` / `{final['selected_checkpoint_sha256']}`",
            "- 证据边界：一个训练 seed（3101），五个推理 seed（2035–2039）。", "",
            "## 合同结论", "",
            f"- matched GDTS 四指标合同：`{'PASS' if result['performance_contract_pass'] else 'FAIL'}`",
            f"- H 相对 G 的 JADE/JFDE 均值改善：`{'PASS' if result['pairing_mechanism_mean_joint_improvement'] else 'FAIL'}`",
            "- G/H 使用相同终点候选集合；H 仅改变联合配对。", "",
            "详细逐 seed 数值见 `PER_SEED_METRICS.csv`，完整机器结果见 `FINAL_RESULT_SUMMARY.json`。", "",
        ]
        (self.root / "reports/FINAL_REPORT_ZH.md").write_text("\n".join(report), encoding="utf-8")
        self.update("COMPLETE", "COMPLETE", result=result)

    def execute(self) -> None:
        try:
            self.verify_assets(); self.prepare_config(); self.export_banks()
            self.train(); self.finalize()
        except Exception as error:
            self.update(self.state.get("stage", "UNKNOWN"), "FAILED", failure=repr(error))
            raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    Pipeline(Path(args.experiment_root), args.source_commit).execute()


if __name__ == "__main__":
    main()
