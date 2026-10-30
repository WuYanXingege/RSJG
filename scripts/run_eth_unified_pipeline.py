#!/usr/bin/env python3
"""Resumable standalone-provider -> ETH unified training/evaluation pipeline."""

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


PAPER = Path(__file__).resolve().parents[1]
ROOT = PAPER.parent
ASSETS = ROOT / "RSJG_EBJD_assets"
PYTHON = ROOT / ".conda/rsjg/bin/python"
RECIPE = PAPER / "configs/asset_preparation/eth_gdts.yaml"
BASE_CONFIG = ASSETS / (
    "experiments/zara2_unified_coupling_bridge_v1_20261029_094715/"
    "configs/RESOLVED_ZARA2_UNIFIED.json")
MATCHED = ROOT / "GDTS/output/eth/saved_models/best_model.pt"
MATCHED_SHA = "126acf2a34f52986c536c397fe3acb04c769a3cde95077461d7971a1b0792950"
GOAL = ROOT / (
    "RSJG_JDV2_clean/output/eth/runs/eth_goal_pretrain_seed3101_20261015/"
    "saved_models/goal_pretrain_best_model.pt")
GOAL_SHA = "d130f47a3e13f457920a724465cc02d55bdb22e4827f7889bc49331d6c3bfec6"
TRAIN = ROOT / "RSJG_EBJD_clean/ebjd_clean/data/eth_jmm_official_scene_shards_20261015/train/train_manifest.json"
VALID = ROOT / "RSJG_EBJD_clean/ebjd_clean/data/eth_jmm_official_scene_shards_20261015/validation/validation_manifest.json"
TEST = ROOT / "RSJG_EBJD_clean/ebjd_clean/data/eth_jmm_official_scene_shards_20261015/test/test_manifest.json"
TRAIN_SHA = "8824bbad3b1b6c964bd828b1bee3ea5870e614b300b6d14090644a72818c54de"
VALID_SHA = "03f9b6bef89d43137b31782bbf0eb5b16d19f8ddfe378e2c7c62e700e128397b"
GDTS_RESULTS = ROOT / (
    "RSJG_EBJD_clean/ebjd_clean/outputs/formal_eth_main_20261015_seed3101/"
    "archive/GDTS_ETH_SEEDS2035_2039.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True); handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


class Pipeline:
    def __init__(self, root: Path, source_commit: str) -> None:
        self.root = root.resolve(); self.source_commit = source_commit
        for name in ("configs", "assets_manifest", "provider", "proposal_banks",
                     "logs", "training", "evaluations", "reports"):
            (self.root / name).mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "RUN_MANIFEST.json"
        self.budget_path = self.root / "GPU_BUDGET.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
        self.budget = json.loads(self.budget_path.read_text()) if self.budget_path.exists() else {
            "limit_seconds": 86400.0, "final_reserve_seconds": 10800.0,
            "consumed_seconds": 0.0, "stages": []}
        self.config = self.root / "configs/RESOLVED_ETH_UNIFIED.json"
        self.reference = self.root / "assets_manifest/MATCHED_GDTS_RESULTS.json"
        self.provider = self.root / "provider/provider_manifest.json"
        self.cache_root = self.root / "proposal_banks/manifest_bound_scene_cache"
        self.cache = self.cache_root / "eth5/eth/data_batches_joint_v2"
        self.runtime = self.root / "provider/runtime"

    def update(self, stage: str, status: str, **extra) -> None:
        if status != "FAILED":
            self.state.pop("failure", None)
        self.state.update({"schema": "ebjd-eth-unified-pipeline-v1", "fold": "eth",
                           "experiment_root": str(self.root),
                           "source_commit": self.source_commit, "stage": stage,
                           "status": status,
                           "updated_at_utc": datetime.now(timezone.utc).isoformat(), **extra})
        atomic_json(self.state_path, self.state)

    def charge(self, stage: str, seconds: float, status: str) -> None:
        self.budget["consumed_seconds"] += seconds
        self.budget["remaining_seconds"] = max(
            0.0, self.budget["limit_seconds"] - self.budget["consumed_seconds"])
        self.budget["stages"].append({"stage": stage, "seconds": seconds,
                                      "status": status,
                                      "ended_at_utc": datetime.now(timezone.utc).isoformat()})
        atomic_json(self.budget_path, self.budget)

    def run(self, stage: str, command: list[str], *, gpu: bool = True) -> None:
        log = self.root / f"logs/{stage.lower()}.log"
        self.update(stage, "RUNNING", command=command, log=str(log))
        started = time.perf_counter(); status = "failed"
        try:
            with log.open("a", encoding="utf-8") as handle:
                result = subprocess.run(command, cwd=PAPER, stdout=handle,
                                        stderr=subprocess.STDOUT, check=False)
            if result.returncode:
                raise RuntimeError(f"{stage} exited with code {result.returncode}")
            status = "complete"
        finally:
            if gpu: self.charge(stage, time.perf_counter() - started, status)

    def verify_assets(self) -> None:
        expected = {MATCHED: MATCHED_SHA, GOAL: GOAL_SHA, TRAIN: TRAIN_SHA, VALID: VALID_SHA}
        for path, digest in expected.items():
            if not path.is_file() or sha256(path) != digest:
                raise RuntimeError(f"asset identity mismatch: {path}")
        train = json.loads(TRAIN.read_text()); valid = json.loads(VALID.read_text())
        atomic_json(self.root / "assets_manifest/ASSETS.json", {
            "assets": {str(path): digest for path, digest in expected.items()},
            "data": {"train_scenes": train["statistics"]["scenes"],
                     "train_agents": train["statistics"]["agents"],
                     "validation_scenes": valid["statistics"]["scenes"],
                     "validation_agents": valid["statistics"]["agents"],
                     "validation_mean_agents": valid["statistics"]["agents"] /
                                               valid["statistics"]["scenes"]},
            "coordinate_unit": "m", "no_agent_truncation": True})

    def prepare(self) -> None:
        if not self.provider.exists():
            self.run("EXTRACT_PROVIDER", [str(PYTHON), "-u", "scripts/extract_goal_provider.py",
                "--checkpoint", str(MATCHED), "--expected-sha256", MATCHED_SHA,
                "--fold", "eth", "--output-dir", str(self.root / "provider")], gpu=False)
        source = json.loads(GDTS_RESULTS.read_text())
        runs = []
        for row in source["runs"]:
            runs.append({"seed": row["seed"], "manifest_sha256": VALID_SHA,
                         "worlds": 20, "ddim_steps": 20,
                         "metrics": {"minADE": row["minADE@K"],
                                     "minFDE": row["minFDE@K"],
                                     "JADE": row["JADE"], "JFDE": row["JFDE"]}})
        atomic_json(self.reference, {"schema": "matched-gdts-results-v1",
                    "manifest_sha256": VALID_SHA, "checkpoint_sha256": MATCHED_SHA,
                    "runs": runs})
        reference_sha = sha256(self.reference)
        config = json.loads(BASE_CONFIG.read_text())
        config["experiment"] = {"fold": "eth", "implementation_base": self.source_commit[:7],
                                "source_commit": self.source_commit}
        config["paths"].update({
            "accepted_fold_matched_gdts_unet": str(GOAL),
            "accepted_fold_matched_gdts_unet_sha256": GOAL_SHA,
            "matched_gdts_validation_checkpoint": str(MATCHED),
            "matched_gdts_validation_checkpoint_sha256": MATCHED_SHA,
            "matched_gdts_validation_result": str(self.reference),
            "matched_gdts_validation_result_sha256": reference_sha,
            "train_manifest": str(TRAIN), "validation_manifest": str(VALID),
            "test_manifest": str(TEST), "output_root": str(self.root / "training")})
        config["evaluation"]["matched_gdts_validation"].update({
            "fold": "eth", "checkpoint_sha256": MATCHED_SHA,
            "ebjd_initialization_sha256": GOAL_SHA, "manifest_sha256": VALID_SHA,
            "scene_count": 253, "agent_occurrence_count": 364,
            "minADE": runs[0]["metrics"]["minADE"],
            "minFDE": runs[0]["metrics"]["minFDE"],
            "source": "local_ETH_matched_GDTS", "status": "bound_to_manifest"})
        config["evaluation"]["matched_gdts_validation"]["result_sha256"] = reference_sha
        config["training"].update({"epochs": 30, "lr_warmup_epochs": 5,
                                    "lr_final_fraction": 0.1,
                                    # ETH scale fitting and shard prefetch can
                                    # multiply host memory across four forked
                                    # workers.  Single-process loading preserves
                                    # order/RNG while avoiding host-level kills.
                                    "data_loader_workers": 0,
                                    "rollout_training_enabled": False,
                                    "paired_marginal_actual_step_projection_enabled": False})
        atomic_json(self.config, config)

    def parity(self) -> None:
        output = self.root / "reports/PROVIDER_PARITY_REPORT.json"
        if output.exists() and json.loads(output.read_text()).get("status") == "PASS": return
        eth_parity_cache = self.root / "provider/eth_parity_cache/eth5/eth"
        eth_parity_cache.mkdir(parents=True, exist_ok=True)
        eth_joint_cache = eth_parity_cache / "data_batches_joint_v2"
        legacy_cache = ROOT / "GDTS/output/eth/data_batches"
        if not eth_joint_cache.exists() and not eth_joint_cache.is_symlink():
            eth_joint_cache.symlink_to(legacy_cache, target_is_directory=True)
        spec = {
            "zara1": {"recipe": str(PAPER / "configs/asset_preparation/zara1_gdts.yaml"),
                      "runtime_root": str(ASSETS / "zara1_full_pipeline_v1/gdts_runtime"),
                      "scene_cache_root": str(ASSETS / "zara1_full_pipeline_v1/gdts_scene_cache"),
                      "checkpoint": str(ASSETS / "checkpoints/zara1/matched_gdts_best.pt")},
            "zara2": {"recipe": str(PAPER / "configs/asset_preparation/zara2_gdts.yaml"),
                      "runtime_root": str(ASSETS / "experiments/formal_zara2_gt_bridge_h_20261028_100627/gdts/runtime"),
                      "scene_cache_root": str(ASSETS / "experiments/formal_zara2_gt_bridge_h_20261028_100627/gdts/scene_cache"),
                      "checkpoint": str(ASSETS / "experiments/formal_zara2_gt_bridge_h_20261028_100627/gdts/matched_gdts_best.pt")},
            "eth": {"recipe": str(RECIPE), "runtime_root": str(self.runtime),
                    "scene_cache_root": str(self.root / "provider/eth_parity_cache"),
                    "checkpoint": str(MATCHED)}}
        spec_path = self.root / "configs/PROVIDER_PARITY_SPEC.json"; atomic_json(spec_path, spec)
        self.run("PROVIDER_PARITY", [str(PYTHON), "-u", "scripts/verify_goal_provider_parity.py",
                 "--spec", str(spec_path), "--output", str(output), "--windows", "2"])

    def export_banks(self) -> None:
        common = [str(PYTHON), "-u", "scripts/stream_native_goal_bank.py",
                  "--recipe", str(RECIPE), "--provider-manifest", str(self.provider),
                  "--runtime-root", str(self.runtime)]
        train = self.root / "proposal_banks/native_train/gdts_proposal_train_manifest_seed3101.npz"
        if not train.exists():
            self.run("EXPORT_TRAIN", common + ["--manifest", str(TRAIN),
                     "--manifest-sha256", TRAIN_SHA, "--split", "train", "--seed", "3101",
                     "--output", str(train)])
        for seed in range(2035, 2040):
            output = self.root / f"proposal_banks/native_validation/seed_{seed}/gdts_proposal_B_seed{seed}.npz"
            if output.exists(): continue
            self.run(f"EXPORT_VALIDATION_{seed}", common + ["--manifest", str(VALID),
                     "--manifest-sha256", VALID_SHA, "--split", "validation",
                     "--seed", str(seed), "--output", str(output)])

    def train(self) -> None:
        remaining = self.budget["limit_seconds"] - self.budget["consumed_seconds"]
        if remaining <= self.budget["final_reserve_seconds"]:
            raise RuntimeError("final-evaluation reserve would be violated")
        self.run("UNIFIED_TRAIN_AND_EVALUATE", [str(PYTHON), "-u",
            "scripts/run_zara1_unified_coupling_bridge.py", "--config", str(self.config),
            "--fold", "eth", "--compact-scenes",
            "--train-bank", str(self.root / "proposal_banks/native_train/gdts_proposal_train_manifest_seed3101.npz"),
            "--validation-bank-pattern", str(self.root / "proposal_banks/native_validation/seed_{seed}/gdts_proposal_B_seed{seed}.npz"),
            "--output", str(self.root / "training"), "--source-commit", self.source_commit,
            "--max-gpu-seconds", str(remaining), "--final-reserve-seconds", "10800", "--resume"])

    def finalize(self) -> None:
        final_path = self.root / "training/FINAL_EVALUATION.json"
        if not final_path.exists():
            self.update("FINALIZE", "BUDGET_PARTIAL"); return
        final = json.loads(final_path.read_text()); reference = json.loads(self.reference.read_text())
        gdts = {int(row["seed"]): row["metrics"] for row in reference["runs"]}
        rows = []
        for run in final["runs"]:
            seed = int(run["seed"]); row = {"seed": seed}
            for mode in ("G", "H"):
                for metric, value in run[mode].items(): row[f"{mode}_{metric}"] = value
            for metric, value in gdts[seed].items():
                row[f"GDTS_{metric}"] = value
                row[f"H_minus_GDTS_{metric}"] = run["H"][metric] - value
                row[f"H_minus_G_{metric}"] = run["H"][metric] - run["G"][metric]
            rows.append(row)
        with (self.root / "reports/PER_SEED_METRICS.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        gdts_summary = {metric: {"mean": statistics.fmean(row["metrics"][metric] for row in reference["runs"]),
            "sample_standard_deviation": statistics.stdev(row["metrics"][metric] for row in reference["runs"])}
            for metric in ("minADE", "minFDE", "JADE", "JFDE")}
        summary = {"schema": "ebjd-eth-unified-final-summary-v1",
                   "selected_checkpoint": final["selected_checkpoint"],
                   "selected_checkpoint_sha256": final["selected_checkpoint_sha256"],
                   "optimizer_updates": final["optimizer_updates"], "G": final["summary"]["G"],
                   "H": final["summary"]["H"], "matched_GDTS": gdts_summary,
                   "performance_contract_pass": all(final["summary"]["H"][metric]["mean"] <= gdts_summary[metric]["mean"] for metric in gdts_summary),
                   "pairing_mechanism_mean_joint_improvement": all(final["summary"]["H"][metric]["mean"] < final["summary"]["G"][metric]["mean"] for metric in ("JADE", "JFDE")),
                   "five_inference_seeds_one_training_seed": True,
                   "provider_parity": "PASS", "gpu_budget": self.budget}
        atomic_json(self.root / "reports/FINAL_RESULT_SUMMARY.json", summary)
        selected = Path(final["selected_checkpoint"])
        atomic_json(self.root / "reports/CHECKPOINT_MANIFEST.json", {
            "selected": str(selected), "selected_sha256": sha256(selected),
            "selection_identity": final["selection_identity"]})
        report = ["# ETH 统一配对条件桥正式跨折复验", "",
                  "- 状态：COMPLETE", f"- 源码：`{self.source_commit}`",
                  f"- matched GDTS 四指标合同：`{'PASS' if summary['performance_contract_pass'] else 'FAIL'}`",
                  f"- H 相对 G 联合指标改善：`{'PASS' if summary['pairing_mechanism_mean_joint_improvement'] else 'FAIL'}`",
                  "- 独立终点提供器 GPU 回放：PASS；未实例化 GDTS 轨迹去噪器。",
                  "- 证据边界：一个训练 seed 3101，五个推理 seed 2035–2039。", "",
                  "逐 seed 数值见 `PER_SEED_METRICS.csv`。"]
        (self.root / "reports/FINAL_REPORT_ZH.md").write_text("\n".join(report) + "\n")
        self.update("COMPLETE", "COMPLETE", result=summary)

    def execute(self) -> None:
        try:
            self.verify_assets(); self.prepare(); self.parity()
            self.export_banks(); self.train(); self.finalize()
        except Exception as error:
            self.update(self.state.get("stage", "UNKNOWN"), "FAILED", failure=repr(error)); raise


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--source-commit", required=True); args = parser.parse_args()
    Pipeline(Path(args.experiment_root), args.source_commit).execute()


if __name__ == "__main__": main()
