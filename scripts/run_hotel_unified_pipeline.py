#!/usr/bin/env python3
"""Resumable HOTEL standalone-provider and unified bridge pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import run_eth_unified_pipeline as base


PAPER = Path(__file__).resolve().parents[1]
ROOT = PAPER.parent
ASSETS = ROOT / "RSJG_EBJD_assets"
PYTHON = ROOT / ".conda/rsjg/bin/python"
RECIPE = PAPER / "configs/asset_preparation/hotel_gdts.yaml"
BASE_CONFIG = ASSETS / (
    "experiments/zara2_unified_coupling_bridge_v1_20261029_094715/"
    "configs/RESOLVED_ZARA2_UNIFIED.json")
MATCHED = ASSETS / "checkpoints/hotel/matched_gdts_best.pt"
MATCHED_SHA = "5c101c2474ebb1a3cb3ecf882f0e9db2fbb2489c741c58068b8183b0b663b97b"
GOAL = ASSETS / "checkpoints/hotel/goal_unet_epoch_096.pt"
GOAL_SHA = "f6a52cf228d733bd684aef043d843ba7d942e40495d5eec354ce1e4e3507ec13"
TRAIN = ASSETS / (
    "data/hotel_official_train_scene_shards_e7a1948/train/train_manifest.json")
VALID = ASSETS / "data/hotel_official/validation/validation_manifest.json"
TEST = ASSETS / "data/hotel_official/test/test_manifest.json"
TRAIN_SHA = "2a5f1909d928e5277d426ba04e3488940e34a5f71fad9da6331e5e673ec1966c"
VALID_SHA = "90fb42545ad043410f7c12b9ca91a1a0b218895b19252650e62f7fbefdb77d6d"
TEST_SHA = "84c95cb431caf7214d1b85e8e7dbfeaf48c9fe301b1cda7f3201c3e8cf199c78"
GDTS_RESULTS = ASSETS / "references/hotel_matched_gdts.json"
GDTS_RESULTS_SHA = "df9aa35cd26296558c864a2bcda792a8411bbc83a530376c45d3429a768eae6c"


class HotelPipeline(base.Pipeline):
    def __init__(self, root: Path, source_commit: str) -> None:
        super().__init__(root, source_commit)
        # HOTEL has an independent 16-hour ceiling.  The inherited ETH
        # orchestrator defaults to 24 hours, so replace that value only for a
        # brand-new HOTEL receipt; a resumed run keeps its accumulated ledger.
        if not self.budget_path.exists():
            self.budget = {
                "limit_seconds": 57600.0,
                "final_reserve_seconds": 10800.0,
                "consumed_seconds": 0.0,
                "remaining_seconds": 57600.0,
                "stages": [],
            }
        self.config = self.root / "configs/RESOLVED_HOTEL_UNIFIED.json"
        self.reference = self.root / "assets_manifest/MATCHED_GDTS_RESULTS.json"
        self.provider = self.root / "provider/provider_manifest.json"
        self.runtime = self.root / "provider/runtime"

    def update(self, stage: str, status: str, **extra) -> None:
        if status != "FAILED":
            self.state.pop("failure", None)
        self.state.update({
            "schema": "ebjd-hotel-unified-pipeline-v1", "fold": "hotel",
            "experiment_root": str(self.root), "source_commit": self.source_commit,
            "stage": stage, "status": status,
            "updated_at_utc": base.datetime.now(base.timezone.utc).isoformat(), **extra,
        })
        base.atomic_json(self.state_path, self.state)

    def verify_assets(self) -> None:
        expected = {
            MATCHED: MATCHED_SHA, GOAL: GOAL_SHA, TRAIN: TRAIN_SHA,
            VALID: VALID_SHA, TEST: TEST_SHA, GDTS_RESULTS: GDTS_RESULTS_SHA,
        }
        for path, digest in expected.items():
            if not path.is_file() or base.sha256(path) != digest:
                raise RuntimeError(f"asset identity mismatch: {path}")
        reference = json.loads(GDTS_RESULTS.read_text())
        if reference.get("parent_sha256") != MATCHED_SHA:
            raise RuntimeError("matched-GDTS result/checkpoint identity mismatch")
        manifests = {name: json.loads(path.read_text()) for name, path in (
            ("train", TRAIN), ("validation", VALID), ("test", TEST))}
        base.atomic_json(self.root / "assets_manifest/ASSETS.json", {
            "assets": {str(path): digest for path, digest in expected.items()},
            "data": {name: {
                "scenes": value["statistics"]["scenes"],
                "agents": value["statistics"]["agents"],
                "mean_agents": value["statistics"]["agents"] /
                               value["statistics"]["scenes"],
            } for name, value in manifests.items()},
            "coordinate_unit": "m", "no_agent_truncation": True,
        })

    def prepare(self) -> None:
        if not self.provider.exists():
            self.run("EXTRACT_PROVIDER", [
                str(PYTHON), "-u", "scripts/extract_goal_provider.py",
                "--checkpoint", str(MATCHED), "--expected-sha256", MATCHED_SHA,
                "--fold", "hotel", "--output-dir", str(self.root / "provider")],
                gpu=False)
        source = json.loads(GDTS_RESULTS.read_text())
        runs = []
        for row in source["runs"]:
            metrics = row["metrics"]
            runs.append({
                "seed": int(row["evaluation_seed"]), "manifest_sha256": VALID_SHA,
                "worlds": 20, "ddim_steps": 20,
                "metrics": {"minADE": metrics["minADE@K"],
                            "minFDE": metrics["minFDE@K"],
                            "JADE": metrics["JADE"], "JFDE": metrics["JFDE"]},
            })
        if sorted(row["seed"] for row in runs) != list(range(2035, 2040)):
            raise RuntimeError("HOTEL matched-GDTS five-seed reference is incomplete")
        base.atomic_json(self.reference, {
            "schema": "matched-gdts-results-v1", "manifest_sha256": VALID_SHA,
            "checkpoint_sha256": MATCHED_SHA, "runs": runs,
        })
        reference_sha = base.sha256(self.reference)
        config = json.loads(BASE_CONFIG.read_text())
        config["experiment"] = {
            "fold": "hotel", "implementation_base": self.source_commit[:7],
            "source_commit": self.source_commit,
        }
        config["paths"].update({
            "accepted_fold_matched_gdts_unet": str(GOAL),
            "accepted_fold_matched_gdts_unet_sha256": GOAL_SHA,
            "matched_gdts_validation_checkpoint": str(MATCHED),
            "matched_gdts_validation_checkpoint_sha256": MATCHED_SHA,
            "matched_gdts_validation_result": str(self.reference),
            "matched_gdts_validation_result_sha256": reference_sha,
            "train_manifest": str(TRAIN), "validation_manifest": str(VALID),
            "test_manifest": str(TEST), "output_root": str(self.root / "training"),
        })
        valid = json.loads(VALID.read_text())
        seed2035 = next(row for row in runs if row["seed"] == 2035)
        config["evaluation"]["matched_gdts_validation"].update({
            "fold": "hotel", "checkpoint_sha256": MATCHED_SHA,
            "ebjd_initialization_sha256": GOAL_SHA,
            "manifest_sha256": VALID_SHA,
            "scene_count": valid["statistics"]["scenes"],
            "agent_occurrence_count": valid["statistics"]["agents"],
            "minADE": seed2035["metrics"]["minADE"],
            "minFDE": seed2035["metrics"]["minFDE"],
            "source": "local_HOTEL_matched_GDTS", "status": "bound_to_manifest",
            "result_sha256": reference_sha,
        })
        config["training"].update({
            "epochs": 30, "lr_warmup_epochs": 5, "lr_final_fraction": 0.1,
            "rollout_training_enabled": False,
            "paired_marginal_actual_step_projection_enabled": False,
        })
        base.atomic_json(self.config, config)

    def parity(self) -> None:
        output = self.root / "reports/PROVIDER_PARITY_REPORT.json"
        if output.exists() and json.loads(output.read_text()).get("status") == "PASS":
            return
        # The legacy evaluator changes into an isolated runtime directory
        # before importing its modules.  Unlike the older fold assets, a new
        # HOTEL experiment has no pre-existing runtime tree.
        self.runtime.mkdir(parents=True, exist_ok=True)
        cache_root = self.root / "provider/hotel_parity_cache"
        cache = cache_root / "eth5/hotel/data_batches"
        cache.parent.mkdir(parents=True, exist_ok=True)
        legacy_cache = ROOT / "GDTS_official_297d508_HOTEL/output/hotel/data_batches"
        if not cache.exists() and not cache.is_symlink():
            cache.symlink_to(legacy_cache, target_is_directory=True)
        if cache.resolve() != legacy_cache.resolve():
            raise RuntimeError("HOTEL parity cache target mismatch")
        spec = {"hotel": {
            "recipe": str(RECIPE), "runtime_root": str(self.runtime),
            "scene_cache_root": str(cache_root),
            "checkpoint": str(MATCHED),
            "cache_goal_model_type": "independent",
        }}
        spec_path = self.root / "configs/PROVIDER_PARITY_SPEC.json"
        base.atomic_json(spec_path, spec)
        self.run("PROVIDER_PARITY", [
            str(PYTHON), "-u", "scripts/verify_goal_provider_parity.py",
            "--spec", str(spec_path), "--output", str(output), "--windows", "2"])

    def export_banks(self) -> None:
        common = [
            str(PYTHON), "-u", "scripts/stream_native_goal_bank.py",
            "--recipe", str(RECIPE), "--provider-manifest", str(self.provider),
            "--runtime-root", str(self.runtime),
        ]
        train = self.root / "proposal_banks/native_train/gdts_proposal_train_manifest_seed3101.npz"
        if not train.exists():
            self.run("EXPORT_TRAIN", common + [
                "--manifest", str(TRAIN), "--manifest-sha256", TRAIN_SHA,
                "--split", "train", "--seed", "3101", "--output", str(train)])
        for seed in range(2035, 2040):
            output = self.root / f"proposal_banks/native_validation/seed_{seed}/gdts_proposal_B_seed{seed}.npz"
            if output.exists():
                continue
            self.run(f"EXPORT_VALIDATION_{seed}", common + [
                "--manifest", str(VALID), "--manifest-sha256", VALID_SHA,
                "--split", "validation", "--seed", str(seed), "--output", str(output)])

    def train(self) -> None:
        remaining = self.budget["limit_seconds"] - self.budget["consumed_seconds"]
        if remaining <= self.budget["final_reserve_seconds"]:
            raise RuntimeError("final-evaluation reserve would be violated")
        self.run("UNIFIED_TRAIN_AND_EVALUATE", [
            str(PYTHON), "-u", "scripts/run_zara1_unified_coupling_bridge.py",
            "--config", str(self.config), "--fold", "hotel", "--compact-scenes",
            "--train-bank", str(self.root / "proposal_banks/native_train/gdts_proposal_train_manifest_seed3101.npz"),
            "--validation-bank-pattern", str(self.root / "proposal_banks/native_validation/seed_{seed}/gdts_proposal_B_seed{seed}.npz"),
            "--output", str(self.root / "training"), "--source-commit", self.source_commit,
            "--max-gpu-seconds", str(remaining), "--final-reserve-seconds", "10800",
            "--resume",
        ])

    def finalize(self) -> None:
        final_path = self.root / "training/FINAL_EVALUATION.json"
        if not final_path.exists():
            self.update("FINALIZE", "BUDGET_PARTIAL")
            return
        final = json.loads(final_path.read_text())
        reference = json.loads(self.reference.read_text())
        gdts = {int(row["seed"]): row["metrics"] for row in reference["runs"]}
        rows = []
        for run in final["runs"]:
            seed = int(run["seed"]); row = {"seed": seed}
            for mode in ("G", "H"):
                for metric, value in run[mode].items():
                    row[f"{mode}_{metric}"] = value
            for metric, value in gdts[seed].items():
                row[f"GDTS_{metric}"] = value
                row[f"H_minus_GDTS_{metric}"] = run["H"][metric] - value
                row[f"H_minus_G_{metric}"] = run["H"][metric] - run["G"][metric]
            rows.append(row)
        with (self.root / "reports/PER_SEED_METRICS.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        metrics = ("minADE", "minFDE", "JADE", "JFDE")
        gdts_summary = {metric: {
            "mean": statistics.fmean(row["metrics"][metric] for row in reference["runs"]),
            "sample_standard_deviation": statistics.stdev(
                row["metrics"][metric] for row in reference["runs"]),
        } for metric in metrics}
        summary = {
            "schema": "ebjd-hotel-unified-final-summary-v1",
            "selected_checkpoint": final["selected_checkpoint"],
            "selected_checkpoint_sha256": final["selected_checkpoint_sha256"],
            "optimizer_updates": final["optimizer_updates"],
            "G": final["summary"]["G"], "H": final["summary"]["H"],
            "matched_GDTS": gdts_summary,
            "performance_contract_pass": all(
                final["summary"]["H"][metric]["mean"] <= gdts_summary[metric]["mean"]
                for metric in metrics),
            "pairing_mechanism_mean_joint_improvement": all(
                final["summary"]["H"][metric]["mean"] <
                final["summary"]["G"][metric]["mean"] for metric in ("JADE", "JFDE")),
            "five_inference_seeds_one_training_seed": True,
            "provider_parity": "PASS", "gpu_budget": self.budget,
        }
        base.atomic_json(self.root / "reports/FINAL_RESULT_SUMMARY.json", summary)
        selected = Path(final["selected_checkpoint"])
        base.atomic_json(self.root / "reports/CHECKPOINT_MANIFEST.json", {
            "selected": str(selected), "selected_sha256": base.sha256(selected),
            "selection_identity": final["selection_identity"],
        })
        report = [
            "# HOTEL 统一配对条件桥正式复验", "", "- 状态：COMPLETE",
            f"- 源码：`{self.source_commit}`",
            f"- matched GDTS 四指标合同：`{'PASS' if summary['performance_contract_pass'] else 'FAIL'}`",
            f"- H 相对 G 联合指标改善：`{'PASS' if summary['pairing_mechanism_mean_joint_improvement'] else 'FAIL'}`",
            "- 独立终点提供器回放：PASS；未实例化 GDTS 轨迹去噪器。",
            "- 固定模型：192/4/768，6 FJAB，rollout/projection 全程 OFF。",
            "- 证据边界：一个训练 seed 3101，五个推理 seed 2035–2039。", "",
            "逐 seed 数值见 `PER_SEED_METRICS.csv`。",
        ]
        (self.root / "reports/FINAL_REPORT_ZH.md").write_text("\n".join(report) + "\n")
        self.update("COMPLETE", "COMPLETE", result=summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    HotelPipeline(Path(args.experiment_root), args.source_commit).execute()


if __name__ == "__main__":
    main()
