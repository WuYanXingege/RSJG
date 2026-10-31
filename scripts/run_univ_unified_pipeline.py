#!/usr/bin/env python3
"""Resumable UNIV standalone-provider and unified bridge pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import run_hotel_unified_pipeline as hotel


PAPER = Path(__file__).resolve().parents[1]
ROOT = PAPER.parent
ASSETS = ROOT / "RSJG_EBJD_assets"
PYTHON = ROOT / ".conda/rsjg/bin/python"
RECIPE = PAPER / "configs/asset_preparation/univ_gdts.yaml"
MATCHED = ASSETS / "checkpoints/univ/matched_gdts_best.pt"
MATCHED_SHA = "ebfbae9de25463497c7bcc20981022f0638bc0349c7eeeb38bfc235dfaf61a3e"
GOAL = ASSETS / "checkpoints/univ/goal_unet_best.pt"
GOAL_SHA = "9fe5b2a7ae1e765d2706763b342d461c11ac0e736a2b597b6e18d63381652229"
TRAIN = ASSETS / "data/univ_official_train_scene_shards_20261010/train/train_manifest.json"
VALID = ASSETS / "data/univ_official/validation/validation_manifest.json"
TEST = ASSETS / "data/univ_official/test/test_manifest.json"
TRAIN_SHA = "6f6fca362ec1e591bf9a79c9b1a0204123947a0c7b754992f131432af40dfd5d"
VALID_SHA = "46c52540aac6f0313865f01f53d7c735d91cb8116fdbee39cb65f0cf545c7d8e"
TEST_SHA = "aabb35de82e936495fcd948a7ba0416d92d4fc93366151b99573a4d7683ee7e0"
GDTS_RESULTS = ROOT / (
    "RSJG_JDV2_clean/outputs/joint_dependency_v2/univ/joint_dependency_v2/"
    "audits/gdts_baseline_five_seed/results.json")
GDTS_RESULTS_SHA = "04f8d43f92b94ec09037bb267949f66092fa2dd2f42fe2b3e4388b33602c3477"
OLD_UNIV_RESULTS = ROOT / (
    "RSJG_EBJD_clean/ebjd_clean/outputs/formal_univ_main_20261013_seed3101/"
    "archive/EBJD_UNIV_SEEDS2035_2039.json")
OLD_UNIV_RESULTS_SHA = "dcecc0680740ca85e18eb46bfda9b24bffe621a3d33e30b68d7f39f11e04d217"

# The inherited implementation resolves these module globals at call time.
hotel.RECIPE = RECIPE
hotel.MATCHED = MATCHED
hotel.MATCHED_SHA = MATCHED_SHA
hotel.GOAL = GOAL
hotel.GOAL_SHA = GOAL_SHA
hotel.TRAIN = TRAIN
hotel.VALID = VALID
hotel.TEST = TEST
hotel.TRAIN_SHA = TRAIN_SHA
hotel.VALID_SHA = VALID_SHA
hotel.TEST_SHA = TEST_SHA
hotel.GDTS_RESULTS = GDTS_RESULTS
hotel.GDTS_RESULTS_SHA = GDTS_RESULTS_SHA


class UnivPipeline(hotel.HotelPipeline):
    def __init__(self, root: Path, source_commit: str) -> None:
        super().__init__(root, source_commit)
        if not self.budget_path.exists():
            self.budget = {
                "limit_seconds": 86400.0,
                "final_reserve_seconds": 10800.0,
                "consumed_seconds": 0.0,
                "remaining_seconds": 86400.0,
                "stages": [],
            }
        self.config = self.root / "configs/RESOLVED_UNIV_UNIFIED.json"
        self.reference = self.root / "assets_manifest/MATCHED_GDTS_RESULTS.json"
        self.provider = self.root / "provider/provider_manifest.json"
        self.runtime = self.root / "provider/runtime"

    def update(self, stage: str, status: str, **extra) -> None:
        if status != "FAILED":
            self.state.pop("failure", None)
        self.state.update({
            "schema": "ebjd-univ-unified-pipeline-v1", "fold": "univ",
            "experiment_root": str(self.root), "source_commit": self.source_commit,
            "stage": stage, "status": status,
            "updated_at_utc": hotel.base.datetime.now(
                hotel.base.timezone.utc).isoformat(), **extra,
        })
        hotel.base.atomic_json(self.state_path, self.state)

    def verify_assets(self) -> None:
        expected = {
            MATCHED: MATCHED_SHA, GOAL: GOAL_SHA, TRAIN: TRAIN_SHA,
            VALID: VALID_SHA, TEST: TEST_SHA, GDTS_RESULTS: GDTS_RESULTS_SHA,
        }
        for path, digest in expected.items():
            if not path.is_file() or hotel.base.sha256(path) != digest:
                raise RuntimeError(f"asset identity mismatch: {path}")
        reference = json.loads(GDTS_RESULTS.read_text())
        if reference.get("checkpoint", {}).get("sha256") != MATCHED_SHA:
            raise RuntimeError("matched-GDTS result/checkpoint identity mismatch")
        manifests = {name: json.loads(path.read_text()) for name, path in (
            ("train", TRAIN), ("validation", VALID), ("test", TEST))}
        hotel.base.atomic_json(self.root / "assets_manifest/ASSETS.json", {
            "assets": {str(path): digest for path, digest in expected.items()},
            "data": {name: {
                "scenes": value["statistics"]["scenes"],
                "agents": value["statistics"]["agents"],
                "mean_agents": value["statistics"]["agents"] /
                               value["statistics"]["scenes"],
                "max_agents": value["statistics"]["max_N"],
            } for name, value in manifests.items()},
            "coordinate_unit": "m", "no_agent_truncation": True,
        })
        if (not OLD_UNIV_RESULTS.is_file()
                or hotel.base.sha256(OLD_UNIV_RESULTS) != OLD_UNIV_RESULTS_SHA):
            raise RuntimeError("historical UNIV epoch73 result identity mismatch")
        manifest_path = self.root / "assets_manifest/ASSETS.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["assets"][str(OLD_UNIV_RESULTS)] = OLD_UNIV_RESULTS_SHA
        manifest["historical_only_not_loaded"] = {
            "epoch73_checkpoint_sha256":
                "f526506055f346513b7c57994f9935d40e624162d74e110104073312cfc4bbe9",
            "epoch81_checkpoint_sha256":
                "e9fd76be585fef0d9cf6ab82422e5a5775392030c59283054a5247a6f47eafdd",
        }
        hotel.base.atomic_json(manifest_path, manifest)

    def prepare(self) -> None:
        if not self.provider.exists():
            self.run("EXTRACT_PROVIDER", [
                str(PYTHON), "-u", "scripts/extract_goal_provider.py",
                "--checkpoint", str(MATCHED), "--expected-sha256", MATCHED_SHA,
                "--fold", "univ", "--output-dir", str(self.root / "provider")],
                gpu=False)
        source = json.loads(GDTS_RESULTS.read_text())
        runs = [{
            "seed": int(row["seed"]), "manifest_sha256": VALID_SHA,
            "worlds": 20, "ddim_steps": 20,
            "metrics": {
                "minADE": row["minADE@K"], "minFDE": row["minFDE@K"],
                "JADE": row["JADE"], "JFDE": row["JFDE"],
            },
        } for row in source["runs"]]
        if sorted(row["seed"] for row in runs) != list(range(2035, 2040)):
            raise RuntimeError("UNIV matched-GDTS five-seed reference is incomplete")
        hotel.base.atomic_json(self.reference, {
            "schema": "matched-gdts-results-v1", "manifest_sha256": VALID_SHA,
            "checkpoint_sha256": MATCHED_SHA, "runs": runs,
        })
        reference_sha = hotel.base.sha256(self.reference)
        config = json.loads(hotel.BASE_CONFIG.read_text())
        config["experiment"] = {
            "fold": "univ", "implementation_base": self.source_commit[:7],
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
            "fold": "univ", "checkpoint_sha256": MATCHED_SHA,
            "ebjd_initialization_sha256": GOAL_SHA,
            "manifest_sha256": VALID_SHA,
            "scene_count": valid["statistics"]["scenes"],
            "agent_occurrence_count": valid["statistics"]["agents"],
            "minADE": seed2035["metrics"]["minADE"],
            "minFDE": seed2035["metrics"]["minFDE"],
            "source": "local_UNIV_matched_GDTS", "status": "bound_to_manifest",
            "result_sha256": reference_sha,
        })
        config["training"].update({
            "epochs": 30, "lr_warmup_epochs": 5, "lr_final_fraction": 0.1,
            "rollout_training_enabled": False,
            "paired_marginal_actual_step_projection_enabled": False,
        })
        hotel.base.atomic_json(self.config, config)

    def parity(self) -> None:
        output = self.root / "reports/PROVIDER_PARITY_REPORT.json"
        if output.exists() and json.loads(output.read_text()).get("status") == "PASS":
            return
        self.runtime.mkdir(parents=True, exist_ok=True)
        cache_root = self.root / "provider/univ_parity_cache"
        cache = cache_root / "eth5/univ/data_batches"
        cache.parent.mkdir(parents=True, exist_ok=True)
        legacy_cache = ROOT / "GDTS/output/univ/data_batches"
        if not cache.exists() and not cache.is_symlink():
            cache.symlink_to(legacy_cache, target_is_directory=True)
        if cache.resolve() != legacy_cache.resolve():
            raise RuntimeError("UNIV parity cache target mismatch")
        spec_path = self.root / "configs/PROVIDER_PARITY_SPEC.json"
        hotel.base.atomic_json(spec_path, {"univ": {
            "recipe": str(RECIPE), "runtime_root": str(self.runtime),
            "scene_cache_root": str(cache_root), "checkpoint": str(MATCHED),
            "cache_goal_model_type": "independent",
        }})
        self.run("PROVIDER_PARITY", [
            str(PYTHON), "-u", "scripts/verify_goal_provider_parity.py",
            "--spec", str(spec_path), "--output", str(output), "--windows", "3"])

    def train(self) -> None:
        remaining = self.budget["limit_seconds"] - self.budget["consumed_seconds"]
        if remaining <= self.budget["final_reserve_seconds"]:
            raise RuntimeError("final-evaluation reserve would be violated")
        self.run("UNIFIED_TRAIN_AND_EVALUATE", [
            str(PYTHON), "-u", "scripts/run_zara1_unified_coupling_bridge.py",
            "--config", str(self.config), "--fold", "univ", "--compact-scenes",
            "--train-bank", str(self.root / (
                "proposal_banks/native_train/gdts_proposal_train_manifest_seed3101.npz")),
            "--validation-bank-pattern", str(self.root / (
                "proposal_banks/native_validation/seed_{seed}/"
                "gdts_proposal_B_seed{seed}.npz")),
            "--output", str(self.root / "training"),
            "--source-commit", self.source_commit,
            "--max-gpu-seconds", str(remaining),
            "--final-reserve-seconds", "10800", "--resume",
        ])

    def finalize(self) -> None:
        super().finalize()
        summary_path = self.root / "reports/FINAL_RESULT_SUMMARY.json"
        if not summary_path.exists():
            return
        summary = json.loads(summary_path.read_text())
        historical = json.loads(OLD_UNIV_RESULTS.read_text())
        summary["schema"] = "ebjd-univ-unified-final-summary-v1"
        summary["historical_UNIV_epoch73"] = {
            "checkpoint_sha256": historical["checkpoint_sha256"],
            "mean": historical["mean"],
            "sample_standard_deviation": historical["sample_standard_deviation"],
            "role": "historical_comparison_only_not_loaded",
        }
        summary["H_minus_historical_epoch73"] = {
            metric: summary["H"][metric]["mean"] - historical["mean"][metric]
            for metric in ("minADE", "minFDE", "JADE", "JFDE")
        }
        crossfold_roots = {
            "zara1": "zara1_unified_coupling_bridge_v1_20261028_214026",
            "zara2": "zara2_unified_coupling_bridge_v1_20261029_094715",
            "eth": "eth_unified_coupling_bridge_v1_20261030_104912",
            "hotel": "hotel_unified_coupling_bridge_v1_20261030_211601",
        }
        crossfold = {}
        for fold, directory in crossfold_roots.items():
            path = ASSETS / f"experiments/{directory}/reports/FINAL_RESULT_SUMMARY.json"
            if path.is_file():
                value = json.loads(path.read_text())
                crossfold[fold] = {
                    "source_sha256": hotel.base.sha256(path),
                    "H_joint_better_than_G": value.get(
                        "pairing_mechanism_mean_joint_improvement"),
                    "four_metric_contract_pass": value.get("performance_contract_pass"),
                }
        summary["completed_crossfold_direction"] = crossfold
        hotel.base.atomic_json(summary_path, summary)
        report = [
            "# UNIV 统一配对条件桥正式跨折复验", "",
            "- 状态：COMPLETE",
            f"- 源码：`{self.source_commit}`",
            f"- matched GDTS 四指标合同：`{'PASS' if summary['performance_contract_pass'] else 'FAIL'}`",
            f"- H 相对 G 联合指标改善：`{'PASS' if summary['pairing_mechanism_mean_joint_improvement'] else 'FAIL'}`",
            "- 原生路线 A/A 与独立提供器 A/C 回放：PASS。",
            "- 固定模型：192/4/768，6 FJAB，rollout/projection 全程 OFF。",
            "- 旧 UNIV epoch73/81 均未加载；epoch73 只作历史五种子比较。",
            "- validation/test 来自官方包镜像成员，不是独立测试成员。",
            "- 证据边界：一个训练 seed 3101，五个推理 seed 2035–2039。", "",
            "逐 seed 数值见 `PER_SEED_METRICS.csv`；完整数值与历史差值见 "
            "`FINAL_RESULT_SUMMARY.json`。缓存评测耗时不代表完整在线推理耗时。",
        ]
        (self.root / "reports/FINAL_REPORT_ZH.md").write_text(
            "\n".join(report) + "\n")
        self.update("COMPLETE", "COMPLETE", result=summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    UnivPipeline(Path(args.experiment_root), args.source_commit).execute()


if __name__ == "__main__":
    main()
