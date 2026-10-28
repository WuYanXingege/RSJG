#!/usr/bin/env python3
"""Resumable, fail-closed ZARA2 cross-fold reproduction pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml


SEEDS = [2035, 2036, 2037, 2038, 2039]


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
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


def merge(left: dict, right: dict) -> dict:
    output = deepcopy(left)
    for key, value in right.items():
        if isinstance(value, dict) and isinstance(output.get(key), dict):
            output[key] = merge(output[key], value)
        else:
            output[key] = deepcopy(value)
    return output


def checkpoint_epoch(path: Path) -> int | None:
    if not path.is_file():
        return None
    try:
        return int(torch.load(path, map_location="cpu", weights_only=False)["epoch"])
    except (KeyError, OSError, RuntimeError, TypeError, ValueError):
        return None


class Pipeline:
    def __init__(self, protocol_path: Path, source_commit: str) -> None:
        self.paper = Path(__file__).resolve().parents[1]
        self.protocol_path = protocol_path.resolve()
        self.protocol = yaml.safe_load(self.protocol_path.read_text(encoding="utf-8"))
        if self.protocol.get("fold") != "zara2":
            raise ValueError("this entrypoint is fixed to the ZARA2 fold")
        self.recipe_path = self.paper / self.protocol["stages"]["matched_gdts"]["recipe"]
        self.recipe = yaml.safe_load(self.recipe_path.read_text(encoding="utf-8"))
        self.root = Path(self.recipe["pipeline_root"]).resolve()
        self.python = str(Path(self.recipe["python"]).resolve())
        self.source_commit = source_commit
        self.status_path = self.root / "RUN_MANIFEST.json"
        self.runtime = self.root / "gdts/runtime"
        self.independent_cache = self.root / "gdts/independent_cache"
        self.scene_cache = self.root / "gdts/scene_cache"
        self.logs = self.root / "logs"
        self.maximum_gpu_seconds = float(self.protocol["gpu_wallclock_limit_seconds"])
        if self.status_path.is_file():
            self.state = json.loads(self.status_path.read_text(encoding="utf-8"))
        else:
            self.state = {
                "schema": "ebjd-zara2-crossfold-run-v1", "fold": "zara2",
                "status": "RUNNING", "stage": "initializing",
                "source_commit": source_commit, "gpu_seconds": 0.0,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "stage_receipts": {}, "child_pid": None,
            }
        if self.state.get("source_commit") != source_commit:
            raise ValueError("run manifest is bound to another source commit")

    def persist(self, **updates) -> None:
        self.state.update(updates)
        self.state["updated_at"] = datetime.now(timezone.utc).isoformat()
        atomic_json(self.status_path, self.state)

    def ensure_source(self) -> None:
        head = subprocess.run(
            ["git", "-C", str(self.paper), "rev-parse", "HEAD"], check=True,
            text=True, capture_output=True).stdout.strip()
        if head != self.source_commit:
            raise RuntimeError(f"source HEAD changed: {head} != {self.source_commit}")

    def run(self, stage: str, command: list[str], *, gpu: bool = True) -> None:
        self.ensure_source()
        if gpu and float(self.state.get("gpu_seconds", 0.0)) >= self.maximum_gpu_seconds:
            self.persist(status="PARTIAL_GPU_BUDGET_EXHAUSTED", stage=stage)
            raise RuntimeError("48-hour GPU budget is exhausted")
        log = self.logs / f"{stage}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        self.persist(stage=stage, status="RUNNING", command=command,
                     log_path=str(log), child_pid=None)
        with log.open("ab", buffering=0) as handle:
            process = subprocess.Popen(
                command, cwd=self.paper, stdout=handle, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                env={**os.environ, "PYTHONUNBUFFERED": "1",
                     "PYTHONDONTWRITEBYTECODE": "1"})
            self.persist(child_pid=process.pid)
            timeout = (self.maximum_gpu_seconds
                       - float(self.state.get("gpu_seconds", 0.0))) if gpu else None
            try:
                return_code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                elapsed = time.perf_counter() - started
                self.state["gpu_seconds"] = self.maximum_gpu_seconds
                self.persist(status="PARTIAL_GPU_BUDGET_EXHAUSTED", child_pid=None,
                             last_stage_seconds=elapsed,
                             failure=f"{stage} reached the 48-hour GPU limit")
                raise RuntimeError("48-hour GPU budget reached during a stage")
        elapsed = time.perf_counter() - started
        if gpu:
            self.state["gpu_seconds"] = float(self.state.get("gpu_seconds", 0.0)) + elapsed
        self.persist(child_pid=None, last_stage_seconds=elapsed,
                     last_stage_exit_code=return_code)
        if return_code:
            self.persist(status="FAILED", failure=f"{stage} exit={return_code}")
            raise RuntimeError(f"{stage} failed; see {log}")

    def receipt(self, name: str, **payload) -> None:
        receipts = dict(self.state.get("stage_receipts", {}))
        receipts[name] = payload
        self.persist(stage_receipts=receipts, stage=name, status="COMPLETE")

    def validate_data(self) -> None:
        data = {}
        for split, path in self.recipe["manifests"].items():
            manifest_path = Path(path)
            if not manifest_path.is_file():
                raise FileNotFoundError(manifest_path)
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            if payload["metadata"]["fold"] != "zara2":
                raise ValueError("manifest fold mismatch")
            data[split] = {
                "path": str(manifest_path), "sha256": sha256(manifest_path),
                **payload["statistics"],
            }
        held_out_average = data["validation"]["agents"] / data["validation"]["scenes"]
        lower, upper = self.protocol["held_out_average_agents_gate"]["accepted_interval"]
        if not lower <= held_out_average <= upper:
            raise ValueError(
                f"ZARA2 held-out average agents {held_out_average:.6f} is outside "
                f"the frozen [{lower}, {upper}] gate")
        if data["validation"]["sha256"] != data["test"]["sha256"]:
            validation = json.loads(Path(data["validation"]["path"]).read_text())
            test = json.loads(Path(data["test"]["path"]).read_text())
            if validation["window_keys"] != test["window_keys"]:
                raise ValueError("expected mirrored official validation/test windows")
        self.receipt("data", splits=data, held_out_average_agents=held_out_average,
                     no_agent_truncation=True)

    def legacy(self, stage: str, goal: Path | None = None) -> list[str]:
        command = [
            self.python, "-u", "-B", str(self.paper / "scripts/run_legacy_gdts_stage.py"),
            "--recipe", str(self.recipe_path), "--stage", stage,
            "--runtime-root", str(self.runtime), "--batch-cache-root",
            str(self.scene_cache if stage == "scene_cache" else self.independent_cache),
        ]
        if goal is not None:
            command += ["--goal-checkpoint", str(goal)]
        return command

    def prepare_gdts(self) -> tuple[Path, str, Path, str, Path]:
        self.runtime.mkdir(parents=True, exist_ok=True)
        data_link = self.runtime / "data"
        source_data = Path(self.recipe["source_data_root"]).resolve()
        if not data_link.exists() and not data_link.is_symlink():
            data_link.symlink_to(source_data, target_is_directory=True)
        goal_run = self.runtime / "output/zara2/runs" / self.recipe["goal_pretrain"]["run_name"]
        goal_source = goal_run / "saved_models/goal_pretrain_best_model.pt"
        goal_last = goal_run / "saved_models/goal_pretrain_last_model.pt"
        if checkpoint_epoch(goal_last) != 150 or not goal_source.is_file():
            self.run("goal_pretrain", self.legacy("goal"))
        goal_payload = torch.load(goal_source, map_location="cpu", weights_only=False)
        goal_last_payload = torch.load(goal_last, map_location="cpu", weights_only=False)
        if (goal_payload.get("test_set") != "zara2"
                or int(goal_last_payload.get("epoch", 0)) != 150
                or not goal_last_payload.get("full_runtime_resume_contract")):
            raise RuntimeError("ZARA2 goal pretraining contract failed")
        goal = self.root / "gdts/goal_unet_best.pt"
        if not goal.is_file():
            shutil.copy2(goal_source, goal)
        goal_hash = sha256(goal)
        baseline_run = self.runtime / "output/zara2/runs" / self.recipe["gdts_baseline"]["run_name"]
        baseline_source = baseline_run / "saved_models/best_model.pt"
        baseline_last = baseline_run / "saved_models/last_model.pt"
        if checkpoint_epoch(baseline_last) != 250 or not baseline_source.is_file():
            self.run("gdts_baseline", self.legacy("baseline", goal))
        baseline_payload = torch.load(baseline_source, map_location="cpu", weights_only=False)
        baseline_last_payload = torch.load(baseline_last, map_location="cpu", weights_only=False)
        initialization = baseline_payload.get("goal_pretrain_initialization") or {}
        if (int(baseline_last_payload.get("epoch", 0)) != 250
                or not baseline_last_payload.get("full_runtime_resume_contract")
                or initialization.get("sha256") != goal_hash):
            raise RuntimeError("ZARA2 matched GDTS completion/binding failed")
        baseline = self.root / "gdts/matched_gdts_best.pt"
        if not baseline.is_file():
            shutil.copy2(baseline_source, baseline)
        baseline_hash = sha256(baseline)
        scene_root = self.scene_cache / "eth5/zara2/data_batches_joint_v2"
        complete = all((scene_root / f"finished_{name}_batches.txt").is_file()
                       for name in ("train", "valid", "test"))
        if not complete:
            self.run("synchronized_scene_cache", self.legacy("scene_cache"), gpu=False)
        parity = self.root / "gdts/validation_manifest_parity.json"
        if not parity.is_file():
            self.run("validation_manifest_parity", [
                self.python, "-B", str(self.paper / "scripts/audit_export_parity.py"),
                "--manifest", self.recipe["manifests"]["validation"],
                "--legacy-batches", str(self.independent_cache / "eth5/zara2/data_batches/valid_batches"),
                "--grouped-batches", str(scene_root / "valid_batches"),
                "--homography", str(Path(self.recipe["source_data_root"]) / "eth5/zara2/H.txt"),
                "--semantic-map", str(Path(self.recipe["source_data_root"]) / "eth5/zara2/pred_mask.png"),
                "--source-scene", "zara2", "--output", str(parity),
            ], gpu=False)
        if not json.loads(parity.read_text())["passed"]:
            raise RuntimeError("ZARA2 manifest parity failed")
        matched = self.root / "evaluation/matched_gdts/results.json"
        if not matched.is_file() or json.loads(matched.read_text()).get("status") != "complete":
            self.run("matched_gdts_five_seed", [
                self.python, "-u", "-B", str(self.paper / "scripts/evaluate_matched_gdts.py"),
                "--recipe", str(self.recipe_path), "--runtime-root", str(self.runtime),
                "--scene-cache-root", str(self.scene_cache), "--checkpoint", str(baseline),
                "--checkpoint-sha256", baseline_hash, "--manifest",
                self.recipe["manifests"]["validation"], "--parity-receipt", str(parity),
                "--output-dir", str(matched.parent), "--split", "valid", "--seeds",
                *map(str, SEEDS),
            ])
        self.receipt("matched_gdts", goal=str(goal), goal_sha256=goal_hash,
                     baseline=str(baseline), baseline_sha256=baseline_hash,
                     result=str(matched), result_sha256=sha256(matched))
        return goal, goal_hash, baseline, baseline_hash, matched

    def write_model_config(self, goal: Path, goal_hash: str, baseline: Path,
                           baseline_hash: str, matched: Path) -> Path:
        architecture = yaml.safe_load((self.paper / "configs/architecture.yaml").read_text())
        training = yaml.safe_load((self.paper / "configs/training.yaml").read_text())
        result = json.loads(matched.read_text())
        seed2035 = next(row for row in result["runs"] if row["seed"] == 2035)["metrics"]
        fold = {
            "seed": 3101,
            "experiment": {"fold": "zara2", "source_commit": self.source_commit,
                           "implementation_base": "a9fb0c1"},
            "paths": {
                "train_manifest": self.recipe["manifests"]["train"],
                "validation_manifest": self.recipe["manifests"]["validation"],
                "test_manifest": self.recipe["manifests"]["test"],
                "output_root": str(self.root / "reference_base/output"),
                "accepted_fold_matched_gdts_unet": str(goal),
                "accepted_fold_matched_gdts_unet_sha256": goal_hash,
                "initialization_role": "trainable_goal_unet_only",
                "matched_gdts_validation_checkpoint": str(baseline),
                "matched_gdts_validation_checkpoint_sha256": baseline_hash,
                "matched_gdts_validation_result": str(matched),
                "matched_gdts_validation_result_sha256": sha256(matched),
            },
            "evaluation": {"matched_gdts_validation": {
                "status": "bound_to_manifest", "minADE": seed2035["minADE"],
                "minFDE": seed2035["minFDE"], "source": "local_ZARA2_matched_GDTS",
                "reference_limitation": "one_training_seed_and_mirrored_validation_test",
                "manifest_sha256": sha256(self.recipe["manifests"]["validation"]),
                "checkpoint_sha256": baseline_hash,
                "ebjd_initialization_sha256": goal_hash, "fold": "zara2", "unit": "m",
                "worlds": 20, "steps": 20, "evaluation_seed": 2035,
                "scene_count": result["scene_count"],
                "agent_occurrence_count": result["agent_occurrence_count"],
                "result_sha256": sha256(matched),
            }},
            "training": {"epochs": 250, "rollout_training_enabled": False,
                         "paired_marginal_actual_step_projection_enabled": False},
        }
        config = merge(merge(architecture, training), fold)
        target = self.root / "protocol/RESOLVED_ZARA2.json"
        atomic_json(target, config)
        return target

    def train_reference_base(self, config: Path) -> tuple[Path, str]:
        run_id = "zara2_reference_base_seed3101_e96_of250"
        run_dir = self.root / "reference_base/output/zara2/3101/none" / run_id
        last = run_dir / "last.pt"
        if checkpoint_epoch(last) != 96:
            command = [self.python, "-u", "-B", "-m", "ebjd.train", "--config",
                       str(config), "--ablation", "none", "--device", "cuda:0",
                       "--run-id", run_id, "--stop-after-epoch", "96"]
            if last.is_file():
                command += ["--resume", str(last)]
            self.run("reference_base_epoch96", command)
        if checkpoint_epoch(last) != 96:
            raise RuntimeError("reference base did not finish epoch 96")
        digest = sha256(last)
        self.receipt("reference_base", checkpoint=str(last), checkpoint_sha256=digest,
                     epoch=96, scheduler_total_epochs=250)
        return last, digest

    def export_native_banks(self, baseline: Path, baseline_hash: str) -> tuple[Path, dict[int, Path]]:
        train_manifest = Path(self.recipe["manifests"]["train"])
        validation_manifest = Path(self.recipe["manifests"]["validation"])
        train_bank = self.root / "proposal_banks/native_train/gdts_proposal_train_seed3101.npz"
        if not train_bank.is_file():
            self.run("native_proposal_train", [
                self.python, "-u", "-B", str(self.paper / "scripts/export_zara1_train_proposals.py"),
                "--recipe", str(self.recipe_path), "--checkpoint", str(baseline),
                "--checkpoint-sha256", baseline_hash, "--manifest", str(train_manifest),
                "--manifest-sha256", sha256(train_manifest), "--scene-cache-root",
                str(self.scene_cache), "--runtime-root", str(self.runtime), "--output-dir",
                str(train_bank.parent), "--seed", "3101",
            ])
        validation = {}
        for seed in SEEDS:
            output = self.root / f"proposal_banks/native_validation/seed_{seed}"
            bank = output / f"gdts_proposal_B_seed{seed}.npz"
            if not bank.is_file():
                self.run(f"native_proposal_validation_{seed}", [
                    self.python, "-u", "-B", str(self.paper / "scripts/export_zara1_endpoint_banks.py"),
                    "gdts", "--recipe", str(self.recipe_path), "--runtime-root", str(self.runtime),
                    "--scene-cache-root", str(self.scene_cache), "--checkpoint", str(baseline),
                    "--checkpoint-sha256", baseline_hash, "--manifest", str(validation_manifest),
                    "--manifest-sha256", sha256(validation_manifest), "--output-dir", str(output),
                    "--seed", str(seed),
                ])
            validation[seed] = bank
        return train_bank, validation

    def adapt_reference(self, config: Path, parent: Path, parent_hash: str,
                        baseline_hash: str, train_bank: Path,
                        validation_banks: dict[int, Path]) -> tuple[Path, str]:
        output = self.root / "reference_adaptation"
        result = output / "SEED2035_RESULT.json"
        pilot = self.root / "protocol/REFERENCE_ADAPTATION.yaml"
        atomic_json(self.root / "protocol/REFERENCE_ADAPTATION.json", {})
        pilot_payload = {
            "schema": "ebjd-crossfold-native-prior-adaptation-v1",
            "training_seed": 3101, "parent_checkpoint": str(parent),
            "parent_checkpoint_sha256": parent_hash,
            "gdts_checkpoint_sha256": baseline_hash,
            "train_manifest": self.recipe["manifests"]["train"],
            "validation_manifest": self.recipe["manifests"]["validation"],
            "validation_manifest_sha256": sha256(self.recipe["manifests"]["validation"]),
            "stage_a_epochs": 3, "arm_epochs": 3,
            "candidate_trust_weight": 0.01, "arms": ["Candidate"],
        }
        pilot.write_text(yaml.safe_dump(pilot_payload, sort_keys=False), encoding="utf-8")
        if not result.is_file():
            self.run("reference_stage_a_and_candidate", [
                self.python, "-u", "-B", str(self.paper / "scripts/run_zara1_native_prior_pilot.py"),
                "--model-config", str(config), "--pilot-config", str(pilot),
                "--train-bank", str(train_bank), "--validation-bank",
                str(validation_banks[2035]), "--output", str(output), "--device", "cuda:0",
            ])
        payload = json.loads(result.read_text())
        candidate = Path(payload["arms"]["Candidate"]["checkpoint"])
        digest = payload["arms"]["Candidate"]["checkpoint_sha256"]
        if sha256(candidate) != digest:
            raise RuntimeError("Candidate checkpoint hash mismatch")
        self.receipt("reference_candidate", checkpoint=str(candidate),
                     checkpoint_sha256=digest)
        return candidate, digest

    def build_mpjc(self, config: Path, candidate: Path, candidate_hash: str,
                   baseline_hash: str, train_bank: Path,
                   validation_banks: dict[int, Path]) -> tuple[Path, dict[int, Path]]:
        train_output = self.root / "proposal_banks/mpjc_train"
        train_h = train_output / "seed_3101/MPJC_ENDPOINT_BANK.npz"
        if not train_h.is_file():
            self.run("mpjc_train", [
                self.python, "-u", "-B", str(self.paper / "scripts/evaluate_zara1_mpjc_endpoints.py"),
                "--model-config", str(config), "--candidate-checkpoint", str(candidate),
                "--candidate-sha256", candidate_hash, "--bank-pattern", str(train_bank),
                "--gdts-sha256", baseline_hash, "--manifest-sha256",
                sha256(self.recipe["manifests"]["train"]), "--split", "train",
                "--output", str(train_output), "--seeds", "3101", "--device", "cuda:0",
            ])
        validation_output = self.root / "proposal_banks/mpjc_validation"
        validation_h = {seed: validation_output / f"seed_{seed}/MPJC_ENDPOINT_BANK.npz"
                        for seed in SEEDS}
        if not all(path.is_file() for path in validation_h.values()):
            pattern = str(validation_banks[2035]).replace("seed_2035", "seed_{seed}").replace(
                "seed2035", "seed{seed}")
            self.run("mpjc_validation_five_seed", [
                self.python, "-u", "-B", str(self.paper / "scripts/evaluate_zara1_mpjc_endpoints.py"),
                "--model-config", str(config), "--candidate-checkpoint", str(candidate),
                "--candidate-sha256", candidate_hash, "--bank-pattern", pattern,
                "--gdts-sha256", baseline_hash, "--manifest-sha256",
                sha256(self.recipe["manifests"]["validation"]), "--split", "validation",
                "--output", str(validation_output), "--seeds", *map(str, SEEDS),
                "--device", "cuda:0",
            ])
        return train_h, validation_h

    def train_gt_bridge(self, config: Path, candidate: Path, candidate_hash: str,
                        validation_h: dict[int, Path]) -> tuple[Path, str, Path]:
        output = self.root / "gt_bridge/e3"
        result = output / "TRAINING_RESULT.json"
        if not result.is_file():
            self.run("gt_conditioned_bridge_e3", [
                self.python, "-u", "-B", str(self.paper / "scripts/run_zara1_gt_conditioned_bridge.py"),
                "--model-config", str(config), "--candidate-checkpoint", str(candidate),
                "--candidate-sha256", candidate_hash, "--validation-bank",
                str(validation_h[2035]), "--validation-manifest-sha256",
                sha256(self.recipe["manifests"]["validation"]), "--output", str(output),
                "--device", "cuda:0",
            ])
        payload = json.loads(result.read_text())
        checkpoint = Path(payload["checkpoint"])
        digest = payload["checkpoint_sha256"]
        if sha256(checkpoint) != digest:
            raise RuntimeError("E3 checkpoint hash mismatch")
        five = self.root / "evaluation/e3_g_h/FIVE_SEED_RESULT.json"
        if not five.is_file():
            self.run("e3_g_h_five_seed", self.evaluation_command(
                config, checkpoint, digest, candidate_hash, validation_h,
                five.parent))
        return checkpoint, digest, five

    def evaluation_command(self, config: Path, checkpoint: Path, checkpoint_hash: str,
                           candidate_hash: str, banks: dict[int, Path], output: Path) -> list[str]:
        pattern = str(banks[2035]).replace("seed_2035", "seed_{seed}")
        return [
            self.python, "-u", "-B", str(self.paper / "scripts/evaluate_zara1_gt_conditioned_bridge.py"),
            "--model-config", str(config), "--checkpoint", str(checkpoint),
            "--checkpoint-sha256", checkpoint_hash, "--candidate-sha256", candidate_hash,
            "--manifest-sha256", sha256(self.recipe["manifests"]["validation"]),
            "--bank-2035", str(banks[2035]), "--bank-pattern", pattern,
            "--output", str(output), "--seeds", *map(str, SEEDS), "--device", "cuda:0",
        ]

    def bounded_long(self, config: Path, e3: Path, e3_hash: str, e3_five: Path,
                     candidate_hash: str, banks: dict[int, Path], matched: Path) -> tuple[Path, str]:
        e3_eval = json.loads(e3_five.read_text())
        gdts = json.loads(matched.read_text())
        initial = next(row["H"]["metrics"] for row in e3_eval["records"] if row["seed"] == 2035)
        baseline = next(row["metrics"] for row in gdts["runs"] if row["seed"] == 2035)
        remaining_budget = self.maximum_gpu_seconds - float(
            self.state.get("gpu_seconds", 0.0))
        if remaining_budget < 1200.0:
            raise RuntimeError("insufficient remaining budget for bounded continuation evaluation")
        plan = {
            "schema": "ebjd-crossfold-gt-bridge-h-bounded-long-v1",
            "fold": "zara2", "training_seed": 3101, "model_config": str(config),
            "parent": {"checkpoint": str(e3), "checkpoint_sha256": e3_hash,
                       "epoch": int(torch.load(e3, map_location="cpu", weights_only=False)["epoch"]),
                       "update": int(torch.load(e3, map_location="cpu", weights_only=False)["update_index"]),
                       "restore_optimizer": True, "restore_rng": True},
            "frozen_inference": {
                "candidate_reference_sha256": candidate_hash,
                "validation_manifest_sha256": sha256(self.recipe["manifests"]["validation"]),
                "h_bank_seed2035": str(banks[2035]),
                "h_bank_seed_pattern": str(banks[2035]).replace("seed_2035", "seed_{seed}"),
                "worlds": 20, "ddim_steps": 20, "selection_seed": 2035,
                "formal_seeds": SEEDS, "e3_five_seed_result": str(e3_five),
                "e3_five_seed_result_sha256": sha256(e3_five),
                "gdts_five_seed_result": str(matched),
                "gdts_five_seed_result_sha256": sha256(matched),
            },
            "model_contract": {"hidden_dim": 192, "heads": 4, "ff_dim": 768,
                               "fjab_blocks": 6},
            "continuation": {"maximum_additional_epochs": 30,
                             "main_initial_lr": 2e-5, "unet_initial_lr": 2e-6,
                             "cosine_final_fraction": 0.1,
                             "rollout_training_enabled": False,
                             "paired_actual_step_projection_enabled": False,
                             "gpu_wallclock_limit_seconds": remaining_budget,
                             "final_evaluation_reserve_seconds": 1200,
                             "expected_epoch_seconds": 900.0},
            "selection": {"metric": "JADE", "direction": "min", "require_eligible": True,
                          "patience_epochs": 8,
                          "patience_reset_minimum_improvement_m": 0.0005,
                          "preserve_actual_lowest_eligible": True,
                          "matched_gdts_seed2035": baseline,
                          "initial_parent_h_seed2035": initial},
        }
        plan_path = self.root / "protocol/BOUNDED_LONG.yaml"
        plan_path.write_text(yaml.safe_dump(plan, sort_keys=False), encoding="utf-8")
        output = self.root / "gt_bridge/bounded_long/training"
        result = output / "RESULT.json"
        if not result.is_file() or json.loads(result.read_text()).get("status") != "COMPLETE":
            self.run("bounded_long", [
                self.python, "-u", "-B", str(self.paper / "scripts/run_zara1_gt_bridge_bounded_long.py"),
                "--plan-config", str(plan_path), "--output", str(output),
                "--source-commit", self.source_commit, "--device", "cuda:0",
            ])
        payload = json.loads(result.read_text())
        selected = Path(payload["selection"]["best_checkpoint"])
        digest = payload["selection"]["best_checkpoint_sha256"]
        if sha256(selected) != digest:
            raise RuntimeError("selected bounded checkpoint hash mismatch")
        return selected, digest

    def final_report(self, final_eval: Path, matched: Path, selected: Path,
                     selected_hash: str) -> None:
        bridge = json.loads(final_eval.read_text())
        gdts = json.loads(matched.read_text())
        summary = {
            "schema": "ebjd-zara2-crossfold-final-v1", "status": "COMPLETE",
            "training_seed": 3101, "inference_seeds": SEEDS,
            "selected_checkpoint": str(selected),
            "selected_checkpoint_sha256": selected_hash,
            "G": bridge["summary"]["G"], "H": bridge["summary"]["H"],
            "GDTS": gdts["summary"],
            "H_minus_G_per_seed": bridge["summary"]["H_minus_G_per_seed"],
            "H_improvement_seed_count": bridge["summary"]["H_improvement_seed_count"],
            "gpu_seconds": self.state["gpu_seconds"],
        }
        h = summary["H"]["mean"]
        g = summary["G"]["mean"]
        b = {name: gdts["summary"][name]["mean"] for name in ("minADE", "minFDE", "JADE", "JFDE")}
        summary["H_contract_vs_GDTS"] = (
            h["minADE"] <= b["minADE"] and h["minFDE"] <= b["minFDE"]
            and h["JADE"] < b["JADE"] and h["JFDE"] < b["JFDE"])
        summary["H_joint_improves_over_G"] = h["JADE"] < g["JADE"] and h["JFDE"] < g["JFDE"]
        target = self.root / "reports/SUMMARY.json"
        atomic_json(target, summary)
        lines = [
            "# ZARA2 固定方法跨折复验", "", "- 状态：`COMPLETE`",
            f"- H 相对 matched GDTS 合同：`{'PASS' if summary['H_contract_vs_GDTS'] else 'FAIL'}`",
            f"- H 相对同桥 G 的联合改善：`{'YES' if summary['H_joint_improves_over_G'] else 'NO'}`",
            f"- GPU 累计秒：`{summary['gpu_seconds']:.3f}`", "",
            "| 路径 | minADE | minFDE | JADE | JFDE |", "|---|---:|---:|---:|---:|",
        ]
        for label, values in (("GDTS", b), ("Bridge+G", g), ("Bridge+H", h)):
            lines.append(f"| {label} | {values['minADE']:.6f} | {values['minFDE']:.6f} | {values['JADE']:.6f} | {values['JFDE']:.6f} |")
        lines += ["", "本结果为一个训练 seed 与五个随机推理 seed；完成后未自动启动其他 fold。", ""]
        (self.root / "reports/REPORT_ZH.md").write_text("\n".join(lines), encoding="utf-8")

    def execute(self) -> None:
        self.validate_data()
        goal, goal_hash, baseline, baseline_hash, matched = self.prepare_gdts()
        config = self.write_model_config(goal, goal_hash, baseline, baseline_hash, matched)
        parent, parent_hash = self.train_reference_base(config)
        train_bank, validation_banks = self.export_native_banks(baseline, baseline_hash)
        candidate, candidate_hash = self.adapt_reference(
            config, parent, parent_hash, baseline_hash, train_bank, validation_banks)
        _, h_banks = self.build_mpjc(
            config, candidate, candidate_hash, baseline_hash, train_bank, validation_banks)
        e3, e3_hash, e3_five = self.train_gt_bridge(config, candidate, candidate_hash, h_banks)
        selected, selected_hash = self.bounded_long(
            config, e3, e3_hash, e3_five, candidate_hash, h_banks, matched)
        final = self.root / "evaluation/final_bridge_g_h/FIVE_SEED_RESULT.json"
        if not final.is_file():
            self.run("final_bridge_g_h_five_seed", self.evaluation_command(
                config, selected, selected_hash, candidate_hash, h_banks, final.parent))
        self.final_report(final, matched, selected, selected_hash)
        self.persist(status="COMPLETE", stage="report", child_pid=None,
                     report=str(self.root / "reports/REPORT_ZH.md"),
                     next_fold_auto_launch=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    pipeline = Pipeline(Path(args.protocol), args.source_commit)
    try:
        pipeline.execute()
    except Exception as error:
        pipeline.persist(status="FAILED", child_pid=None,
                         failure=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
