#!/usr/bin/env python3
"""Persistent fail-closed runner that reuses the certified rollout-0.20 arm."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

from ebjd.config import config_hash, load_config
from ebjd.diagnostic_bank import atomic_json, sha256_file
from ebjd.train import validate_execution_migration_settings


SEEDS = (2035, 2036, 2037, 2038, 2039)
EXPECTED_MANIFEST_SHA256 = (
    "84c95cb431caf7214d1b85e8e7dbfeaf48c9fe301b1cda7f3201c3e8cf199c78")
EXPECTED_SOURCE_SHA256 = (
    "425988097793b639eb868fcc9ff28e41a3022c08598d13f396383da89268e1b1")
EXPECTED_CONTROL_EPOCH50_SHA256 = (
    "c44af3ed4456282033e2ffefbc1293e2ef5e55eff2a394f6e8b8a5d124ef9817")
EXPECTED_CONTROL_SOURCE_COMMIT = "69a4adf298b59b7724f98b1f638a17f6082c1006"


def utc_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


class Pipeline:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = root / "PIPELINE_STATE.json"
        self.state = {
            "schema": "ebjd-rollout-weight005-pilot-pipeline-v1",
            "status": "RUNNING", "started_at": utc_now(), "stages": [],
            "supervisor_pid": os.getpid(),
        }
        self.write_state()

    def write_state(self) -> None:
        self.state["updated_at"] = utc_now()
        atomic_json(self.state, self.state_path)

    def stage(self, name: str, status: str, **fields) -> None:
        self.state["current_stage"] = name
        self.state["stages"].append({
            "name": name, "status": status, "timestamp": utc_now(), **fields})
        self.write_state()

    def run(self, name: str, command: list[str], cwd: Path, env=None) -> None:
        log = self.root / "logs" / f"{name}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        self.stage(name, "STARTED", command=command, log=str(log))
        started = time.monotonic()
        with log.open("ab") as handle:
            process = subprocess.Popen(
                command, cwd=cwd, env=env, stdout=handle,
                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            self.state["active_child_pid"] = process.pid
            self.write_state()
            returncode = process.wait()
        self.state.pop("active_child_pid", None)
        if returncode:
            self.stage(name, "FAILED", returncode=returncode,
                       elapsed_seconds=time.monotonic() - started)
            raise RuntimeError(f"stage {name} failed with exit code {returncode}")
        self.stage(name, "COMPLETED", returncode=0,
                   elapsed_seconds=time.monotonic() - started)


def json_lines(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def git_output(repository: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repository, text=True).strip()


def validate_boundary(run_dir: Path, expected_weight: float, require_weight_logs: bool) -> dict:
    required = [
        run_dir / "RUN_CONTRACT.json", run_dir / "SHORT_RUN_SELECTION.json",
        run_dir / "train.jsonl", run_dir / "EPOCH_SUMMARY.jsonl",
        run_dir / "RESOLVED_CONFIG.json", run_dir / "RUN_IDENTITY.json",
        run_dir / "last.pt", run_dir / "best_joint_short.pt",
        run_dir / "best_marginal_short.pt", run_dir / "least_violation_short.pt",
        *(run_dir / f"epoch_{epoch:03d}.pt" for epoch in range(46, 51)),
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"run is missing required artifacts: {missing}")
    config = load_config(run_dir / "RESOLVED_CONFIG.json")
    if float(config["objectives"]["goal_v_weight"]) != 1.0:
        raise RuntimeError("formal rollout arm changed goal_v_weight")
    if float(config["training"]["rollout_weight_end"]) != expected_weight:
        raise RuntimeError("formal rollout arm has the wrong end weight")
    profile = validate_execution_migration_settings(config)
    contract = json.loads((run_dir / "RUN_CONTRACT.json").read_text())
    expected_contract = {
        "origin_epoch": 45, "absolute_stop_epoch": 50,
        "lr_schedule_total_epochs": 100, "origin_update_index": 11970,
        "expected_terminal_update_index": 13300, "updates_per_epoch": 266,
    }
    for name, expected in expected_contract.items():
        if int(contract[name]) != expected:
            raise RuntimeError(f"run contract {name}={contract[name]} != {expected}")
    records = json_lines(run_dir / "train.jsonl")
    summaries = json_lines(run_dir / "EPOCH_SUMMARY.jsonl")
    if len(records) != 1330:
        raise RuntimeError(f"expected 1330 updates, found {len(records)}")
    if [int(item["update"]) for item in records] != list(range(11971, 13301)):
        raise RuntimeError("optimizer updates are not exactly 11971..13300")
    if [int(item["epoch"]) for item in summaries] != list(range(46, 51)):
        raise RuntimeError("epoch summaries are not exactly 46..50")
    if any(int(item["updates"]) != 266 for item in summaries):
        raise RuntimeError("an epoch did not contain exactly 266 optimizer updates")
    if any(not all(isinstance(value, (int, float)) and math.isfinite(float(value))
                   for key, value in item.items()
                   if key not in {"logical_scene_id_sha256", "random_draw_sha256"})
           for item in records):
        raise RuntimeError("training record contains non-finite/non-numeric values")
    weight_deviation = coefficient_deviation = compensation_deviation = None
    if require_weight_logs:
        weight_deviation = max(abs(float(item["active_rollout_weight"]) - expected_weight)
                               for item in records)
        compensation_deviation = max(abs(
            float(item["rollout_frequency_compensation"]) - 4.0) for item in records)
        coefficient_deviation = max(abs(
            float(item["effective_rollout_coefficient"]) - 4.0 * expected_weight)
            for item in records)
        if max(weight_deviation, compensation_deviation, coefficient_deviation) > 1e-12:
            raise RuntimeError("logged rollout weight/compensation contract differs")
    checkpoint = torch.load(
        run_dir / "epoch_050.pt", map_location="cpu", weights_only=False, mmap=True)
    if int(checkpoint["epoch"]) != 50 or int(checkpoint["update_index"]) != 13300:
        raise RuntimeError("epoch50 checkpoint boundary is incorrect")
    if checkpoint.get("epoch_boundary_only") is not True:
        raise RuntimeError("epoch50 checkpoint is not an epoch boundary")
    del checkpoint
    checkpoint_hashes = {
        path.name: sha256_file(path) for path in required if path.suffix == ".pt"}
    return {
        "run_directory": str(run_dir), "rollout_weight_end": expected_weight,
        "goal_v_weight": 1.0, "updates": len(records),
        "first_update": records[0]["update"], "last_update": records[-1]["update"],
        "rollout_updates": sum(int(item["rollout_update"]) for item in records),
        "projection_active_updates": sum(
            int(float(item["projection_active"]) > 0) for item in records),
        "epoch_wall_seconds": [float(item["wall_seconds"]) for item in summaries],
        "peak_cuda_allocated_bytes": max(
            int(item["peak_cuda_allocated_bytes"]) for item in summaries),
        "checkpointed_rollout_group_fraction_by_epoch": [
            {
                "epoch": epoch,
                "mean": float(sum(
                    float(item["rollout_step_checkpointed_group_fraction"])
                    for item in records if int(item["epoch"]) == epoch
                    and int(item["rollout_update"])) / max(1, sum(
                    int(item["rollout_update"]) for item in records
                    if int(item["epoch"]) == epoch))),
            }
            for epoch in range(46, 51)],
        "execution_profile": profile,
        "config_hash": config_hash(config),
        "checkpoint_sha256": checkpoint_hashes,
        "epoch50_sha256": checkpoint_hashes["epoch_050.pt"],
        "active_rollout_weight_max_abs_deviation": weight_deviation,
        "rollout_compensation_max_abs_deviation": compensation_deviation,
        "effective_coefficient_max_abs_deviation": coefficient_deviation,
        "records": records,
    }


def audit_control_reuse(run_dir: Path, config_path: Path, bank_root: Path) -> dict:
    control = validate_boundary(run_dir, 0.20, require_weight_logs=False)
    if control["epoch50_sha256"] != EXPECTED_CONTROL_EPOCH50_SHA256:
        raise RuntimeError("reused control epoch50 checkpoint SHA256 mismatch")
    run_identity = json.loads((run_dir / "RUN_IDENTITY.json").read_text())
    migration = json.loads((run_dir / "MIGRATION_RECEIPT.json").read_text())
    if run_identity["source_commit"] != EXPECTED_CONTROL_SOURCE_COMMIT:
        raise RuntimeError("reused control execution commit mismatch")
    if migration["origin_checkpoint_sha256"] != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("reused control did not originate at the bound epoch45")
    if not all(migration["restoration_checks"].values()) or not all(
            migration["secondary_restore_checks"].values()):
        raise RuntimeError("reused control full-state restoration was not exact")
    if load_config(config_path) != load_config(run_dir / "RESOLVED_CONFIG.json"):
        raise RuntimeError("reused control config artifact differs from executed config")
    bank_receipts = []
    for seed in SEEDS:
        manifest = bank_root / f"seed_{seed}" / "manifest.json"
        payload = json.loads(manifest.read_text())
        if payload.get("status") != "COMPLETED":
            raise RuntimeError(f"reused control seed {seed} bank is incomplete")
        if payload["checkpoint_sha256"] != EXPECTED_CONTROL_EPOCH50_SHA256:
            raise RuntimeError(f"reused control seed {seed} bank checkpoint differs")
        if payload["source_manifest_sha256"] != EXPECTED_MANIFEST_SHA256:
            raise RuntimeError(f"reused control seed {seed} test manifest differs")
        if int(payload["scene_count"]) != 445 or int(payload["agent_occurrences"]) != 1197:
            raise RuntimeError(f"reused control seed {seed} bank count differs")
        bank_receipts.append({"seed": seed, "manifest": str(manifest),
                              "manifest_sha256": sha256_file(manifest),
                              "rng_state_chain_sha256": payload.get(
                                  "rng_state_chain_sha256")})
    return {
        "schema": "ebjd-rollout-weight005-control-reuse-v1",
        "status": "PREVALIDATED", "decision": "REUSE",
        "reason": (
            "bound epoch45 origin, exact full-state restore, certified shared execution "
            "profile, 1330 updates, epoch50 hash and all five formal banks verified"),
        "code_compatibility": {
            "control_execution_commit": EXPECTED_CONTROL_SOURCE_COMMIT,
            "new_code_changes": (
                "strict rollout-weight fork plus metadata logs; default rollout0.20 "
                "loss/RNG/optimizer semantics retained"),
            "final_pairing_deferred_until_candidate_completion": True,
        },
        "control": {key: value for key, value in control.items() if key != "records"},
        "bank_receipts": bank_receipts,
        "control_config": str(config_path),
        "control_config_file_sha256": sha256_file(config_path),
    }, control


def validate_pair(control: dict, candidate: dict) -> dict:
    left, right = control.pop("records"), candidate.pop("records")
    fields = (
        "epoch", "update", "rollout_update", "logical_scene_id_sha256",
        "random_draw_sha256", "learning_rate", "unet_learning_rate",
    )
    mismatches = []
    for index, (first, second) in enumerate(zip(left, right, strict=True)):
        changed = [name for name in fields if first[name] != second[name]]
        if changed:
            mismatches.append({"record_index": index, "fields": changed})
            if len(mismatches) >= 10:
                break
    return {"status": "PASSED" if not mismatches else "FAILED",
            "updates_compared": len(left), "compared_fields": list(fields),
            "mismatches": mismatches}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-config", required=True)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--pipeline-root", required=True)
    parser.add_argument("--archive-dir", required=True)
    parser.add_argument("--candidate-run-id", required=True)
    parser.add_argument("--control-run-dir", required=True)
    parser.add_argument("--control-config", required=True)
    parser.add_argument("--control-bank", required=True)
    parser.add_argument("--gdts-bank", required=True)
    parser.add_argument("--epoch45-bank", required=True)
    parser.add_argument("--preflight-receipt", required=True)
    parser.add_argument("--materialization-receipt", required=True)
    parser.add_argument("--push-final", action="store_true")
    args = parser.parse_args()

    repository = Path(__file__).resolve().parents[2]
    package = repository / "ebjd_clean"
    pipeline = Pipeline(Path(args.pipeline_root).resolve())
    try:
        head = git_output(repository, "rev-parse", "HEAD")
        if git_output(repository, "status", "--porcelain", "--", "ebjd_clean"):
            raise RuntimeError("formal pipeline requires a clean ebjd_clean tree")
        source = Path(args.source_checkpoint).resolve()
        if sha256_file(source) != EXPECTED_SOURCE_SHA256:
            raise RuntimeError("source epoch45 checkpoint SHA256 mismatch")
        candidate_config = Path(args.candidate_config).resolve()
        config = load_config(candidate_config)
        if config["experiment"]["source_commit"] != head:
            raise RuntimeError("candidate config is not bound to the execution commit")
        preflight = json.loads(Path(args.preflight_receipt).read_text())
        if preflight.get("status") != "PASSED" or preflight.get(
                "execution_commit") != head:
            raise RuntimeError("real CUDA preflight is absent, failed or bound elsewhere")
        pipeline.state.update({
            "execution_commit": head, "source_checkpoint": str(source),
            "source_checkpoint_sha256": EXPECTED_SOURCE_SHA256,
            "control_policy": "reuse_prevalidated_goal1_rollout020",
        })
        pipeline.write_state()

        control_reuse, control = audit_control_reuse(
            Path(args.control_run_dir).resolve(), Path(args.control_config).resolve(),
            Path(args.control_bank).resolve())
        control_reuse_path = pipeline.root / "CONTROL_REUSE_RECEIPT.json"
        atomic_json(control_reuse, control_reuse_path)
        pipeline.stage("control_reuse_prevalidation", "COMPLETED",
                       receipt=str(control_reuse_path), decision="REUSE")

        candidate_run_dir = (
            Path(config["paths"]["output_root"]) / config["experiment"]["fold"]
            / str(config["seed"]) / "none" / args.candidate_run_id)
        env = dict(os.environ)
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        train_command = [
            sys.executable, "-u", "-m", "ebjd.train", "--config",
            str(candidate_config), "--objective-fork-from", str(source),
            "--objective-fork-from-sha256", EXPECTED_SOURCE_SHA256,
            "--objective-fork-contract", "rollout_weight_end_005",
            "--max-epochs-this-run", "5", "--absolute-stop-epoch", "50",
            "--run-id", args.candidate_run_id,
        ]
        pipeline.run("candidate_train", train_command, package, env)
        candidate = validate_boundary(
            candidate_run_dir, 0.05, require_weight_logs=True)
        fork_receipt = json.loads(
            (candidate_run_dir / "OBJECTIVE_FORK_RECEIPT.json").read_text())
        if fork_receipt.get("contract") != "rollout_weight_end_005":
            raise RuntimeError("candidate objective-fork contract differs")
        if fork_receipt["origin_checkpoint_sha256"] != EXPECTED_SOURCE_SHA256:
            raise RuntimeError("candidate objective fork origin differs")
        if not all(fork_receipt["restoration_checks"].values()) or not all(
                fork_receipt["secondary_restore_checks"].values()):
            raise RuntimeError("candidate full-state restoration was not exact")
        pipeline.stage("candidate_validate", "COMPLETED", receipt={
            key: value for key, value in candidate.items() if key != "records"})

        pairing = validate_pair(control, candidate)
        if pairing["status"] != "PASSED":
            raise RuntimeError(f"reused control/candidate input pairing failed: {pairing}")
        control_reuse["status"] = "PASSED"
        control_reuse["code_compatibility"][
            "final_pairing_deferred_until_candidate_completion"] = False
        control_reuse["final_training_input_pairing"] = pairing
        atomic_json(control_reuse, control_reuse_path)
        pipeline.stage("training_pairing_and_control_reuse", "COMPLETED", receipt=pairing)

        training_receipt = {
            "schema": "ebjd-rollout-weight005-training-pairing-v1",
            "status": "PASSED", "execution_commit": head,
            "source_checkpoint_sha256": EXPECTED_SOURCE_SHA256,
            "sole_semantic_change": {"training.rollout_weight_end": {
                "control": 0.20, "candidate": 0.05}},
            "control": control, "candidate": candidate, "pairing": pairing,
            "candidate_config": str(candidate_config),
            "candidate_config_file_sha256": sha256_file(candidate_config),
        }
        training_path = pipeline.root / "TRAINING_AND_PAIRING_RECEIPT.json"
        atomic_json(training_receipt, training_path)

        banks = pipeline.root / "banks"
        export_command = [
            sys.executable, "-u", str(package / "scripts" /
                                       "export_epoch45_diagnostic_bank.py"),
            "--config", str(candidate_config),
            "--checkpoint", str(candidate_run_dir / "epoch_050.pt"),
            "--output-root", str(banks),
            "--expected-checkpoint-sha256", candidate["epoch50_sha256"],
            "--expected-config-sha256", sha256_file(candidate_config),
            "--expected-manifest-sha256", EXPECTED_MANIFEST_SHA256,
            "--bank-name", "candidate", "--model-label",
            "EBJD_rollout005_candidate_epoch50", "--seeds",
            *(str(seed) for seed in SEEDS), "--device", "cuda", "--resume",
        ]
        pipeline.run("candidate_bank_export", export_command, package, env)

        result_dir = pipeline.root / "analysis"
        archive = Path(args.archive_dir).resolve()
        analysis_command = [
            sys.executable, "-u", str(package / "scripts" /
                                       "analyze_rollout_weight005_pilot_banks.py"),
            "--gdts-bank", str(Path(args.gdts_bank).resolve()),
            "--epoch45-bank", str(Path(args.epoch45_bank).resolve()),
            "--control-bank", str(Path(args.control_bank).resolve()),
            "--candidate-bank", str(banks / "candidate"),
            "--control-reuse-receipt", str(control_reuse_path),
            "--training-receipt", str(training_path),
            "--preflight-receipt", str(Path(args.preflight_receipt).resolve()),
            "--output-dir", str(result_dir), "--archive-dir", str(archive),
        ]
        pipeline.run("cpu_final_analysis", analysis_command, package, env)

        archive.mkdir(parents=True, exist_ok=True)
        copies = {
            control_reuse_path: "CONTROL_REUSE_RECEIPT.json",
            training_path: "TRAINING_AND_PAIRING_RECEIPT.json",
            Path(args.preflight_receipt).resolve(): "ROLLOUT_WEIGHT005_REAL_CUDA_PREFLIGHT.json",
            Path(args.materialization_receipt).resolve(): "MATERIALIZATION_RECEIPT.json",
        }
        for source_path, name in copies.items():
            shutil.copy2(source_path, archive / name)
        pipeline.state["status"] = "COMPLETED"
        pipeline.state["completed_at"] = utc_now()
        pipeline.state["final_result_receipt"] = str(
            result_dir / "FINAL_RESULT_RECEIPT.json")
        pipeline.write_state()
        shutil.copy2(pipeline.state_path, archive / "PIPELINE_STATE.json")

        if args.push_final:
            relative = archive.relative_to(repository)
            subprocess.run(["git", "add", "--", str(relative)],
                           cwd=repository, check=True)
            subprocess.run([
                "git", "commit", "-m", "Archive rollout-weight 0.05 pilot results",
            ], cwd=repository, check=True)
            subprocess.run([
                "git", "push", "git@github.com:WuYanXingege/RSJG.git",
                "research/ebjd-rollout-weight005-pilot-v1",
            ], cwd=repository, check=True)
            pipeline.state["final_archive_commit"] = git_output(
                repository, "rev-parse", "HEAD")
            pipeline.state["final_push_status"] = "COMPLETED"
            pipeline.write_state()
    except BaseException as error:
        pipeline.state.pop("active_child_pid", None)
        pipeline.state["status"] = "FAILED"
        pipeline.state["failed_at"] = utc_now()
        pipeline.state["error_type"] = type(error).__name__
        pipeline.state["error"] = str(error)
        pipeline.write_state()
        raise


if __name__ == "__main__":
    main()
