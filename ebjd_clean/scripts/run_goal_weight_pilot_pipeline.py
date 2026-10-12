#!/usr/bin/env python3
"""Persistent, fail-closed serial runner for the epoch-45 goal-weight pilot."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import yaml

from ebjd.diagnostic_bank import atomic_json, sha256_file


SEEDS = (2035, 2036, 2037, 2038, 2039)
EXPECTED_MANIFEST_SHA256 = (
    "84c95cb431caf7214d1b85e8e7dbfeaf48c9fe301b1cda7f3201c3e8cf199c78")


def utc_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


class Pipeline:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = root / "PIPELINE_STATE.json"
        self.state = {
            "schema": "ebjd-goal-weight-pilot-pipeline-v1",
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
        if returncode != 0:
            self.stage(name, "FAILED", returncode=returncode,
                       elapsed_seconds=time.monotonic() - started)
            raise RuntimeError(f"stage {name} failed with exit code {returncode}")
        self.stage(name, "COMPLETED", returncode=0,
                   elapsed_seconds=time.monotonic() - started)


def load_json_lines(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def run_directory(config_path: Path, run_id: str) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return (Path(config["paths"]["output_root"]) / config["experiment"]["fold"]
            / str(config["seed"]) / "none" / run_id)


def validate_arm(run_dir: Path, expected_weight: float) -> dict:
    required = [
        run_dir / "RUN_CONTRACT.json", run_dir / "SHORT_RUN_SELECTION.json",
        run_dir / "train.jsonl", run_dir / "EPOCH_SUMMARY.jsonl",
        run_dir / "last.pt", run_dir / "best_joint_short.pt",
        run_dir / "best_marginal_short.pt", run_dir / "least_violation_short.pt",
        *(run_dir / f"epoch_{epoch:03d}.pt" for epoch in range(46, 51)),
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"arm is missing required artifacts: {missing}")
    contract = json.loads((run_dir / "RUN_CONTRACT.json").read_text(encoding="utf-8"))
    expected_contract = {
        "origin_epoch": 45, "absolute_stop_epoch": 50,
        "lr_schedule_total_epochs": 100, "origin_update_index": 11970,
        "expected_terminal_update_index": 13300,
    }
    for name, expected in expected_contract.items():
        if int(contract[name]) != expected:
            raise RuntimeError(f"run contract {name}={contract[name]} != {expected}")
    records = load_json_lines(run_dir / "train.jsonl")
    summaries = load_json_lines(run_dir / "EPOCH_SUMMARY.jsonl")
    if len(records) != 1330:
        raise RuntimeError(f"expected 1330 updates, found {len(records)}")
    if [int(item["update"]) for item in records] != list(range(11971, 13301)):
        raise RuntimeError("optimizer update indices are not exactly 11971..13300")
    if [int(item["epoch"]) for item in summaries] != list(range(46, 51)):
        raise RuntimeError("epoch summaries are not exactly 46..50")
    if any(int(item["updates"]) != 266 for item in summaries):
        raise RuntimeError("an epoch did not contain exactly 266 optimizer updates")
    if any(float(item["active_goal_v_weight"]) != expected_weight for item in records):
        raise RuntimeError("active goal weight differs from arm contract")
    if any(not all(
        isinstance(value, (int, float)) and __import__("math").isfinite(float(value))
        for key, value in item.items()
        if key not in {"logical_scene_id_sha256", "random_draw_sha256"}
    ) for item in records):
        raise RuntimeError("training record contains non-finite/non-numeric values")
    checkpoint = torch.load(
        run_dir / "epoch_050.pt", map_location="cpu", weights_only=False, mmap=True)
    if int(checkpoint["epoch"]) != 50 or int(checkpoint["update_index"]) != 13300:
        raise RuntimeError("epoch-50 checkpoint boundary is incorrect")
    if checkpoint.get("epoch_boundary_only") is not True:
        raise RuntimeError("epoch-50 checkpoint is not an epoch boundary")
    del checkpoint
    checkpoint_hashes = {
        path.name: sha256_file(path) for path in required
        if path.suffix == ".pt"
    }
    return {
        "run_directory": str(run_dir), "goal_v_weight": expected_weight,
        "updates": len(records), "first_update": records[0]["update"],
        "last_update": records[-1]["update"],
        "rollout_updates": sum(int(item["rollout_update"]) for item in records),
        "projection_active_updates": sum(
            int(float(item["projection_active"]) > 0) for item in records),
        "epoch_wall_seconds": [float(item["wall_seconds"]) for item in summaries],
        "checkpoint_sha256": checkpoint_hashes,
        "epoch50_sha256": checkpoint_hashes["epoch_050.pt"],
        "records": records,
    }


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
    return {
        "status": "PASSED" if not mismatches else "FAILED",
        "updates_compared": len(left), "compared_fields": list(fields),
        "mismatches": mismatches,
    }


def git_output(repository: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repository, text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-config", required=True)
    parser.add_argument("--candidate-config", required=True)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--source-checkpoint-sha256", required=True)
    parser.add_argument("--pipeline-root", required=True)
    parser.add_argument("--original-bank-root", required=True)
    parser.add_argument("--archive-dir", required=True)
    parser.add_argument("--control-run-id", required=True)
    parser.add_argument("--candidate-run-id", required=True)
    parser.add_argument("--push-final", action="store_true")
    args = parser.parse_args()

    repository = Path(__file__).resolve().parents[2]
    package_root = repository / "ebjd_clean"
    pipeline = Pipeline(Path(args.pipeline_root).resolve())
    try:
        control_config = Path(args.control_config).resolve()
        candidate_config = Path(args.candidate_config).resolve()
        source = Path(args.source_checkpoint).resolve()
        if sha256_file(source) != args.source_checkpoint_sha256:
            raise RuntimeError("source epoch-45 checkpoint SHA256 mismatch")
        head = git_output(repository, "rev-parse", "HEAD")
        dirty = git_output(repository, "status", "--porcelain", "--", "ebjd_clean")
        if dirty:
            raise RuntimeError("formal pipeline requires a clean ebjd_clean tree")
        for config_path in (control_config, candidate_config):
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            if config["experiment"]["source_commit"] != head:
                raise RuntimeError("materialized config is not bound to current clean HEAD")
        pipeline.state["execution_commit"] = head
        pipeline.state["source_checkpoint"] = str(source)
        pipeline.state["source_checkpoint_sha256"] = args.source_checkpoint_sha256
        pipeline.write_state()

        environment = dict(os.environ)
        environment["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        common = ["--max-epochs-this-run", "5", "--absolute-stop-epoch", "50"]
        control_command = [
            sys.executable, "-u", "-m", "ebjd.train", "--config", str(control_config),
            "--resume-from", str(source), "--resume-from-sha256",
            args.source_checkpoint_sha256, *common, "--run-id", args.control_run_id,
        ]
        candidate_command = [
            sys.executable, "-u", "-m", "ebjd.train", "--config", str(candidate_config),
            "--objective-fork-from", str(source), "--objective-fork-from-sha256",
            args.source_checkpoint_sha256, "--objective-fork-contract", "goal_v_weight_2",
            *common, "--run-id", args.candidate_run_id,
        ]
        pipeline.run("control_train", control_command, package_root, environment)
        control_dir = run_directory(control_config, args.control_run_id)
        control = validate_arm(control_dir, 1.0)
        pipeline.stage("control_validate", "COMPLETED", receipt={
            key: value for key, value in control.items() if key != "records"})
        pipeline.run("candidate_train", candidate_command, package_root, environment)
        candidate_dir = run_directory(candidate_config, args.candidate_run_id)
        candidate = validate_arm(candidate_dir, 2.0)
        pipeline.stage("candidate_validate", "COMPLETED", receipt={
            key: value for key, value in candidate.items() if key != "records"})
        pairing = validate_pair(control, candidate)
        if pairing["status"] != "PASSED":
            raise RuntimeError(f"paired training input sequence mismatch: {pairing}")
        pipeline.stage("training_pairing", "COMPLETED", receipt=pairing)

        banks = pipeline.root / "banks"
        export_script = package_root / "scripts" / "export_epoch45_diagnostic_bank.py"
        for arm, config_path, arm_receipt in (
            ("control", control_config, control),
            ("candidate", candidate_config, candidate),
        ):
            checkpoint = Path(arm_receipt["run_directory"]) / "epoch_050.pt"
            command = [
                sys.executable, "-u", str(export_script), "--config", str(config_path),
                "--checkpoint", str(checkpoint), "--output-root", str(banks),
                "--expected-checkpoint-sha256", arm_receipt["epoch50_sha256"],
                "--expected-config-sha256", sha256_file(config_path),
                "--expected-manifest-sha256", EXPECTED_MANIFEST_SHA256,
                "--bank-name", arm, "--model-label", f"EBJD_{arm}_epoch50",
                "--seeds", *(str(seed) for seed in SEEDS), "--device", "cuda", "--resume",
            ]
            pipeline.run(f"{arm}_bank_export", command, package_root, environment)

        training_receipt = {
            "schema": "ebjd-goal-weight-pilot-training-pairing-v1",
            "status": "PASSED", "execution_commit": head,
            "source_checkpoint_sha256": args.source_checkpoint_sha256,
            "control": control, "candidate": candidate, "pairing": pairing,
            "control_config": str(control_config),
            "control_config_sha256": sha256_file(control_config),
            "candidate_config": str(candidate_config),
            "candidate_config_sha256": sha256_file(candidate_config),
        }
        training_receipt_path = pipeline.root / "TRAINING_AND_PAIRING_RECEIPT.json"
        atomic_json(training_receipt, training_receipt_path)

        original = Path(args.original_bank_root).resolve()
        result_dir = pipeline.root / "analysis"
        archive = Path(args.archive_dir).resolve()
        analysis_command = [
            sys.executable, "-u", str(package_root / "scripts" / "analyze_goal_weight_pilot_banks.py"),
            "--gdts-bank", str(original / "gdts"),
            "--epoch45-bank", str(original / "ebjd"),
            "--control-bank", str(banks / "control"),
            "--candidate-bank", str(banks / "candidate"),
            "--output-dir", str(result_dir), "--archive-dir", str(archive),
        ]
        pipeline.run("cpu_final_analysis", analysis_command, package_root, environment)
        archive.mkdir(parents=True, exist_ok=True)
        shutil.copy2(training_receipt_path, archive / training_receipt_path.name)
        shutil.copy2(pipeline.state_path, archive / "PIPELINE_STATE_PRE_ARCHIVE.json")

        pipeline.state["status"] = "COMPLETED"
        pipeline.state["completed_at"] = utc_now()
        pipeline.state["final_result_receipt"] = str(result_dir / "FINAL_RESULT_RECEIPT.json")
        pipeline.write_state()
        shutil.copy2(pipeline.state_path, archive / "PIPELINE_STATE.json")
        if args.push_final:
            relative = archive.relative_to(repository)
            subprocess.run(["git", "add", "--", str(relative)], cwd=repository, check=True)
            subprocess.run([
                "git", "commit", "-m", "Archive epoch45 paired goal weight pilot results",
            ], cwd=repository, check=True)
            subprocess.run([
                "git", "push", "git@github.com:WuYanXingege/RSJG.git",
                "research/ebjd-goalweight-paired-pilot-v1",
            ], cwd=repository, check=True)
            pipeline.state["final_archive_commit"] = git_output(repository, "rev-parse", "HEAD")
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
