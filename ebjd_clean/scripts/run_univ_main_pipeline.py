#!/usr/bin/env python3
"""Persistent goal-pretrain -> fold EBJD -> P20 evaluation supervisor."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import torch


SEEDS = (2035, 2036, 2037, 2038, 2039)


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(payload: dict, path: Path) -> None:
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


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


class Supervisor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.fold = args.fold.lower()
        self.prefix = self.fold.upper()
        self.repo = Path(__file__).resolve().parents[2]
        self.ebjd_root = self.repo / "ebjd_clean"
        self.workspace = self.repo.parent
        self.goal_repo = self.workspace / "RSJG_JDV2_clean"
        self.python = Path(args.python).resolve()
        self.runtime = Path(args.runtime_dir).resolve()
        self.runtime.mkdir(parents=True, exist_ok=True)
        self.state_path = self.runtime / "PIPELINE_STATE.json"
        self.state = {
            "schema": f"ebjd-{self.fold}-main-pipeline-state-v1",
            "status": "STARTING", "stage": "STARTING",
            "supervisor_pid": os.getpid(), "started_at": now(),
            "updated_at": now(), "source_worktree": str(self.repo),
            "runtime_dir": str(self.runtime),
        }
        self.update()

    def update(self, **fields) -> None:
        self.state.update(fields)
        self.state["updated_at"] = now()
        atomic_json(self.state, self.state_path)

    def progress(self, stage: str) -> dict:
        result = {"stage": stage}
        if stage == "GOAL_PRETRAIN_RUNNING":
            curve = self.goal_run_dir / "pretrain_log_curve.txt"
            if curve.is_file():
                lines = [line for line in curve.read_text().splitlines()[1:] if line.strip()]
                if lines:
                    result["completed_epoch"] = int(lines[-1].split(",", 1)[0])
                    result["planned_epochs"] = 150
        if stage == "EBJD_TRAINING_RUNNING" and self.run_dir.is_dir():
            summary = self.run_dir / "EPOCH_SUMMARY.jsonl"
            train_log = self.run_dir / "train.jsonl"
            if summary.is_file() and summary.stat().st_size:
                rows = [line for line in summary.read_text().splitlines() if line.strip()]
                last = json.loads(rows[-1])
                result.update({
                    "completed_epoch": int(last["epoch"]),
                    "planned_epochs": 100,
                    "completed_update": int(last["end_update_inclusive"]),
                    "last_validation": last["validation"],
                    "last_epoch_wall_seconds": last["wall_seconds"],
                })
                if len(rows) >= 2:
                    recent = [json.loads(line)["wall_seconds"] for line in rows[-3:]]
                    result["rough_remaining_seconds"] = (
                        (100 - int(last["epoch"])) * sum(recent) / len(recent))
            elif train_log.is_file() and train_log.stat().st_size:
                last = json.loads(train_log.read_text().splitlines()[-1])
                result.update({
                    "active_epoch": int(last["epoch"]),
                    "completed_update": int(last["update"]),
                    "planned_updates": getattr(self, "expected_terminal_update", None),
                })
        return result

    def run_logged(self, command: list[str], cwd: Path, log: Path, stage: str) -> None:
        log.parent.mkdir(parents=True, exist_ok=True)
        environment = dict(os.environ)
        environment.update({
            "PYTHONPATH": ".", "PYTHONUNBUFFERED": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
        })
        self.update(
            status="RUNNING", stage=stage, command=command,
            stage_log=str(log), stage_started_at=now())
        with log.open("ab", buffering=0) as handle:
            process = subprocess.Popen(
                command, cwd=cwd, env=environment, stdout=handle,
                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                start_new_session=False)
            self.update(child_pid=process.pid)
            while process.poll() is None:
                self.update(**self.progress(stage))
                time.sleep(30)
            code = int(process.returncode)
        self.update(child_pid=None, last_exit_code=code)
        if code != 0:
            raise RuntimeError(f"{stage} exited with code {code}; see {log}")

    def run_goal_pretrain(self) -> Path:
        self.goal_run_id = self.args.goal_run_id
        self.goal_run_dir = (
            self.goal_repo / "output" / self.fold / "runs" / self.goal_run_id)
        best = self.goal_run_dir / "saved_models" / "goal_pretrain_best_model.pt"
        last = self.goal_run_dir / "saved_models" / "goal_pretrain_last_model.pt"
        completed = False
        if last.is_file():
            payload = torch.load(last, map_location="cpu", weights_only=False)
            completed = int(payload.get("epoch", -1)) == 150
        if not completed:
            command = [
                str(self.python), "-u", "-B", "main.py",
                "--dataset", "eth5", "--test_set", self.fold,
                "--phase", "goal_pretrain", "--run_name", self.goal_run_id,
                "--num_epochs", "150", "--batch_size", "64",
                "--start_validation", "1", "--validate_every", "1",
                "--learning_rate", "0.001", "--optimizer", "Adam",
                "--scheduler", "ExponentialLR", "--num_workers", "0",
                "--clip", "1", "--data_augmentation", "False",
                "--shuffle_train_batches", "False",
                "--shuffle_test_batches", "False",
                "--reproducibility", "True", "--seed", "3101",
                "--device", "cuda:0", "--use_wandb", "False",
                "--amp_enabled", "False", "--amp_dtype", "fp32",
                "--force_reprocess", "False", "--save_every", "10",
            ]
            if last.is_file():
                command.extend(("--load_checkpoint", "last"))
            self.run_logged(
                command, self.goal_repo, self.runtime / "goal_pretrain.log",
                "GOAL_PRETRAIN_RUNNING")
        if not best.is_file() or not last.is_file():
            raise RuntimeError("goal pretraining ended without best/last checkpoints")
        last_payload = torch.load(last, map_location="cpu", weights_only=False)
        if int(last_payload.get("epoch", -1)) != 150:
            raise RuntimeError("goal pretraining did not complete epoch 150")
        receipt = {
            "stage": "GOAL_PRETRAIN_COMPLETE", "run_id": self.goal_run_id,
            "run_dir": str(self.goal_run_dir),
            "best_checkpoint": str(best), "best_checkpoint_sha256": sha256(best),
            "best_epoch": int(torch.load(
                best, map_location="cpu", weights_only=False)["epoch"]),
            "last_checkpoint": str(last), "last_checkpoint_sha256": sha256(last),
            "last_epoch": 150,
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=self.goal_repo,
                text=True).strip(),
            "source_worktree_has_untracked_runtime_files": bool(
                subprocess.check_output(
                    ["git", "status", "--porcelain"], cwd=self.goal_repo,
                    text=True).strip()),
        }
        atomic_json(receipt, self.runtime / "GOAL_PRETRAIN_RECEIPT.json")
        self.update(status="RUNNING", **receipt)
        return best

    def materialize(self, goal_checkpoint: Path) -> Path:
        self.archive = self.runtime / "archive"
        self.formal_output_root = self.runtime / "formal_training"
        config = self.archive / f"endpoint_bridge_joint_{self.fold}_seed3101.yaml"
        command = [
            str(self.python), "-u", "-B",
            "scripts/prepare_univ_main_experiment.py",
            "--fold", self.fold,
            "--base-univ-config", str(Path(self.args.base_config).resolve()),
            "--hotel-checkpoint", str(Path(self.args.hotel_checkpoint).resolve()),
            "--goal-checkpoint", str(goal_checkpoint),
            "--experiment-output-root", str(self.formal_output_root),
            "--archive-dir", str(self.archive), "--output-config", str(config),
        ]
        self.run_logged(
            command, self.ebjd_root, self.runtime / "materialization.log",
            "MATERIALIZING_FORMAL_CONFIG")
        receipt = json.loads((self.archive / "MATERIALIZATION_RECEIPT.json").read_text())
        self.expected_updates_per_epoch = int(receipt["expected_updates_per_epoch"])
        self.expected_terminal_update = int(receipt["expected_terminal_update_index"])
        self.update(status="RUNNING", stage="FORMAL_CONFIG_READY",
                    formal_config=str(config), formal_config_sha256=sha256(config))
        return config

    def preflight(self, config: Path) -> None:
        output = self.archive / f"{self.prefix}_CUDA_PREFLIGHT.json"
        self.run_logged([
            str(self.python), "-u", "-B", "scripts/preflight_univ_main.py",
            "--config", str(config), "--output", str(output),
        ], self.ebjd_root, self.runtime / "cuda_preflight.log", "CUDA_PREFLIGHT_RUNNING")
        payload = json.loads(output.read_text())
        if payload["status"] != "PASS":
            raise RuntimeError(f"targeted {self.prefix} CUDA preflight failed")
        self.update(status="RUNNING", stage="CUDA_PREFLIGHT_PASS",
                    cuda_preflight=str(output), cuda_preflight_sha256=sha256(output))

    def train(self, config: Path) -> Path:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=self.repo, text=True).strip()
        self.run_id = self.args.run_id or (
            f"{self.fold}_main_seed3101_{commit[:7]}_{self.args.date_tag}")
        self.run_dir = (
            self.formal_output_root / self.fold / "3101" / "none" / self.run_id)
        last = self.run_dir / "last.pt"
        completed = False
        summary = self.run_dir / "EPOCH_SUMMARY.jsonl"
        if summary.is_file() and summary.stat().st_size:
            completed = int(json.loads(summary.read_text().splitlines()[-1])["epoch"]) == 100
        if not completed:
            command = [
                str(self.python), "-u", "-B", "-m", "ebjd.train",
                "--config", str(config), "--device", "cuda",
                "--ablation", "none", "--run-id", self.run_id,
            ]
            if last.is_file():
                command.extend(("--resume", str(last)))
            self.run_logged(
                command, self.ebjd_root, self.runtime / "ebjd_training.log",
                "EBJD_TRAINING_RUNNING")
        if not summary.is_file() or int(json.loads(
                summary.read_text().splitlines()[-1])["epoch"]) != 100:
            raise RuntimeError("formal EBJD training did not complete epoch 100")
        if int(json.loads(summary.read_text().splitlines()[-1])[
                "end_update_inclusive"]) != self.expected_terminal_update:
            raise RuntimeError(
                "formal EBJD terminal update index differs from materialized contract: "
                f"{json.loads(summary.read_text().splitlines()[-1])['end_update_inclusive']} "
                f"!= {self.expected_terminal_update}")
        if not (self.run_dir / "epoch_045.pt").is_file():
            raise RuntimeError(f"formal {self.prefix} epoch-45 reference checkpoint is missing")
        self.update(status="RUNNING", stage="EBJD_TRAINING_COMPLETE",
                    run_dir=str(self.run_dir), completed_epoch=100,
                    completed_update=self.expected_terminal_update)
        return self.run_dir

    def freeze_selection(self, run_dir: Path) -> Path:
        selection = json.loads((run_dir / "SELECTION_STATUS.json").read_text())
        role = "best_eligible" if selection["best_eligible"] is not None else "least_violation"
        candidate = selection[role]
        source = run_dir / ("best.pt" if role == "best_eligible" else "least_violation.pt")
        payload = torch.load(source, map_location="cpu", weights_only=False)
        if int(payload["epoch"]) != int(candidate["epoch"]):
            raise RuntimeError("selector record and saved checkpoint epoch differ")
        selected = run_dir / "selected_formal.pt"
        atomic_copy(source, selected)
        receipt = {
            "schema": "ebjd-formal-selection-freeze-v1", "role": role,
            "candidate": candidate, "source_checkpoint": str(source),
            "selected_checkpoint": str(selected),
            "selected_checkpoint_sha256": sha256(selected),
            "selected_epoch": int(payload["epoch"]), "frozen_at": now(),
        }
        atomic_json(receipt, run_dir / "FORMAL_SELECTION_RECEIPT.json")
        self.update(status="RUNNING", stage="FORMAL_SELECTION_FROZEN", **receipt)
        return selected

    def evaluate(self, config: Path, checkpoint: Path) -> Path:
        output = self.runtime / "evaluation" / f"EBJD_{self.prefix}_SEEDS2035_2039.json"
        before = sha256(checkpoint)
        self.run_logged([
            str(self.python), "-u", "-B", "-m", "ebjd.evaluate",
            "--config", str(config), "--checkpoint", str(checkpoint),
            "--split", "test", "--ablation", "none", "--device", "cuda",
            "--output", str(output),
        ], self.ebjd_root, self.runtime / "formal_evaluation.log",
            "FORMAL_EVALUATION_RUNNING")
        if sha256(checkpoint) != before:
            raise RuntimeError("selected checkpoint changed during evaluation")
        self.update(status="RUNNING", stage="FORMAL_EVALUATION_COMPLETE",
                    evaluation_result=str(output), evaluation_result_sha256=sha256(output))
        return output

    def evaluate_gdts_baseline(self) -> Path:
        output = Path(self.args.gdts_five_seed_result).resolve()
        self.run_logged([
            str(self.python), "-u", "-B",
            str(Path(self.args.gdts_evaluator_script).resolve()),
            "--jdv2-root", str(Path(self.args.gdts_root).resolve()),
            "--config", str(Path(self.args.gdts_config).resolve()),
            "--checkpoint", str(Path(self.args.gdts_checkpoint).resolve()),
            "--manifest", str(Path(self.args.gdts_manifest).resolve()),
            "--output", str(output), "--device", "cuda:0",
            "--seeds", *(str(seed) for seed in SEEDS),
        ], Path(self.args.gdts_root).resolve(),
            self.runtime / "gdts_five_seed_evaluation.log",
            "GDTS_BASELINE_FIVE_SEED_RUNNING")
        payload = json.loads(output.read_text())
        observed = [int(record["seed"]) for record in payload.get("runs", [])]
        if payload.get("status") != "complete" or observed != list(SEEDS):
            raise RuntimeError(
                f"GDTS baseline evaluation is incomplete: {observed} != {list(SEEDS)}")
        self.update(
            status="RUNNING", stage="GDTS_BASELINE_FIVE_SEED_COMPLETE",
            gdts_result=str(output), gdts_result_sha256=sha256(output))
        return output

    def finalize(
            self, run_dir: Path, checkpoint: Path, result: Path,
            baseline: Path) -> None:
        self.run_logged([
            str(self.python), "-u", "-B", "scripts/finalize_univ_main.py",
            "--fold", self.fold,
            "--run-dir", str(run_dir), "--selected-checkpoint", str(checkpoint),
            "--ebjd-result", str(result), "--gdts-result", str(baseline),
            "--archive-dir", str(self.archive),
        ], self.ebjd_root, self.runtime / "finalization.log", "FINALIZING_RESULTS")
        small = {
            "RESOLVED_CONFIG.json", "RUN_IDENTITY.json", "RUN_CONTRACT.json",
            "SELECTION_STATUS.json", "AUXILIARY_SELECTION.json",
            "EPOCH_SUMMARY.jsonl", "FORMAL_SELECTION_RECEIPT.json",
        }
        for name in small:
            source = run_dir / name
            if source.is_file():
                atomic_copy(source, self.archive / name)
        atomic_copy(result, self.archive / result.name)
        atomic_copy(baseline, self.archive / baseline.name)
        checkpoints = {}
        for name in (
            "epoch_045.pt", "last.pt", "best.pt", "least_violation.pt",
            "best_joint_auxiliary.pt", "selected_formal.pt"):
            path = run_dir / name
            if path.is_file():
                checkpoints[name] = {
                    "path": str(path), "bytes": path.stat().st_size,
                    "sha256": sha256(path)}
        logs = {}
        for path in (
            run_dir / "train.jsonl", self.runtime / "goal_pretrain.log",
            self.runtime / "ebjd_training.log", self.runtime / "formal_evaluation.log",
            self.runtime / "gdts_five_seed_evaluation.log"):
            if path.is_file():
                logs[path.name] = {
                    "path": str(path), "bytes": path.stat().st_size,
                    "sha256": sha256(path)}
        atomic_json({
            "schema": f"ebjd-{self.fold}-artifact-index-v1", "checkpoints": checkpoints,
            "logs": logs, "large_artifacts_remain_on_host": True,
            "github_archive_contains_structured_small_files_only": True,
        }, self.archive / "ARTIFACT_INDEX.json")
        self.archive_git()

    def archive_git(self) -> None:
        target = (
            self.ebjd_root / "docs" / "experiments" /
            f"{self.args.date_tag}_{self.fold}_main_seed3101")
        if target.exists():
            raise FileExistsError(f"refuse to replace existing archive: {target}")
        shutil.copytree(self.archive, target)
        subprocess.run(["git", "add", str(target.relative_to(self.repo))],
                       cwd=self.repo, check=True)
        staged = subprocess.check_output(
            ["git", "diff", "--cached", "--name-only"], cwd=self.repo,
            text=True).strip()
        if staged:
            subprocess.run([
                "git", "commit", "-m",
                f"Archive formal {self.prefix} EBJD seed 3101 results",
            ], cwd=self.repo, check=True)
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=self.repo, text=True).strip()
        push = subprocess.run(
            ["git", "push", "-u", "origin",
             self.args.git_branch],
            cwd=self.repo, text=True, capture_output=True)
        push_receipt = {
            "attempted_at": now(), "commit": commit,
            "returncode": push.returncode, "stdout": push.stdout,
            "stderr": push.stderr,
            "status": "PUSHED" if push.returncode == 0 else "PUSH_BLOCKED_AUTH",
        }
        atomic_json(push_receipt, self.runtime / "GITHUB_PUSH_RECEIPT.json")
        self.update(
            status="COMPLETED", stage="COMPLETED", completed_at=now(),
            archive_commit=commit, github_push_status=push_receipt["status"],
            tracked_archive=str(target))

    def execute(self) -> None:
        goal = self.run_goal_pretrain()
        config = self.materialize(goal)
        self.preflight(config)
        run_dir = self.train(config)
        checkpoint = self.freeze_selection(run_dir)
        result = self.evaluate(config, checkpoint)
        baseline = self.evaluate_gdts_baseline()
        self.finalize(run_dir, checkpoint, result, baseline)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", required=True)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--date-tag", required=True)
    parser.add_argument("--git-branch", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--runtime-dir", required=True)
    parser.add_argument("--goal-run-id", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--hotel-checkpoint", required=True)
    parser.add_argument("--gdts-five-seed-result", required=True)
    parser.add_argument("--gdts-evaluator-script", required=True)
    parser.add_argument("--gdts-root", required=True)
    parser.add_argument("--gdts-config", required=True)
    parser.add_argument("--gdts-checkpoint", required=True)
    parser.add_argument("--gdts-manifest", required=True)
    args = parser.parse_args()
    supervisor = Supervisor(args)
    try:
        supervisor.execute()
    except BaseException as exc:
        supervisor.update(
            status="FAILED", stage="FAILED", failed_at=now(),
            error_type=type(exc).__name__, error=str(exc))
        raise


if __name__ == "__main__":
    main()
