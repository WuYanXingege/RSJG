#!/usr/bin/env python3
"""Authenticated post-task-only recovery for the HOTEL A0/A1 queue."""

from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from hotel_a0_a1_queue import (
    atomic_json,
    gpu_processes,
    sha256,
    start_ticks,
    utc_now,
    verified_terminate,
)


ORIGINAL_MANIFEST_SHA256 = (
    "530d210f744b1576c69fe1f13d23441c1cf8517e5242d9bdb22317adb4b4ec19")
ORIGINAL_SOURCE_COMMIT = "664ad0410f5cf3b44d3a6c1892c98e37aa936b9e"
EXPECTED_FAILED_TASK = "E_joint_baseline"
ALLOWED_CODE_CHANGES = {
    "tools/hotel_a0_a1_baseline_eval.py",
    "tools/hotel_a0_a1_post_resume.py",
}


def git_output(source_dir: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=source_dir, text=True,
        capture_output=True, check=True).stdout.strip()


def validate_source(source_dir: Path, manifest: dict) -> str:
    current = git_output(source_dir, "rev-parse", "HEAD")
    if manifest.get("run_source_commit") != ORIGINAL_SOURCE_COMMIT:
        raise RuntimeError("Unexpected original queue source commit")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ORIGINAL_SOURCE_COMMIT,
         current], cwd=source_dir).returncode
    if ancestor != 0 or current == ORIGINAL_SOURCE_COMMIT:
        raise RuntimeError("Postfix source is not a descendant repair commit")
    dirty = git_output(
        source_dir, "status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise RuntimeError("Execution worktree has tracked modifications")
    changed = set(filter(None, git_output(
        source_dir, "diff", "--name-only",
        f"{ORIGINAL_SOURCE_COMMIT}..{current}").splitlines()))
    code_changes = {name for name in changed if not name.startswith("docs/")}
    if not code_changes.issubset(ALLOWED_CODE_CHANGES):
        raise RuntimeError(
            f"Postfix contains unauthorized code changes: {sorted(code_changes)}")
    if "tools/hotel_a0_a1_baseline_eval.py" not in code_changes:
        raise RuntimeError("Postfix does not contain the baseline loader repair")
    return current


def validate_failure_and_training(run_dir: Path, manifest: dict) -> dict:
    preserved_progress = run_dir / "QUEUE_PROGRESS_PRE_POSTFIX.json"
    progress_path = (preserved_progress if preserved_progress.is_file()
                     else run_dir / "QUEUE_PROGRESS.json")
    progress = json.loads(progress_path.read_text())
    if (progress.get("status") != "CONTRACT_FAILED" or
            progress.get("failed_task") != EXPECTED_FAILED_TASK):
        raise RuntimeError("Queue is not at the authenticated baseline failure")
    failed_receipt = run_dir / "receipts" / f"{EXPECTED_FAILED_TASK}.json"
    failed = json.loads(failed_receipt.read_text())
    if failed.get("status") != "CONTRACT_FAILED" or failed.get(
            "returncode") != 1:
        raise RuntimeError("Baseline failure receipt mismatch")
    completed = []
    for arm in manifest["arms"]:
        label = f'{arm["arm"]}_{arm["seed"]}'
        receipt_path = run_dir / "receipts" / f"{label}.json"
        receipt = json.loads(receipt_path.read_text())
        checkpoint = Path(receipt.get("path", ""))
        if (receipt.get("status") != "COMPLETED" or
                receipt.get("returncode") != 0 or
                receipt.get("epoch") != 40 or
                not checkpoint.is_file() or
                sha256(checkpoint) != receipt.get("sha256")):
            raise RuntimeError(f"Completed arm receipt mismatch: {label}")
        completed.append(label)
    return {
        "completed_arms": completed,
        "failed_progress_sha256": sha256(progress_path),
        "failed_receipt_sha256": sha256(failed_receipt),
    }


def preserve_once(source: Path, target: Path) -> None:
    if target.exists():
        return
    shutil.copy2(source, target)


def completed_postfix_task(run_dir: Path, task: dict) -> dict | None:
    receipt_path = run_dir / "receipts" / f'{task["name"]}.postfix.json'
    result_path = Path(task["result"])
    if not receipt_path.is_file():
        return None
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get("status") != "COMPLETED" or not result_path.is_file() or
            sha256(result_path) != receipt.get("result_sha256")):
        raise RuntimeError(f"Invalid postfix receipt: {task['name']}")
    return receipt


def build_results(run_dir: Path, manifest: dict, repair: dict) -> None:
    arm_receipts = {}
    for arm in manifest["arms"]:
        label = f'{arm["arm"]}_{arm["seed"]}'
        arm_receipts[label] = json.loads(
            (run_dir / "receipts" / f"{label}.json").read_text())
    evaluations = {}
    task_receipts = {}
    for task in manifest.get("post_tasks", []):
        evaluations[task["name"]] = json.loads(Path(task["result"]).read_text())
        task_receipts[task["name"]] = json.loads(
            (run_dir / "receipts" /
             f'{task["name"]}.postfix.json').read_text())
    atomic_json(run_dir / "RESULTS.json", {
        "status": "COMPLETED_POSTFIX_RECOVERY",
        "parent_sha256": manifest["parent_sha256"],
        "training_order": manifest["training_order"],
        "evaluation_seeds": manifest["evaluation_seeds"],
        "arm_receipts": arm_receipts,
        "postfix_task_receipts": task_receipts,
        "evaluations": evaluations,
        "repair": repair,
        "limitations": [
            "HOTEL official validation and test bytes are identical",
            "three heads share one selected GDTS parent",
            "this is not an independent held-out or cross-scene result",
        ],
    })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    manifest_path = Path(args.manifest).resolve()
    if sha256(manifest_path) != ORIGINAL_MANIFEST_SHA256:
        raise RuntimeError("Frozen queue manifest SHA256 mismatch")
    manifest = json.loads(manifest_path.read_text())
    source_dir = Path(manifest["source_dir"]).resolve()
    postfix_commit = validate_source(source_dir, manifest)
    run_dir = manifest_path.parent
    validation = validate_failure_and_training(run_dir, manifest)
    repair = {
        "reason": "baseline evaluator selected legacy cache because raw YAML stored jdv2_active=null",
        "scope": "post_tasks_only_no_training",
        "original_manifest_sha256": ORIGINAL_MANIFEST_SHA256,
        "original_source_commit": ORIGINAL_SOURCE_COMMIT,
        "postfix_source_commit": postfix_commit,
        **validation,
    }
    if args.preflight_only:
        print(json.dumps({"status": "POSTFIX_PREFLIGHT_PASS", **repair},
                         indent=2, sort_keys=True))
        return 0

    deadline = datetime.datetime.fromisoformat(
        manifest["deadline_utc"].replace("Z", "+00:00"))
    if utc_now() >= deadline:
        raise RuntimeError("Original queue deadline has already passed")
    lock = run_dir / "QUEUE_POSTFIX.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    os.write(descriptor, str(os.getpid()).encode())
    os.close(descriptor)
    preserve_once(
        run_dir / "QUEUE_PROGRESS.json",
        run_dir / "QUEUE_PROGRESS_PRE_POSTFIX.json")
    preserve_once(
        run_dir / "receipts" / f"{EXPECTED_FAILED_TASK}.json",
        run_dir / "receipts" / f"{EXPECTED_FAILED_TASK}.pre_postfix.json")
    progress = {
        "status": "TRAINING_COMPLETED_POSTFIX_EVALUATION_RUNNING",
        "started_utc": utc_now().isoformat(),
        "deadline_utc": deadline.isoformat(),
        "completed_arms": validation["completed_arms"],
        "completed_tasks": [],
        "current_task": None,
        "repair": repair,
    }
    atomic_json(run_dir / "QUEUE_PROGRESS.json", progress)
    try:
        for task in manifest.get("post_tasks", []):
            prior = completed_postfix_task(run_dir, task)
            if prior is not None:
                progress["completed_tasks"].append(task["name"])
                atomic_json(run_dir / "QUEUE_PROGRESS.json", progress)
                continue
            if utc_now() >= deadline:
                progress.update({
                    "status": "BUDGET_CENSORED",
                    "reason": "original queue deadline reached before postfix task",
                    "current_task": task["name"],
                })
                atomic_json(run_dir / "QUEUE_PROGRESS.json", progress)
                return 3
            processes = gpu_processes()
            if processes:
                progress.update({
                    "status": "RESOURCE_BLOCKED",
                    "current_task": task["name"],
                    "gpu_processes": processes,
                })
                atomic_json(run_dir / "QUEUE_PROGRESS.json", progress)
                return 4
            name = task["name"]
            identity_path = run_dir / "workers" / f"{name}.postfix.identity.json"
            log_path = run_dir / "logs" / f"{name}.postfix.log"
            wrapper = [
                manifest["python"], "-u",
                str(source_dir / "tools" / "hotel_a0_a1_task_worker.py"),
                "--identity", str(identity_path), "--cwd", str(source_dir),
                *task["command"],
            ]
            with log_path.open("ab", buffering=0) as log:
                process = subprocess.Popen(
                    wrapper, cwd=source_dir, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True)
            observed_ticks = start_ticks(process.pid)
            progress.update({
                "current_task": name,
                "worker_pid": process.pid,
                "worker_proc_start_ticks": observed_ticks,
                "worker_pgid": os.getpgid(process.pid),
                "worker_sid": os.getsid(process.pid),
                "worker_log": str(log_path),
            })
            atomic_json(run_dir / "QUEUE_PROGRESS.json", progress)
            while process.poll() is None:
                if utc_now() >= deadline:
                    verified_terminate(process.pid, observed_ticks)
                    progress.update({
                        "status": "BUDGET_CENSORED",
                        "reason": "original queue deadline reached during postfix task",
                    })
                    atomic_json(run_dir / "QUEUE_PROGRESS.json", progress)
                    return 3
                time.sleep(15)
            result_source = Path(task.get(
                "result_source", task.get("result", "")))
            result_target = Path(task["result"])
            receipt = {
                "name": name,
                "returncode": process.returncode,
                "finished_utc": utc_now().isoformat(),
                "log": str(log_path),
                "postfix_source_commit": postfix_commit,
            }
            if process.returncode == 0 and result_source.is_file():
                result_target.parent.mkdir(parents=True, exist_ok=True)
                if result_source.resolve() != result_target.resolve():
                    shutil.copy2(result_source, result_target)
                receipt.update({
                    "status": "COMPLETED",
                    "result": str(result_target),
                    "result_sha256": sha256(result_target),
                })
            else:
                receipt["status"] = "CONTRACT_FAILED"
            atomic_json(
                run_dir / "receipts" / f"{name}.postfix.json", receipt)
            if receipt["status"] != "COMPLETED":
                progress.update({
                    "status": "CONTRACT_FAILED",
                    "failed_task": name,
                    "task_receipt": receipt,
                })
                atomic_json(run_dir / "QUEUE_PROGRESS.json", progress)
                return 6
            progress["completed_tasks"].append(name)
            progress["current_task"] = None
            atomic_json(run_dir / "QUEUE_PROGRESS.json", progress)
        build_results(run_dir, manifest, repair)
        progress.update({
            "status": "COMPLETED_POSTFIX_RECOVERY",
            "current_task": None,
            "finished_utc": utc_now().isoformat(),
            "results": str(run_dir / "RESULTS.json"),
            "results_sha256": sha256(run_dir / "RESULTS.json"),
        })
        atomic_json(run_dir / "QUEUE_PROGRESS.json", progress)
        return 0
    finally:
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    sys.exit(main())
