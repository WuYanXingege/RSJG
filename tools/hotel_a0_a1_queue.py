#!/usr/bin/env python3
"""Fail-closed sequential queue for the bounded HOTEL A0/A1 arms."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import signal
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc)


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def start_ticks(pid: int) -> int:
    return int(Path(f"/proc/{pid}/stat").read_text().split()[21])


def gpu_processes():
    command = [
        "nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=True)
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def identify(path: Path, intended) -> dict:
    pid = os.getpid()
    return {
        "pid": pid, "ppid": os.getppid(), "pgid": os.getpgid(pid),
        "sid": os.getsid(pid), "proc_start_ticks": start_ticks(pid),
        "started_utc": utc_now().isoformat(), "intended_manifest": intended,
        "cmdline": Path(f"/proc/{pid}/cmdline").read_bytes().replace(
            b"\0", b" ").decode(errors="replace").strip(),
    }


def verified_terminate(pid: int, expected_ticks: int) -> None:
    try:
        if start_ticks(pid) != expected_ticks:
            raise RuntimeError("Refusing to signal reused/unverified PID")
        os.kill(pid, signal.SIGTERM)
    except FileNotFoundError:
        return


def checkpoint_receipt(model_dir: Path) -> dict:
    checkpoint = model_dir / "saved_models" / "last_model.pt"
    if not checkpoint.is_file():
        return {"checkpoint_exists": False}
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return {
        "checkpoint_exists": True, "path": str(checkpoint),
        "sha256": sha256(checkpoint), "epoch": int(payload.get("epoch", -1)),
        "successful_updates": int(payload.get(
            "stage_optimizer_steps_completed", -1)),
        "optimizer_attempts": int(payload.get("optimizer_attempts", -1)),
        "checkpoint_role": payload.get("checkpoint_role"),
        "resume_safe": payload.get("resume_safe"),
        "best_selection": payload.get("best_selection"),
        "frozen_state_sha256": payload.get("frozen_state_sha256"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text())
    run_dir = manifest_path.parent
    lock = run_dir / "QUEUE.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    os.write(descriptor, str(os.getpid()).encode())
    os.close(descriptor)
    atomic_json(run_dir / "manager_identity.json", identify(
        run_dir / "manager_identity.json", str(manifest_path)))
    deadline = datetime.datetime.fromisoformat(
        manifest["deadline_utc"].replace("Z", "+00:00"))
    progress = {
        "status": "HOTEL_A0_A1_RUNNING", "started_utc": utc_now().isoformat(),
        "deadline_utc": deadline.isoformat(), "completed_arms": [],
        "current_arm": None,
    }
    atomic_json(run_dir / "QUEUE_PROGRESS.json", progress)
    try:
        for arm in manifest["arms"]:
            if utc_now() >= deadline:
                progress["status"] = "BUDGET_CENSORED"
                progress["reason"] = "queue deadline reached before arm start"
                atomic_json(run_dir / "QUEUE_PROGRESS.json", progress)
                return 3
            processes = gpu_processes()
            if processes:
                progress["status"] = "RESOURCE_BLOCKED"
                progress["gpu_processes"] = processes
                atomic_json(run_dir / "QUEUE_PROGRESS.json", progress)
                return 4
            label = f'{arm["arm"]}_{arm["seed"]}'
            identity_path = run_dir / "workers" / f"{label}.identity.json"
            log_path = run_dir / "logs" / f"{label}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            command = [
                manifest["python"], "-u",
                str(Path(manifest["source_dir"]) /
                    "tools/hotel_a0_a1_worker.py"),
                "--identity", str(identity_path),
                "--python", manifest["python"],
                "--source-dir", manifest["source_dir"],
                "--run-name", arm["run_name"],
            ]
            with log_path.open("ab", buffering=0) as log:
                process = subprocess.Popen(
                    command, cwd=manifest["source_dir"], stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True)
            observed_ticks = start_ticks(process.pid)
            progress.update({
                "current_arm": label, "worker_pid": process.pid,
                "worker_proc_start_ticks": observed_ticks,
                "worker_pgid": os.getpgid(process.pid),
                "worker_sid": os.getsid(process.pid),
                "worker_log": str(log_path),
            })
            atomic_json(run_dir / "QUEUE_PROGRESS.json", progress)
            while process.poll() is None:
                if utc_now() >= deadline:
                    verified_terminate(process.pid, observed_ticks)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        if start_ticks(process.pid) == observed_ticks:
                            os.kill(process.pid, signal.SIGKILL)
                    progress["status"] = "BUDGET_CENSORED"
                    progress["reason"] = "queue deadline reached during arm"
                    progress["last_checkpoint"] = checkpoint_receipt(
                        Path(arm["model_dir"]))
                    atomic_json(run_dir / "QUEUE_PROGRESS.json", progress)
                    return 3
                time.sleep(15)
            receipt = {
                "arm": label, "returncode": process.returncode,
                "finished_utc": utc_now().isoformat(),
                "log": str(log_path),
                **checkpoint_receipt(Path(arm["model_dir"])),
            }
            receipt["status"] = (
                "COMPLETED" if process.returncode == 0 and
                receipt.get("epoch") == 40 else "CONTRACT_FAILED")
            atomic_json(run_dir / "receipts" / f"{label}.json", receipt)
            if receipt["status"] != "COMPLETED":
                progress.update({"status": "CONTRACT_FAILED",
                                 "failed_arm": label, "receipt": receipt})
                atomic_json(run_dir / "QUEUE_PROGRESS.json", progress)
                return 5
            progress["completed_arms"].append(label)
            progress["current_arm"] = None
            atomic_json(run_dir / "QUEUE_PROGRESS.json", progress)
        progress["status"] = "TRAINING_COMPLETED_EVALUATION_RUNNING"
        progress["current_arm"] = None
        atomic_json(run_dir / "QUEUE_PROGRESS.json", progress)
        for task in manifest.get("post_tasks", []):
            if utc_now() >= deadline:
                progress.update({
                    "status": "BUDGET_CENSORED",
                    "reason": "queue deadline reached before post task",
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
            identity_path = run_dir / "workers" / f"{name}.identity.json"
            log_path = run_dir / "logs" / f"{name}.log"
            wrapper = [
                manifest["python"], "-u",
                str(Path(manifest["source_dir"]) /
                    "tools/hotel_a0_a1_task_worker.py"),
                "--identity", str(identity_path),
                "--cwd", manifest["source_dir"],
                *task["command"],
            ]
            with log_path.open("ab", buffering=0) as log:
                process = subprocess.Popen(
                    wrapper, cwd=manifest["source_dir"],
                    stdin=subprocess.DEVNULL, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True)
            observed_ticks = start_ticks(process.pid)
            progress.update({
                "current_task": name, "worker_pid": process.pid,
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
                        "reason": "queue deadline reached during post task",
                    })
                    atomic_json(run_dir / "QUEUE_PROGRESS.json", progress)
                    return 3
                time.sleep(15)
            task_receipt = {
                "name": name, "returncode": process.returncode,
                "finished_utc": utc_now().isoformat(), "log": str(log_path),
            }
            result_source = Path(task.get(
                "result_source", task.get("result", "")))
            result_target = Path(task.get("result", ""))
            if process.returncode == 0 and result_source.is_file():
                result_target.parent.mkdir(parents=True, exist_ok=True)
                if result_source.resolve() != result_target.resolve():
                    shutil.copy2(result_source, result_target)
                task_receipt.update({
                    "status": "COMPLETED", "result": str(result_target),
                    "result_sha256": sha256(result_target),
                })
            else:
                task_receipt["status"] = "CONTRACT_FAILED"
            atomic_json(run_dir / "receipts" / f"{name}.json", task_receipt)
            if task_receipt["status"] != "COMPLETED":
                progress.update({
                    "status": "CONTRACT_FAILED", "failed_task": name,
                    "task_receipt": task_receipt,
                })
                atomic_json(run_dir / "QUEUE_PROGRESS.json", progress)
                return 6
            progress.setdefault("completed_tasks", []).append(name)
            atomic_json(run_dir / "QUEUE_PROGRESS.json", progress)
        progress["status"] = "COMPLETED"
        progress["current_task"] = None
        progress["finished_utc"] = utc_now().isoformat()
        atomic_json(run_dir / "QUEUE_PROGRESS.json", progress)
        arm_receipts = {}
        for arm in manifest["arms"]:
            label = f'{arm["arm"]}_{arm["seed"]}'
            arm_receipts[label] = json.loads(
                (run_dir / "receipts" / f"{label}.json").read_text())
        result_payloads = {}
        for task in manifest.get("post_tasks", []):
            result = Path(task["result"])
            if result.is_file():
                result_payloads[task["name"]] = json.loads(result.read_text())
        atomic_json(run_dir / "RESULTS.json", {
            "status": "COMPLETED", "parent_sha256": manifest["parent_sha256"],
            "training_order": manifest["training_order"],
            "evaluation_seeds": manifest["evaluation_seeds"],
            "arm_receipts": arm_receipts, "evaluations": result_payloads,
            "limitations": [
                "HOTEL official validation and test bytes are identical",
                "three heads share one selected GDTS parent",
                "this is not an independent held-out or cross-scene result",
            ],
        })
        return 0
    finally:
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    sys.exit(main())
