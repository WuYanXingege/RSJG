#!/usr/bin/env python3
"""Wait for the formal SDD pipeline to finish, then run the EBJD benchmark."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def git(paper: Path, *arguments: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *arguments], cwd=paper, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    if check and result.returncode:
        raise RuntimeError(
            f"git {' '.join(arguments)} exited {result.returncode}: {result.stdout}")
    return result.stdout.strip()


def gpu_is_idle() -> bool:
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    return result.returncode == 0 and not result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", required=True)
    parser.add_argument("--formal-root", required=True)
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--branch", required=True)
    args = parser.parse_args()
    paper, formal, experiment = map(
        lambda value: Path(value).resolve(),
        (args.paper, args.formal_root, args.experiment_root))
    status_path = experiment / "BACKGROUND_STATUS.json"
    log_path = experiment / "BACKGROUND_BENCHMARK.log"
    state_path, lock_path = (
        formal / "PIPELINE_STATE.json", formal / "SUPERVISOR.lock")
    python = Path(
        "/media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/"
        ".conda/rsjg/bin/python")

    def status(state: str, **extra) -> None:
        atomic_json(status_path, {
            "schema": "ebjd-input-pipeline-background-v1",
            "status": state,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            **extra,
        })

    status("WAITING_FOR_FORMAL_PIPELINE", formal_state=str(state_path))
    lock_handle = None
    while True:
        try:
            formal_state = json.loads(state_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(60); continue
        if formal_state.get("status") != "COMPLETE":
            status(
                "WAITING_FOR_FORMAL_PIPELINE",
                formal_stage=formal_state.get("stage"),
                formal_status=formal_state.get("status"))
            time.sleep(60); continue
        candidate = lock_path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(candidate.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            candidate.close(); time.sleep(60); continue
        lock_handle = candidate
        break

    status("WAITING_FOR_GPU_IDLE", required_continuous_idle_seconds=120)
    idle_since = None
    while idle_since is None or time.monotonic() - idle_since < 120:
        if gpu_is_idle():
            idle_since = idle_since or time.monotonic()
        else:
            idle_since = None
        time.sleep(10)

    actual_head = git(paper, "rev-parse", "HEAD")
    if actual_head != args.expected_head:
        status("ABORTED_SOURCE_CHANGED", expected=args.expected_head,
               actual=actual_head)
        return
    if git(paper, "diff", "--name-only") or git(
            paper, "diff", "--cached", "--name-only"):
        status("ABORTED_TRACKED_WORKTREE_DIRTY")
        return

    status("RUNNING_BENCHMARK", source_commit=actual_head)
    command = [
        str(python), "scripts/benchmark_ebjd_input_pipeline.py",
        "--spec", str(experiment / "BENCHMARK_SPEC.json"),
        "--output", str(experiment), "--correctness-updates", "8",
        "--warmup-updates", "4", "--measured-updates", "32",
        "--repeats", "2",
    ]
    with log_path.open("a", encoding="utf-8") as handle:
        result = subprocess.run(
            command, cwd=paper, stdout=handle, stderr=subprocess.STDOUT,
            check=False)
    if result.returncode:
        status("FAILED_BENCHMARK", returncode=result.returncode,
               log=str(log_path))
        return

    asset_report = experiment / "reports"
    report_command = [
        str(python), "scripts/report_ebjd_input_pipeline.py",
        "--results", str(experiment / "BENCHMARK_RESULTS.json"),
        "--audit", str(experiment / "DATA_PATH_AUDIT.json"),
        "--output-dir", str(asset_report),
    ]
    subprocess.run(report_command, cwd=paper, check=True)
    docs = paper / "docs/performance/20261106_ebjd_input_pipeline_applicability"
    docs.mkdir(parents=True, exist_ok=True)
    for name in (
        "REPORT_ZH.md", "RESULTS.csv", "BENCHMARK_RESULTS.json",
        "DATA_PATH_AUDIT.json",
    ):
        shutil.copy2(asset_report / name, docs / name)
    shutil.copy2(experiment / "BENCHMARK_SPEC.json", docs / "BENCHMARK_SPEC.json")
    git(paper, "add", str(docs.relative_to(paper)))
    git(paper, "commit", "-m", "docs: report EBJD input pipeline benchmark")
    push_result = subprocess.run(
        ["git", "push", "origin", args.branch], cwd=paper, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    final_head = git(paper, "rev-parse", "HEAD")
    status(
        "COMPLETE" if push_result.returncode == 0 else "REPORT_PUSH_FAILED",
        benchmark_results=str(experiment / "BENCHMARK_RESULTS.json"),
        report=str(asset_report / "REPORT_ZH.md"),
        report_commit=final_head, push_output=push_result.stdout)
    # Keep the lock handle alive until all GPU and Git work is durable.
    assert lock_handle is not None


if __name__ == "__main__":
    main()
