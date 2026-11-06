#!/usr/bin/env python3
"""Replace the SDD supervisor only after matched-GDTS exits durably."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import torch


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


def cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    except FileNotFoundError:
        return ""


def process_state(pid: int) -> str | None:
    try:
        return Path(f"/proc/{pid}/stat").read_text().split()[2]
    except FileNotFoundError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--old-supervisor-pid", type=int, required=True)
    parser.add_argument("--old-trainer-pid", type=int, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--old-wait-command-fragment", required=True)
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args()
    paper, root = Path(args.paper).resolve(), Path(args.root).resolve()
    receipt = Path(args.receipt).resolve()
    supervisor_command = cmdline(args.old_supervisor_pid)
    trainer_command = cmdline(args.old_trainer_pid)
    if (str(root) not in supervisor_command or
            "run_sdd_unified_pipeline.py" not in supervisor_command):
        raise RuntimeError("old supervisor identity mismatch")
    if (str(root) not in trainer_command or
            "run_legacy_gdts_stage.py" not in trainer_command or
            "--stage baseline" not in trainer_command):
        raise RuntimeError("old matched-GDTS trainer identity mismatch")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=paper, check=True,
        text=True, capture_output=True).stdout.strip()
    if head != args.source_commit:
        raise RuntimeError(f"HEAD {head} != requested source {args.source_commit}")

    os.kill(args.old_supervisor_pid, signal.SIGSTOP)
    atomic_json(receipt, {
        "schema": "ebjd-sdd-supervisor-stage-handoff-v1",
        "status": "WAITING_FOR_MATCHED_GDTS_COMPLETION",
        "old_supervisor_pid": args.old_supervisor_pid,
        "matched_gdts_pid": args.old_trainer_pid,
        "source_commit": args.source_commit,
        "supervisor_stopped_only": True,
        "trainer_was_not_signalled": True,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    while process_state(args.old_trainer_pid) not in (None, "Z"):
        time.sleep(30)

    completion = root / "04_matched_gdts/MATCHED_GDTS_COMPLETION.json"
    if not completion.exists():
        os.kill(args.old_supervisor_pid, signal.SIGCONT)
        atomic_json(receipt, {
            "schema": "ebjd-sdd-supervisor-stage-handoff-v1",
            "status": "ABORTED_MATCHED_GDTS_NOT_COMPLETE",
            "old_supervisor_resumed": True,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        })
        return
    completion_payload = json.loads(completion.read_text(encoding="utf-8"))
    if completion_payload.get("stage_status") != "COMPLETED":
        os.kill(args.old_supervisor_pid, signal.SIGCONT)
        raise RuntimeError("matched-GDTS completion receipt is not COMPLETED")

    last = Path(completion_payload["last_checkpoint"])
    last_payload = torch.load(last, map_location="cpu", weights_only=False)
    persisted_updates = int(last_payload["epoch"]) * 2544
    os.kill(args.old_supervisor_pid, signal.SIGTERM)
    os.kill(args.old_supervisor_pid, signal.SIGCONT)
    deadline = time.monotonic() + 30
    while process_state(args.old_supervisor_pid) not in (None, "Z"):
        if time.monotonic() >= deadline:
            os.kill(args.old_supervisor_pid, signal.SIGKILL)
            break
        time.sleep(0.2)

    python = Path(
        "/media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/"
        ".conda/rsjg/bin/python")
    supervisor_log = root / "logs/supervisor.log"
    with supervisor_log.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(
            [str(python), "-u", "scripts/run_sdd_unified_pipeline.py",
             "--root", str(root), "--source-commit", args.source_commit,
             "--migration-reason",
             "fivefold EBJD input runtime acceptance before SDD EBJD",
             "--migration-optimizer-updates", str(persisted_updates)],
            cwd=paper, stdout=handle, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True)
    time.sleep(3)
    if process.poll() is not None:
        raise RuntimeError(f"replacement supervisor exited {process.returncode}")

    cancelled = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == os.getpid():
            continue
        command = cmdline(pid)
        if args.old_wait_command_fragment in command:
            os.kill(pid, signal.SIGTERM)
            cancelled.append(pid)
    atomic_json(receipt, {
        "schema": "ebjd-sdd-supervisor-stage-handoff-v1",
        "status": "COMPLETE",
        "matched_gdts_completion": str(completion),
        "matched_gdts_completed_epochs": completion_payload["completed_epochs"],
        "persisted_optimizer_updates": persisted_updates,
        "old_supervisor_pid": args.old_supervisor_pid,
        "new_supervisor_pid": process.pid,
        "cancelled_old_wait_pids": cancelled,
        "source_commit": args.source_commit,
        "next_order": [
            "MATCHED_GDTS_EVAL", "PROPOSAL_BANKS",
            "EBJD_INPUT_PIPELINE_ACCEPTANCE", "EBJD_UNIFIED_TRAIN_EVAL"],
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    })


if __name__ == "__main__":
    main()
