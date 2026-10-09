#!/usr/bin/env python3
"""Preserve a new epoch-boundary checkpoint, then pause a serial queue.

The watcher never interrupts the active epoch.  It waits for ``last.pt`` to be
atomically replaced by a checkpoint at or beyond ``--minimum-epoch``, copies
and hashes that file, terminates the supervisor first (preventing advancement
to the next seed), and then terminates the current child.  It intentionally
does not send SIGKILL; failure to stop is recorded for explicit intervention.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import time
from datetime import datetime
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--minimum-epoch", required=True, type=int)
    parser.add_argument("--supervisor-pid-file", required=True)
    parser.add_argument("--child-pid-file", required=True)
    parser.add_argument("--preserve-directory", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--stop-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--supervisor-command-token", default="run_queue.sh")
    parser.add_argument("--child-command-token", default="instrumented_train.py")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def process_info(pid: int) -> dict:
    proc = Path("/proc") / str(pid)
    if not proc.exists():
        return {"pid": pid, "exists": False, "cmdline": None}
    try:
        command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(
            errors="replace").strip()
    except OSError:
        command = None
    return {"pid": pid, "exists": True, "cmdline": command}


def read_pid(path: Path) -> int:
    value = int(path.read_text().strip())
    if value <= 1:
        raise ValueError(f"unsafe PID in {path}: {value}")
    return value


def terminate(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint).resolve()
    receipt_path = Path(args.receipt).resolve()
    preserve = Path(args.preserve_directory).resolve()
    preserve.mkdir(parents=True, exist_ok=True)
    initial_stat = checkpoint.stat()
    state = {
        "schema": "ebjd-epoch-boundary-queue-pause-v1",
        "status": "WAITING_FOR_EPOCH_BOUNDARY",
        "started_at": datetime.now().astimezone().isoformat(),
        "checkpoint": str(checkpoint),
        "minimum_epoch": args.minimum_epoch,
        "initial_mtime_ns": initial_stat.st_mtime_ns,
        "initial_size": initial_stat.st_size,
    }
    atomic_json(receipt_path, state)
    while True:
        current = checkpoint.stat()
        if current.st_mtime_ns != initial_stat.st_mtime_ns:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if (payload.get("schema") == "ebjd-checkpoint-v2"
                    and payload.get("epoch_boundary_only") is True
                    and int(payload.get("epoch", -1)) >= args.minimum_epoch):
                break
        time.sleep(args.poll_seconds)

    epoch = int(payload["epoch"])
    update = int(payload["update_index"])
    source_hash = sha256(checkpoint)
    destination = preserve / (
        f"seed3101_epoch{epoch:03d}_update{update}_{source_hash}.pt")
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(checkpoint, temporary)
    copied_hash = sha256(temporary)
    if copied_hash != source_hash:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("preserved checkpoint SHA256 differs from source")
    os.replace(temporary, destination)

    supervisor = read_pid(Path(args.supervisor_pid_file))
    child = read_pid(Path(args.child_pid_file))
    before = {
        "supervisor": process_info(supervisor),
        "child": process_info(child),
    }
    for role, token in (
        ("supervisor", args.supervisor_command_token),
        ("child", args.child_command_token),
    ):
        info = before[role]
        if info["exists"] and token not in (info["cmdline"] or ""):
            state.update({
                "status": "PID_IDENTITY_MISMATCH",
                "completed_at": datetime.now().astimezone().isoformat(),
                "checkpoint_epoch": epoch,
                "checkpoint_update_index": update,
                "checkpoint_sha256": source_hash,
                "preserved_checkpoint": str(destination),
                "preserved_checkpoint_sha256": copied_hash,
                "processes_before_sigterm": before,
                "mismatched_role": role,
                "required_command_token": token,
            })
            atomic_json(receipt_path, state)
            raise RuntimeError(f"refusing to signal mismatched {role} PID: {info}")
    terminate(supervisor)
    terminate(child)
    deadline = time.monotonic() + args.stop_timeout_seconds
    while time.monotonic() < deadline and (alive(supervisor) or alive(child)):
        time.sleep(0.25)
    after = {
        "supervisor": process_info(supervisor),
        "child": process_info(child),
    }
    stopped = not after["supervisor"]["exists"] and not after["child"]["exists"]
    state.update({
        "status": "PAUSED_AT_EPOCH_BOUNDARY" if stopped else "STOP_TIMEOUT",
        "completed_at": datetime.now().astimezone().isoformat(),
        "checkpoint_epoch": epoch,
        "checkpoint_update_index": update,
        "checkpoint_sha256": source_hash,
        "preserved_checkpoint": str(destination),
        "preserved_checkpoint_sha256": copied_hash,
        "processes_before_sigterm": before,
        "processes_after_timeout": after,
    })
    atomic_json(receipt_path, state)
    if not stopped:
        raise RuntimeError(f"queue processes did not stop: {after}")


if __name__ == "__main__":
    main()
