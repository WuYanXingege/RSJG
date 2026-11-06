#!/usr/bin/env python3
"""Pause one SDD supervisor immediately after an atomic epoch checkpoint.

This is intentionally scoped to an explicit supervisor/trainer pair and an
explicit checkpoint.  It is used only for execution-code migrations where an
epoch-boundary resume is required.
"""

import argparse
import json
import os
import signal
import tempfile
import time
from pathlib import Path

import torch


def atomic_json(payload, destination):
    destination = Path(destination)
    descriptor, temporary = tempfile.mkstemp(
        prefix=destination.name + ".", suffix=".tmp",
        dir=destination.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--supervisor-pid", required=True, type=int)
    parser.add_argument("--trainer-pid", required=True, type=int)
    parser.add_argument("--after-epoch", required=True, type=int)
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint).resolve()
    receipt = Path(args.receipt).resolve()
    initial = checkpoint.stat().st_mtime_ns
    while True:
        if checkpoint.stat().st_mtime_ns != initial:
            os.kill(args.trainer_pid, signal.SIGSTOP)
            payload = torch.load(
                checkpoint, map_location="cpu", weights_only=False)
            epoch = int(payload["epoch"])
            if epoch <= args.after_epoch:
                os.kill(args.trainer_pid, signal.SIGCONT)
                initial = checkpoint.stat().st_mtime_ns
                continue
            atomic_json({
                "status": "PAUSED_AT_COMPLETE_CHECKPOINT",
                "checkpoint": str(checkpoint),
                "epoch": epoch,
                "optimizer_updates": int(epoch * 2544),
                "supervisor_pid": args.supervisor_pid,
                "trainer_pid": args.trainer_pid,
                "observed_at_unix": time.time(),
            }, receipt)
            try:
                os.kill(args.supervisor_pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            os.kill(args.trainer_pid, signal.SIGTERM)
            os.kill(args.trainer_pid, signal.SIGCONT)
            return
        time.sleep(0.01)


if __name__ == "__main__":
    main()
