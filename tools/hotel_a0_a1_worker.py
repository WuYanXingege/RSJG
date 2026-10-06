#!/usr/bin/env python3
"""Identity-recording exec wrapper for one HOTEL A0/A1 arm."""

import argparse
import datetime
import json
import os
from pathlib import Path


def process_start_ticks(pid: int) -> int:
    return int(Path(f"/proc/{pid}/stat").read_text().split()[21])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--identity", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    pid = os.getpid()
    command = [
        str(Path(args.python).resolve()), "-u", "main.py",
        "--dataset", "eth5", "--test_set", "hotel",
        "--goal_model_type", "joint_dependency_v2",
        "--run_name", args.run_name, "--phase", "train",
    ]
    identity = {
        "pid": pid, "ppid": os.getppid(), "pgid": os.getpgid(pid),
        "sid": os.getsid(pid), "proc_start_ticks": process_start_ticks(pid),
        "started_utc": datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        "wrapper_cmdline": Path(f"/proc/{pid}/cmdline").read_bytes().replace(
            b"\0", b" ").decode(errors="replace").strip(),
        "exec_command": command, "source_dir": str(Path(args.source_dir).resolve()),
        "run_name": args.run_name,
    }
    target = Path(args.identity)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(identity, indent=2, sort_keys=True))
    os.replace(temporary, target)
    os.chdir(args.source_dir)
    os.execv(command[0], command)


if __name__ == "__main__":
    main()
