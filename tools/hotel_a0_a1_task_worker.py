#!/usr/bin/env python3
"""Identity-recording exec wrapper for a frozen post-training task."""

import argparse
import datetime
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--identity", required=True)
    parser.add_argument("--cwd", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.command:
        raise ValueError("post-task command is empty")
    pid = os.getpid()
    identity = {
        "pid": pid, "ppid": os.getppid(), "pgid": os.getpgid(pid),
        "sid": os.getsid(pid),
        "proc_start_ticks": int(Path(f"/proc/{pid}/stat").read_text().split()[21]),
        "started_utc": datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        "exec_command": args.command, "cwd": str(Path(args.cwd).resolve()),
    }
    target = Path(args.identity)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(identity, indent=2, sort_keys=True))
    os.replace(temporary, target)
    os.chdir(args.cwd)
    os.execv(args.command[0], args.command)


if __name__ == "__main__":
    main()
