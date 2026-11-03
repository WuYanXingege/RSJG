#!/usr/bin/env python3
"""Readiness-gated entry for the SDD unified coupling-bridge experiment."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-status", required=True)
    parser.add_argument("--resolved-config", required=True)
    parser.add_argument("--train-bank", required=True)
    parser.add_argument("--validation-bank-pattern", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    status = json.loads(Path(args.asset_status).read_text())
    if status.get("readiness") != "TRAIN_READY":
        missing = status.get("missing", [])
        raise SystemExit(
            "SDD formal launch refused: readiness is "
            f"{status.get('readiness')}; missing={missing}")
    required = [args.resolved_config, args.train_bank]
    required.extend(args.validation_bank_pattern.format(seed=seed)
                    for seed in range(2035, 2040))
    absent = [path for path in required if not Path(path).is_file()]
    if absent:
        raise SystemExit(f"SDD formal launch refused: absent={absent}")
    command = [
        sys.executable, str(Path(__file__).with_name(
            "run_zara1_unified_coupling_bridge.py")),
        "--config", args.resolved_config,
        "--train-bank", args.train_bank,
        "--validation-bank-pattern", args.validation_bank_pattern,
        "--output", args.output, "--source-commit", args.source_commit,
        "--fold", "sdd_trajnet_jmm", "--training-seed", "3101",
        "--provider-seed", "3101", "--compact-scenes",
    ]
    print(json.dumps({"ready": True, "command": command}))
    if args.launch:
        os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
