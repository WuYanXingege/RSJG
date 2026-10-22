#!/usr/bin/env python3
"""Summarize one or more small EBJD evaluation JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+")
    args = parser.parse_args()
    rows = []
    for name in args.results:
        path = Path(name)
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append({
            "file": str(path),
            "checkpoint_sha256": payload.get("checkpoint_sha256"),
            "mean": payload.get("mean"),
            "sample_standard_deviation": payload.get("sample_standard_deviation"),
        })
    print(json.dumps(rows, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
