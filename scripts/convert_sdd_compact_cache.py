#!/usr/bin/env python3
"""Stream-convert legacy SDD batches without dense trajectory heatmaps."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import subprocess
from pathlib import Path

from ebjd.compact_sdd_heatmap_pipeline import SCHEMA
from ebjd.legacy_sdd_input_pipeline import load_batch_cache_inprocess


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def dump_zstd(value, output):
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as handle:
        process = subprocess.Popen(
            ["zstd", "-1", "-q", "-c"], stdin=subprocess.PIPE,
            stdout=handle)
        pickle.dump(value, process.stdin, protocol=pickle.HIGHEST_PROTOCOL)
        process.stdin.close()
        code = process.wait()
    if code:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"zstd failed for {output}")
    os.replace(temporary, output)


def convert_split(source, target):
    target.mkdir(parents=True, exist_ok=True)
    files = sorted(source.glob("*.pkl.zst"))
    total_source = total_output = 0
    identity = hashlib.sha256()
    for position, path in enumerate(files, 1):
        output = target / path.name
        if not output.is_file():
            batch, batch_id = load_batch_cache_inprocess(path)
            maps = batch.pop("input_traj_maps")
            batch["_paper_compact_schema"] = SCHEMA
            batch["_paper_compact_original_hw"] = tuple(maps.shape[-2:])
            dump_zstd((batch, batch_id), output)
        total_source += path.stat().st_size
        total_output += output.stat().st_size
        identity.update(path.name.encode())
        if position % 100 == 0 or position == len(files):
            print(f"{source.name}: {position}/{len(files)}", flush=True)
    return {
        "files": len(files), "source_bytes": total_source,
        "compact_bytes": total_output,
        "ordered_filename_sha256": identity.hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    receipt = {"schema": SCHEMA, "splits": {}}
    for split in ("train", "valid", "test"):
        receipt["splits"][split] = convert_split(
            args.source_root / f"{split}_batches",
            args.output_root / f"{split}_batches")
    atomic_json(args.output_root / "COMPACT_CACHE_RECEIPT.json", receipt)


if __name__ == "__main__":
    main()
