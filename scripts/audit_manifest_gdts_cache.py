#!/usr/bin/env python3
"""CPU identity/coordinate audit for a manifest-bound native GDTS cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    recipe = yaml.safe_load(Path(args.recipe).read_text(encoding="utf-8"))
    legacy = Path(recipe["legacy_gdts"]["root"]).resolve()
    runtime = Path(args.runtime_root).resolve()
    sys.path.insert(0, str(legacy))
    os.chdir(runtime)
    from src.data_src.dataset_src.dataset_eth5 import Dataset_eth5

    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = {}
    for descriptor in manifest["shards"]:
        shard_path = manifest_path.parent / descriptor["path"]
        if sha256(shard_path) != descriptor["sha256"]:
            raise ValueError(f"manifest shard hash mismatch: {shard_path}")
        with np.load(shard_path, allow_pickle=True) as shard:
            for index in range(len(shard["scene_ids"])):
                key = (
                    str(shard["source_sequences"][index]),
                    tuple(int(value) for value in shard["frame_ids"][index]),
                    tuple(int(value) for value in shard["agent_ids"][index]),
                )
                records[key] = np.concatenate((
                    np.asarray(shard["observed"][index]),
                    np.asarray(shard["future"][index]),
                ), axis=1)
    dataset = Dataset_eth5()
    cache_files = sorted(Path(args.cache_dir).resolve().glob("*.pkl"))
    seen = set()
    maximum_error = 0.0
    source_counts: dict[str, int] = {}
    for path in cache_files:
        with path.open("rb") as handle:
            data, identity = pickle.load(handle)  # trusted local asset
        key = (
            Path(identity["data_file_path"]).stem,
            tuple(int(value) for value in identity["frame_ids"]),
            tuple(int(value) for value in identity["agent_ids"]),
        )
        expected = records.get(key)
        if expected is None:
            raise RuntimeError(f"cache identity absent from manifest: {path}")
        if key in seen:
            raise RuntimeError(f"duplicate cache identity: {path}")
        seen.add(key)
        pixel = torch.as_tensor(data["abs_pixel_coord"], dtype=torch.float64)
        world = dataset.scenes[identity["scene_name"]].make_world_coord_torch(
            pixel).permute(1, 0, 2).cpu().numpy()
        maximum_error = max(maximum_error, float(np.abs(world - expected).max()))
        source = Path(identity["data_file_path"]).stem
        source_counts[source] = source_counts.get(source, 0) + 1
    if seen != set(records):
        raise RuntimeError(f"cache missing {len(set(records) - seen)} manifest windows")
    if maximum_error >= 1e-5:
        raise RuntimeError(f"native cache coordinate parity failed: {maximum_error}")
    atomic_json(Path(args.output), {
        "status": "PASS", "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "cache_directory": str(Path(args.cache_dir).resolve()),
        "scene_count": len(seen), "source_counts": source_counts,
        "maximum_world_absolute_error_m": maximum_error,
    })
    print(f"PASS: {len(seen)} scenes, max_abs={maximum_error:.3g} m")


if __name__ == "__main__":
    main()
