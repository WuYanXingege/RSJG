#!/usr/bin/env python3
"""Lightweight identity audit for the already-built SDD GDTS train cache."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

from ebjd.legacy_sdd_input_pipeline import load_batch_cache_inprocess


def atomic_json(payload: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=destination.name + ".", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.train_cache).resolve()
    paths = sorted(root.glob("*.pkl.zst"))
    trajectory_windows = Counter()
    unique_agents = set()
    unique_starts = set()
    raw_scenes = set()
    total_valid_slots = 0
    maximum_agents = 0
    for path in paths:
        batch, identity = load_batch_cache_inprocess(path)
        agents = [str(value) for value in identity["agent_ids"]]
        starts = [int(value) for value in identity["starting_frames"]]
        count = int(batch["abs_pixel_coord"].shape[1])
        if not (len(agents) == len(starts) == count):
            raise RuntimeError(f"batch identity length mismatch: {path}")
        scene = str(identity["scene_name"])
        raw_scenes.add(scene)
        total_valid_slots += count
        maximum_agents = max(maximum_agents, count)
        for agent, start in zip(agents, starts):
            key = (scene, start, agent)
            trajectory_windows[key] += 1
            unique_agents.add((scene, agent))
            unique_starts.add((scene, start))
    duplicates = {
        "|".join(map(str, key)): value
        for key, value in trajectory_windows.items() if value != 1}
    atomic_json({
        "schema": "ebjd-sdd-gdts-cache-identity-audit-v1",
        "cache_root": str(root),
        "physical_batches_optimizer_updates_per_epoch": len(paths),
        "unique_trajectory_windows": len(trajectory_windows),
        "unique_scene_agent_identities": len(unique_agents),
        "unique_scene_start_frames": len(unique_starts),
        "total_valid_agent_slots": total_valid_slots,
        "raw_scene_count": len(raw_scenes),
        "raw_scenes": sorted(raw_scenes),
        "maximum_agents_per_window": maximum_agents,
        "nominal_capacity_at_batch_size_64": len(paths) * 64,
        "duplicate_trajectory_window_ids": duplicates,
        "interpretation": (
            "one legacy independent-cache file is one optimizer update that "
            "packs up to 64 legal per-agent sliding windows; 162816 packed "
            "slots and 8940 synchronized-manifest agent occurrences are "
            "different counting units"),
    }, Path(args.output).resolve())


if __name__ == "__main__":
    main()
