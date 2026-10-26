#!/usr/bin/env python3
"""Build native GDTS batches for exactly the frozen EBJD manifest windows.

The adapter changes only window enumeration.  Pixel conversion, global map
loading, trajectory-map rendering and serialized batch construction remain the
fixed legacy GDTS implementation.  This is needed for ETH source files whose
source-frame phases otherwise omit manifest-valid synchronized windows.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml


def manifest_windows(path: Path) -> tuple[dict[str, list[dict]], int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, list[dict]] = {}
    count = 0
    for descriptor in payload["shards"]:
        with np.load(path.parent / descriptor["path"], allow_pickle=True) as shard:
            for index in range(len(shard["scene_ids"])):
                source = str(shard["source_sequences"][index])
                result.setdefault(source, []).append({
                    "scene_id": str(shard["scene_ids"][index]),
                    "frame_ids": tuple(int(value) for value in shard["frame_ids"][index]),
                    "agent_ids": tuple(int(value) for value in shard["agent_ids"][index]),
                })
                count += 1
    if count != int(payload["statistics"]["scenes"]):
        raise ValueError("manifest scene count mismatch")
    return result, count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--batch-cache-root", required=True)
    args = parser.parse_args()
    recipe = yaml.safe_load(Path(args.recipe).read_text(encoding="utf-8"))
    legacy = Path(recipe["legacy_gdts"]["root"]).resolve()
    runtime = Path(args.runtime_root).resolve()
    runtime.mkdir(parents=True, exist_ok=True)
    data_link = runtime / "data"
    source_data = Path(recipe["source_data_root"]).resolve()
    if not data_link.exists() and not data_link.is_symlink():
        data_link.symlink_to(source_data, target_is_directory=True)
    if data_link.resolve() != source_data:
        raise RuntimeError("runtime data link target mismatch")
    targets, expected_count = manifest_windows(Path(args.manifest).resolve())
    sys.path.insert(0, str(legacy))
    os.chdir(runtime)

    from src.data_pre_process import Trajectory_Data_Pre_Process
    from src.parser import check_and_add_additional_args, get_parser

    original = Trajectory_Data_Pre_Process.make_scene_window_batches

    def manifest_bound(self, scene_data, set_name):
        source = Path(scene_data["file_path"]).stem
        if set_name != "train" or source not in targets:
            return original(self, scene_data, set_name)
        df = scene_data["raw_pixel_data"]
        by_key = df.set_index(["frame_id", "agent_id"], drop=False)
        for record in targets[source]:
            fragments = []
            for agent in record["agent_ids"]:
                rows = []
                for frame in record["frame_ids"]:
                    try:
                        row = by_key.loc[(frame, agent)]
                    except KeyError as error:
                        raise RuntimeError(
                            f"manifest identity absent from GDTS source: "
                            f"{source}, frame={frame}, agent={agent}") from error
                    if getattr(row, "ndim", 1) != 1:
                        raise RuntimeError("duplicate GDTS frame/agent source row")
                    rows.append(row)
                fragment = rows[0].to_frame().T
                if len(rows) > 1:
                    import pandas as pd
                    fragment = pd.DataFrame(rows).reset_index(drop=True)
                fragments.append(fragment)
            identity = {
                "scene_name": scene_data["scene_name"],
                "starting_frames": [record["frame_ids"][0]] * len(fragments),
                "frame_ids": list(record["frame_ids"]),
                "agent_ids": list(record["agent_ids"]),
                "data_file_path": scene_data["file_path"],
                "synchronized_window": True,
                "batch_format_version": 2,
                "manifest_scene_id": record["scene_id"],
            }
            trajectories = [
                fragment[["x_coord", "y_coord"]].to_numpy()
                for fragment in fragments
            ]
            self.massup_batch_and_save(
                trajectories, identity, set_name,
                frame_ids=np.asarray(record["frame_ids"], dtype=np.int64))

    Trajectory_Data_Pre_Process.make_scene_window_batches = manifest_bound
    cli = [
        "--dataset", "eth5", "--test_set", "zara1", "--phase", "pre-process",
        "--batch_size", "64", "--device", "cpu", "--num_workers", "0",
        "--down_factor", "8", "--skip_ts_window", "1",
        "--goal_model_type", "social", "--training_stage", "finetune",
        "--data_augmentation", "False", "--shuffle_train_batches", "False",
        "--shuffle_test_batches", "False", "--reproducibility", "True",
        "--seed", str(recipe["seed"]), "--use_wandb", "False",
        "--batch_cache_root", str(Path(args.batch_cache_root).resolve()),
    ]
    parsed = check_and_add_additional_args(get_parser().parse_args(cli))
    Trajectory_Data_Pre_Process(parsed)
    confirmation = (
        Path(args.batch_cache_root).resolve() / "eth5/zara1/data_batches_joint_v2"
        / "finished_train_batches.txt")
    if not confirmation.is_file() or str(expected_count) not in confirmation.read_text():
        raise RuntimeError("manifest-bound GDTS cache scene count mismatch")
    print(f"manifest-bound native GDTS cache complete: {expected_count} train scenes")


if __name__ == "__main__":
    main()
