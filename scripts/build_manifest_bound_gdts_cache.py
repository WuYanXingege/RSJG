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

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

import numpy as np
import yaml


def manifest_windows(path: Path) -> tuple[dict[str, list[dict]], int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    source_paths = {
        str(item["sequence"]): str(Path(item["path"]).resolve())
        for item in payload["source_files"]
    }
    result: dict[str, list[dict]] = {}
    count = 0
    for descriptor in payload["shards"]:
        with np.load(path.parent / descriptor["path"], allow_pickle=True) as shard:
            for index in range(len(shard["scene_ids"])):
                source = str(shard["source_sequences"][index])
                scale = float(shard["scale_m_per_px"][index])
                if not np.isfinite(scale) or scale <= 0:
                    raise ValueError("invalid SDD pixel-to-meter scale")
                metric = np.concatenate((
                    np.asarray(shard["observed"][index], dtype=np.float64),
                    np.asarray(shard["future"][index], dtype=np.float64),
                ), axis=1)
                result.setdefault(source, []).append({
                    "scene_id": str(shard["scene_ids"][index]),
                    "frame_ids": tuple(int(value) for value in shard["frame_ids"][index]),
                    "agent_ids": tuple(str(value) for value in shard["agent_ids"][index]),
                    "pixel_trajectories": metric / scale,
                    "source_path": source_paths[source],
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
    parser.add_argument(
        "--split", choices=("train", "validation", "test"), default="train")
    args = parser.parse_args()
    os.nice(10)
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

    import torch
    from src.data_src import data_utils
    from src.data_src.dataset_src.dataset_base import Dataset_base
    from src.data_src.dataset_src.dataset_sdd import Dataset_sdd
    from src.data_src.experiment_src.experiment_sdd import Experiment_sdd
    from src.data_src.scene_src.scene_sdd import Scene_sdd
    from src.data_pre_process import Trajectory_Data_Pre_Process
    from src.parser import check_and_add_additional_args, get_parser

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    data_utils.get_data_location = lambda: str(source_data)

    if str(recipe.get("legacy_dataset", "eth5")) == "sdd":
        selected_scenes = set(targets)

        def manifest_only_dataset_init(
                self, verbose=False, load_visual_data=True):
            Dataset_base.__init__(self)
            self.name = "sdd"
            self.dataset_folder = os.path.join(
                self.path_to_root, "data", self.name)
            if args.split == "train":
                self.train_scenes = set(selected_scenes)
                self.test_scenes = set()
            else:
                self.train_scenes = set()
                self.test_scenes = set(selected_scenes)
            self.scenes = {
                key: Scene_sdd(
                    key, verbose, load_visual_data=load_visual_data)
                for key in selected_scenes
            }

        Dataset_sdd.__init__ = manifest_only_dataset_init

        def manifest_only_data_files(self, set_name):
            return [{
                "file_path": targets[scene_name][0]["source_path"],
                "scene_name": scene_name,
                "downsample_frame_rate": self.downsample_frame_rate,
                "set_name": set_name,
                # The frozen manifest already owns trajectory enumeration and
                # coordinates. Avoid parsing multi-million-row bbox tables.
                "raw_pixel_data": None,
            } for scene_name in sorted(self.set_name_to_scenes[set_name])]

        Experiment_sdd._load_data_files = manifest_only_data_files

    original = Trajectory_Data_Pre_Process.make_scene_window_batches

    target_set_name = {"train": "train", "validation": "valid", "test": "test"}[
        args.split]

    def manifest_bound(self, scene_data, set_name):
        source = str(scene_data["scene_name"])
        if set_name != target_set_name or source not in targets:
            return original(self, scene_data, set_name)
        for record in targets[source]:
            identity = {
                "scene_name": scene_data["scene_name"],
                "starting_frames": (
                    [record["frame_ids"][0]] * len(record["agent_ids"])),
                "frame_ids": list(record["frame_ids"]),
                "agent_ids": list(record["agent_ids"]),
                # SDD calls every source file annotations.txt. Store the
                # sequence identity explicitly so downstream manifest joins
                # cannot collapse all scenes to the stem "annotations".
                "data_file_path": source,
                "synchronized_window": True,
                "batch_format_version": 2,
                "manifest_scene_id": record["scene_id"],
            }
            trajectories = [
                np.asarray(value, dtype=np.float64)
                for value in record["pixel_trajectories"]
            ]
            self.massup_batch_and_save(
                trajectories, identity, set_name,
                frame_ids=np.asarray(record["frame_ids"], dtype=np.int64))

    Trajectory_Data_Pre_Process.make_scene_window_batches = manifest_bound
    original_create_data_batches = Trajectory_Data_Pre_Process.create_data_batches

    def selected_split_only(self, set_name):
        if set_name == target_set_name:
            return original_create_data_batches(self, set_name)
        # Leave unrelated splits incomplete. A later process handles the next
        # split after the current process exits and releases all full-resolution
        # SDD maps. This bounds peak memory without invalid completion markers.
        self.num_batches[set_name] = 0

    Trajectory_Data_Pre_Process.create_data_batches = selected_split_only
    dataset = str(recipe.get("legacy_dataset", "eth5"))
    test_set = str(recipe["fold"]) if dataset == "eth5" else dataset
    cli = [
        "--dataset", dataset, "--test_set", test_set,
        "--phase", "pre-process",
        "--batch_size", "64", "--device", "cpu", "--num_workers", "0",
        "--down_factor", "8", "--skip_ts_window", "1",
        "--goal_model_type", "social", "--training_stage", "finetune",
        "--data_augmentation", "False", "--shuffle_train_batches", "False",
        "--shuffle_test_batches", "False", "--reproducibility", "True",
        "--seed", str(recipe["seed"]), "--use_wandb", "False",
        "--compress_batch_cache", "True",
        "--batch_cache_root", str(Path(args.batch_cache_root).resolve()),
    ]
    parsed = check_and_add_additional_args(get_parser().parse_args(cli))
    Trajectory_Data_Pre_Process(parsed)
    confirmation = (
        Path(args.batch_cache_root).resolve()
        / f"{dataset}/{test_set}/data_batches_joint_v2_zstd_v1"
        / f"finished_{target_set_name}_batches.txt")
    if not confirmation.is_file() or str(expected_count) not in confirmation.read_text():
        raise RuntimeError("manifest-bound GDTS cache scene count mismatch")
    print(
        "manifest-bound native GDTS cache complete: "
        f"split={args.split}, scenes={expected_count}")


if __name__ == "__main__":
    main()
