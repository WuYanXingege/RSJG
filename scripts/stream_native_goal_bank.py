#!/usr/bin/env python3
"""Stream legacy-exact preprocessing into a compact standalone-provider bank.

No large GDTS batch cache and no GDTS trajectory model are instantiated.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from ebjd.goal_provider import NativeGoalProvider, file_sha256
from export_zara1_endpoint_banks import manifest_identity, pack_bank, save_bank


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--provider-manifest", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    manifest_path = Path(args.manifest).resolve()
    if file_sha256(manifest_path) != args.manifest_sha256:
        raise ValueError("data manifest SHA256 mismatch")
    records, manifest = manifest_identity(manifest_path)
    targets: dict[str, list[dict]] = {}
    for record in records:
        targets.setdefault(record["scene_id"].split(":")[2], []).append(record)
    recipe = yaml.safe_load(Path(args.recipe).read_text())
    legacy = Path(recipe["legacy_gdts"]["root"]).resolve()
    runtime = Path(args.runtime_root).resolve(); runtime.mkdir(parents=True, exist_ok=True)
    data_link = runtime / "data"; source_data = Path(recipe["source_data_root"]).resolve()
    if not data_link.exists() and not data_link.is_symlink():
        data_link.symlink_to(source_data, target_is_directory=True)
    if data_link.resolve() != source_data:
        raise RuntimeError("runtime data link target mismatch")
    sys.path.insert(0, str(legacy)); os.chdir(runtime)
    from src.data_pre_process import Trajectory_Data_Pre_Process
    from src.parser import check_and_add_additional_args, get_parser

    provider = NativeGoalProvider.from_artifact(args.provider_manifest)
    device = torch.device(args.device); provider.to(device).eval()
    proposals: dict[str, np.ndarray] = {}
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    original_add = Trajectory_Data_Pre_Process.add_pre_computed_cnn_maps

    def stream_scene(self, scene_data, set_name):
        source = Path(scene_data["file_path"]).stem
        if source not in targets:
            return
        dataframe = scene_data["raw_pixel_data"]
        by_key = dataframe.set_index(["frame_id", "agent_id"], drop=False)
        for record in targets[source]:
            trajectories = []
            for agent in record["agent_ids"]:
                rows = [by_key.loc[(frame, int(agent))] for frame in record["frame_ids"]]
                trajectories.append(np.stack([
                    np.asarray(row[["x_coord", "y_coord"]], dtype=np.float32)
                    for row in rows]))
            absolute = np.stack(trajectories).transpose(1, 0, 2)
            data = {"abs_pixel_coord": absolute,
                    "seq_list": np.ones(absolute.shape[:2], dtype=np.float32)}
            identity = {"scene_name": scene_data["scene_name"],
                        "data_file_path": scene_data["file_path"]}
            data = original_add(self, data, identity)
            semantic = data["tensor_image"].to(device).float()
            observed_maps = data["input_traj_maps"].to(device).float()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                _, goal_map = provider.predict_prepared(semantic, observed_maps)
            scene = self.dataset.scenes[scene_data["scene_name"]]
            world = scene.make_world_coord_torch(goal_map.float() * 8.0)
            proposals[record["scene_id"]] = world.cpu().numpy().astype(np.float32)

    Trajectory_Data_Pre_Process.make_scene_window_batches = stream_scene
    # The legacy preprocessor writes completion sentinels for all three splits.
    # Those sentinels are intentionally disposable here: this exporter streams
    # tensors into the compact bank and does not materialize batch-cache files.
    # A fresh scratch root is therefore required for every split/seed replay;
    # reusing one would skip preprocessing and silently yield an empty bank.
    with tempfile.TemporaryDirectory(
            prefix=f"stream_{args.split}_{args.seed}_", dir=runtime) as scratch:
        cli = ["--dataset", "eth5", "--test_set", str(recipe["fold"]),
               "--phase", "pre-process", "--batch_size", "64", "--device", "cpu",
               "--num_workers", "0", "--down_factor", "8", "--skip_ts_window", "1",
               "--goal_model_type", "social", "--training_stage", "finetune",
               "--data_augmentation", "False", "--shuffle_train_batches", "False",
               "--shuffle_test_batches", "False", "--reproducibility", "True",
               "--seed", str(recipe["seed"]), "--use_wandb", "False",
               "--batch_cache_root", scratch]
        parsed = check_and_add_additional_args(get_parser().parse_args(cli))
        if device.type == "cuda": torch.cuda.synchronize(device)
        started = time.perf_counter(); Trajectory_Data_Pre_Process(parsed)
        if device.type == "cuda": torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
    if len(proposals) != len(records):
        missing = [record["scene_id"] for record in records
                   if record["scene_id"] not in proposals]
        raise RuntimeError(f"stream covered {len(proposals)}/{len(records)}; {missing[:3]}")
    provider_manifest = json.loads(Path(args.provider_manifest).read_text())
    metadata = {"schema": "ebjd-standalone-native-proposal-bank-v1",
        "fold": recipe["fold"], "split": args.split, "seed": args.seed,
        "worlds": 20, "coordinate_unit": "m",
        "checkpoint_sha256": provider_manifest["source_checkpoint_sha256"],
        "provider_weights_sha256": provider_manifest["weights_sha256"],
        "manifest_sha256": args.manifest_sha256, "scene_count": len(records),
        "agent_occurrence_count": int(manifest["statistics"]["agents"]),
        "endpoint_source": "standalone_matched_goal_unet_native_ttst",
        "full_gdts_trajectory_network_instantiated": False,
        "large_intermediate_cache_materialized": False,
        "gt_used_for_generation": False, "gpu_elapsed_seconds": elapsed}
    receipt = save_bank(Path(args.output).resolve(), pack_bank(records, proposals), metadata)
    print(json.dumps({"status": "COMPLETE", "receipt": receipt}, sort_keys=True))


if __name__ == "__main__": main()
