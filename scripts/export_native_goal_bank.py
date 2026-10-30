#!/usr/bin/env python3
"""Export a complete proposal bank using only the standalone Goal provider."""

from __future__ import annotations

import argparse
import json
import pickle
import random
import time
from pathlib import Path

import numpy as np
import torch

from ebjd.export_ethucy import pixel_to_world
from ebjd.goal_provider import NativeGoalProvider, file_sha256
from export_zara1_endpoint_banks import manifest_identity, pack_bank, save_bank


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider-manifest", required=True)
    parser.add_argument("--data-manifest", required=True)
    parser.add_argument("--data-manifest-sha256", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--homography", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    manifest_path = Path(args.data_manifest).resolve()
    if file_sha256(manifest_path) != args.data_manifest_sha256:
        raise ValueError("data manifest SHA256 mismatch")
    records, manifest = manifest_identity(manifest_path)
    lookup = {}
    for record in records:
        key = (record["scene_id"].split(":")[2], tuple(record["frame_ids"]),
               tuple(record["agent_ids"]))
        lookup[key] = record
    provider = NativeGoalProvider.from_artifact(args.provider_manifest)
    device = torch.device(args.device); provider.to(device).eval()
    homography = np.loadtxt(args.homography, dtype=np.float64)
    folder = "train_batches" if args.split == "train" else "valid_batches"
    files = sorted((Path(args.cache_root).resolve() / folder).glob("*.pkl"))
    if not files:
        raise FileNotFoundError(f"no native cache files in {folder}")
    proposals = {}; random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    if device.type == "cuda": torch.cuda.synchronize(device)
    started = time.perf_counter()
    for path in files:
        with path.open("rb") as handle:
            data, identity = pickle.load(handle)
        frames = tuple(int(value) for value in identity.get("frame_ids", []))
        agents = tuple(str(int(value)) for value in identity.get("agent_ids", []))
        source = Path(str(identity["data_file_path"])).stem
        record = lookup.get((source, frames, agents))
        if record is None:
            # A cache may also contain the opposite split; ignore only identities
            # that are provably absent from this frozen manifest.
            continue
        semantic = torch.as_tensor(data["tensor_image"], device=device).float()
        observed_maps = torch.as_tensor(data["input_traj_maps"], device=device).float()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            _, goal_map = provider.predict_prepared(semantic, observed_maps)
        full_pixel = (goal_map.float() * 8.0).cpu().numpy()
        world = pixel_to_world(
            full_pixel.reshape(-1, 2), homography, args.scene).reshape(full_pixel.shape)
        proposals[record["scene_id"]] = world.astype(np.float32)
    if device.type == "cuda": torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    if len(proposals) != len(records):
        missing = [record["scene_id"] for record in records
                   if record["scene_id"] not in proposals]
        raise RuntimeError(
            f"standalone provider covered {len(proposals)}/{len(records)} scenes; "
            f"first missing={missing[:3]}")
    arrays = pack_bank(records, proposals)
    provider_manifest = json.loads(Path(args.provider_manifest).read_text())
    metadata = {
        "schema": "ebjd-standalone-native-proposal-bank-v1",
        "fold": provider_manifest["fold"], "split": args.split,
        "seed": args.seed, "worlds": 20, "coordinate_unit": "m",
        "checkpoint_sha256": provider_manifest["source_checkpoint_sha256"],
        "provider_weights_sha256": provider_manifest["weights_sha256"],
        "manifest_sha256": args.data_manifest_sha256,
        "scene_count": len(records),
        "agent_occurrence_count": int(manifest["statistics"]["agents"]),
        "endpoint_source": "standalone_matched_goal_unet_native_ttst",
        "full_gdts_trajectory_network_instantiated": False,
        "gt_used_for_generation": False, "gpu_elapsed_seconds": elapsed,
    }
    receipt = save_bank(Path(args.output).resolve(), arrays, metadata)
    print(json.dumps({"status": "COMPLETE", "receipt": receipt}, sort_keys=True))


if __name__ == "__main__":
    main()
