#!/usr/bin/env python3
"""Export identity-bound native GDTS TTST20 endpoint proposals (no future input)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from ebjd.data import SceneManifestDataset


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def object_array(values) -> np.ndarray:
    answer = np.empty(len(values), dtype=object)
    for index, value in enumerate(values):
        answer[index] = value
    return answer


def save_shard(rows: list[dict], path: Path) -> None:
    np.savez_compressed(
        path,
        window_keys=np.asarray([x["window_key"] for x in rows]),
        scene_ids=np.asarray([x["scene_id"] for x in rows]),
        source_sequences=np.asarray([x["source_sequence"] for x in rows]),
        frame_ids=object_array([np.asarray(x["frame_ids"], dtype=np.int64) for x in rows]),
        agent_ids=object_array([np.asarray(x["agent_ids"]) for x in rows]),
        observed_sha256=np.asarray([x["observed_sha256"] for x in rows]),
        candidates=object_array([x["candidates"] for x in rows]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--gdts-root", required=True)
    parser.add_argument("--gdts-checkpoint", required=True)
    parser.add_argument("--expected-gdts-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-scenes", type=int, default=128)
    args = parser.parse_args()
    manifest = Path(args.manifest).resolve()
    checkpoint = Path(args.gdts_checkpoint).resolve()
    output = Path(args.output).resolve()
    if sha256(manifest) != args.expected_manifest_sha256:
        raise RuntimeError("source manifest SHA256 mismatch")
    if sha256(checkpoint) != args.expected_gdts_sha256:
        raise RuntimeError("GDTS checkpoint SHA256 mismatch")
    output.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("formal native goal export requires CUDA")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    gdts_root = Path(args.gdts_root).resolve()
    sys.path.insert(0, str(gdts_root))
    from src.data_src.dataset_src.dataset_create import create_dataset
    from src.models.model_utils.U_net_CNN import UNet
    from src.models.model_utils.cnn_big_images_utils import (
        create_CNN_inputs_loop, create_tensor_image,
    )
    from src.models.model_utils.sampling_2D_map import TTST_test_time_sampling_trick

    dataset = SceneManifestDataset(manifest)
    native_dataset = create_dataset("eth5", load_visual_data=True)
    source_to_scene = {
        "biwi_eth": "eth", "biwi_hotel": "hotel",
        "students001": "univ", "students003": "univ", "uni_examples": "univ",
        "crowds_zara01": "zara1", "crowds_zara02": "zara2",
        "crowds_zara03": "zara2",
    }
    semantics = {
        name: create_tensor_image(scene.semantic_map_pred, down_factor=8)
        for name, scene in native_dataset.scenes.items()
    }
    module = UNet(
        enc_chs=(14, 32, 32, 64, 64, 64),
        dec_chs=(64, 64, 64, 32, 32), out_chs=12)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload)
    module.load_state_dict({
        key[len("goal_module."):]: value for key, value in state.items()
        if key.startswith("goal_module.")}, strict=True)
    module = module.to(device).float().eval()
    started = time.monotonic()
    rows, files, calls = [], [], 0
    with torch.inference_mode():
        for index in range(len(dataset)):
            item = dataset[index]
            scene_name = source_to_scene[item["source_sequence"]]
            geometry = native_dataset.scenes[scene_name]
            semantic = semantics[scene_name]
            # The provider interface deliberately receives observed positions only.
            observed = item["observed"].transpose(0, 1)
            pixels = geometry.make_pixel_coord_torch(observed) / 8.0
            history = create_CNN_inputs_loop(pixels, semantic)
            image = semantic.unsqueeze(0).repeat(len(item["agent_ids"]), 1, 1, 1)
            provider_input = torch.cat((image, history), dim=1).to(device)
            probability = torch.sigmoid(module(provider_input)[:, -1:]).float()
            native = TTST_test_time_sampling_trick(
                probability, num_goals=20, device=device)
            if native.shape[0] != 21:
                raise RuntimeError("native TTST did not return 20 clusters plus argmax")
            clusters = native[:20, :, 0, :].float()
            world = torch.stack([
                geometry.make_world_coord_torch((value * 8.0).unsqueeze(0))[0]
                for value in clusters]).cpu().numpy().transpose(1, 0, 2)
            if world.shape != (len(item["agent_ids"]), 20, 2):
                raise RuntimeError("converted native goals have wrong shape")
            rows.append({
                "window_key": item["window_key"], "scene_id": item["scene_id"],
                "source_sequence": item["source_sequence"],
                "frame_ids": item["frame_ids"], "agent_ids": item["agent_ids"],
                "observed_sha256": hashlib.sha256(
                    item["observed"].contiguous().numpy().tobytes()).hexdigest(),
                "candidates": world.astype(np.float32),
            })
            calls += 1
            if len(rows) == args.shard_scenes or index + 1 == len(dataset):
                shard = output / f"native_goals_{len(files):04d}.npz"
                save_shard(rows, shard)
                files.append({
                    "path": shard.name, "sha256": sha256(shard),
                    "scenes": len(rows),
                })
                rows = []
            if (index + 1) % 100 == 0:
                print(f"native-goal export {index + 1}/{len(dataset)}", flush=True)
    result = {
        "schema": "ebjd-native-goal-bank-v1", "status": "COMPLETED",
        "unit": "m", "candidates": 20, "seed": args.seed,
        "scene_count": len(dataset),
        "agent_occurrence_count": int(dataset.manifest["statistics"]["agents"]),
        "source_manifest": str(manifest),
        "source_manifest_sha256": sha256(manifest),
        "provider_checkpoint": str(checkpoint),
        "provider_checkpoint_sha256": sha256(checkpoint),
        "provider_code_commit": subprocess.check_output(
            ["git", "-C", str(gdts_root), "rev-parse", "HEAD"], text=True).strip(),
        "provider_interface": "semantic_map + observed[8] history heatmaps only",
        "future_consumed_by_provider": False,
        "sampler": "native GDTS TTST; 20 clusters; appended argmax excluded",
        "precision": "FP32", "forward_calls": calls, "optimizer_updates": 0,
        "elapsed_seconds": time.monotonic() - started,
        "files": files,
    }
    atomic_json(result, output / "manifest.json")
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
