#!/usr/bin/env python3
"""Read-only adapter exporting the frozen native GDTS HOTEL inference."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

# ``ebjd`` is imported before the GDTS worktree is prepended.  This keeps the
# portable bank format in one implementation while every GDTS ``src`` import
# below resolves to the explicitly supplied frozen worktree.
from ebjd.diagnostic_bank import (
    BANK_SCHEMA, SEED_MANIFEST_SCHEMA, BankScene, atomic_json,
    canonical_source, canonical_window_key, save_scene, sha256_file,
    summarize_scenes,
)


EXPECTED_CHECKPOINT_SHA256 = (
    "5c101c2474ebb1a3cb3ecf882f0e9db2fbb2489c741c58068b8183b0b663b97b")


def _flatten_collated(values) -> list[int]:
    result = []
    for value in values:
        tensor = torch.as_tensor(value).reshape(-1)
        if tensor.numel() != 1:
            raise ValueError("collated identity entry is not scalar")
        result.append(int(tensor.item()))
    return result


def complete_seed(path: Path, expected_scenes: int) -> bool:
    manifest = path / "manifest.json"
    if not manifest.is_file():
        return False
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        return (
            payload.get("status") == "COMPLETED"
            and int(payload.get("scene_count", -1)) == expected_scenes
            and all(
                (manifest.parent / item["path"]).is_file()
                and sha256_file(manifest.parent / item["path"]) == item["sha256"]
                for item in payload.get("files", [])))
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export native frozen GDTS HOTEL P20 trajectories")
    parser.add_argument("--gdts-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[2035, 2036, 2037, 2038, 2039])
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()

    gdts_root = Path(args.gdts_root).resolve()
    sys.path.insert(0, str(gdts_root))
    from src.data_loader import get_dataloader
    from src.joint_dependency_v2_cache import sha256_file as gdts_sha256
    from src.metrics import compute_metric_mask
    from src.models.model import GDTS
    from src.utils import isolated_random_seed

    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    if gdts_sha256(checkpoint_path) != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("frozen GDTS checkpoint SHA256 mismatch")
    fields = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    loader_args = SimpleNamespace(**fields)
    loader_args.phase = "test"
    loader_args.shuffle_test_batches = False
    loader_args.jdv2_active = True
    loader = get_dataloader(loader_args, set_name="test")
    scene_limit = len(loader) if args.max_scenes is None else min(
        len(loader), int(args.max_scenes))
    if scene_limit < 1:
        raise ValueError("--max-scenes must be positive")

    model_fields = dict(fields)
    model_fields.update({
        "goal_model_type": "independent", "jdv2_active": False,
        "use_scene_latent": False, "use_dynamic_relation": False,
        "use_joint_energy": False, "use_dependency_corrector": False,
        "training_stage": "baseline", "trajectory_coupling": "none",
        "trajectory_alignment": False,
    })
    model_args = SimpleNamespace(**model_fields)
    device = torch.device(model_args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the frozen GDTS baseline adapter requires CUDA")
    model = GDTS(model_args, device).to(device).eval()
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    output_root = Path(args.output_root).resolve() / "gdts"
    output_root.mkdir(parents=True, exist_ok=True)
    seed_manifests = []

    with torch.no_grad():
        for seed in args.seeds:
            seed_dir = output_root / f"seed_{seed}"
            if args.resume and complete_seed(seed_dir, scene_limit):
                print(f"GDTS seed {seed}: verified existing bank", flush=True)
                seed_manifests.append(seed_dir / "manifest.json")
                continue
            seed_dir.mkdir(parents=True, exist_ok=True)
            seed_start = time.monotonic()
            files, metric_scenes = [], []
            with isolated_random_seed(int(seed), use_cuda=True):
                for batch_index, (batch_data, batch_id) in enumerate(loader):
                    if batch_index >= scene_limit:
                        break
                    inputs, seq_list = model.prepare_inputs(batch_data, batch_id)
                    metric_mask = compute_metric_mask(seq_list)
                    with torch.autocast(
                        device_type="cuda", dtype=torch.bfloat16):
                        predictions, auxiliary = model.forward(
                            inputs, if_test=True)
                    del auxiliary
                    if predictions.shape[0] != 20:
                        raise RuntimeError("native GDTS output is not P20")
                    down = float(model_args.down_factor)
                    pixel_prediction = predictions.detach() * down
                    pixel_truth = inputs["x_augmented"][:, :, 6:8].detach() * down
                    scene_geometry = inputs["scene"]
                    world_prediction = torch.stack([
                        scene_geometry.make_world_coord_torch(value)
                        for value in pixel_prediction])
                    world_truth = scene_geometry.make_world_coord_torch(pixel_truth)
                    valid_positions = metric_mask.nonzero(
                        as_tuple=False).flatten()
                    predicted = world_prediction[
                        :, model_args.obs_length:, valid_positions].permute(
                            0, 2, 1, 3).float().cpu().numpy()
                    history = world_truth[
                        :model_args.obs_length, valid_positions].permute(
                            1, 0, 2).float().cpu().numpy()
                    future = world_truth[
                        model_args.obs_length:, valid_positions].permute(
                            1, 0, 2).float().cpu().numpy()

                    frames = np.asarray(
                        _flatten_collated(batch_id["frame_ids"]),
                        dtype=np.int64)
                    all_agents = _flatten_collated(batch_id["agent_ids"])
                    if len(all_agents) != int(metric_mask.numel()):
                        raise RuntimeError("GDTS agent metadata length mismatch")
                    agents = np.asarray(
                        [str(all_agents[int(index)]) for index in valid_positions.cpu()],
                        dtype=np.str_)
                    source = canonical_source(batch_id["data_file_path"][0])
                    identity = canonical_window_key(source, frames)
                    scene = BankScene(
                        predicted.astype(np.float32), future.astype(np.float32),
                        history.astype(np.float32), frames, agents, source,
                        f"hotel:test:{identity}", identity, 0.4)
                    relative = Path("scenes") / f"scene_{batch_index:06d}.npz"
                    digest = save_scene(scene, seed_dir / relative)
                    metric_scenes.append(scene)
                    files.append({
                        "path": relative.as_posix(), "sha256": digest,
                        "identity": identity, "scene_id": scene.scene_id,
                        "source_sequence": source, "frame_ids": frames.tolist(),
                        "agent_ids": agents.tolist(), "valid_agents": scene.agents,
                        "original_batch_index": batch_index,
                        "original_batch_size": 1,
                        "original_scene_index": inputs["scene_index"].cpu().tolist(),
                        "original_world_order": list(range(20)),
                    })
                    if (batch_index + 1) % 50 == 0:
                        print(
                            f"GDTS seed {seed}: {batch_index + 1}/{scene_limit}",
                            flush=True)
            metrics = summarize_scenes(metric_scenes)
            manifest = {
                "schema": SEED_MANIFEST_SCHEMA, "bank_schema": BANK_SCHEMA,
                "status": "COMPLETED", "model": "GDTS", "fold": "hotel",
                "training_seed": "official checkpoint", "inference_seed": seed,
                "worlds": 20, "steps": int(model_args.ddim_step),
                "scene_count": len(files),
                "agent_occurrences": int(sum(x["valid_agents"] for x in files)),
                "unit": "m", "dtype": "float32", "dt_seconds": 0.4,
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
                "config": str(config_path),
                "config_sha256": sha256_file(config_path),
                "source_manifest": str(
                    Path(fields["jdv2_cache_root"]) / "manifest.json"),
                "source_manifest_sha256": sha256_file(
                    Path(fields["jdv2_cache_root"]) / "manifest.json"),
                "execution_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=gdts_root,
                    text=True).strip(),
                "inference_precision": "BF16 CUDA autocast; coordinate conversion FP32",
                "autocast": "torch.autocast(cuda,bfloat16) around model.forward",
                "input_information": "native GDTS trajectory maps + RGB/semantic tensor",
                "sampler": "native GDTS TTST + tree/DDIM sampler and global isolated RNG",
                "coordinate_conversion": (
                    "native downsampled pixel output multiplied once by down_factor; "
                    "scene homography applied once; history removed after conversion"),
                "optimizer_updates": 0, "forward_calls": len(files),
                "elapsed_seconds": time.monotonic() - seed_start,
                "replay_metrics": metrics, "files": files,
            }
            atomic_json(manifest, seed_dir / "manifest.json")
            seed_manifests.append(seed_dir / "manifest.json")
            print(f"GDTS seed {seed}: {metrics}", flush=True)

    index = {
        "schema": "ebjd-trajectory-bank-index-v1", "model": "GDTS",
        "status": "COMPLETED", "optimizer_updates": 0,
        "seed_manifests": [{
            "path": str(path.relative_to(output_root)),
            "sha256": sha256_file(path)} for path in seed_manifests],
        "elapsed_seconds": time.monotonic() - started,
        "cuda_peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "checkpoint_sha256_after": gdts_sha256(checkpoint_path),
    }
    atomic_json(index, output_root / "index.json")
    print(json.dumps(index, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
