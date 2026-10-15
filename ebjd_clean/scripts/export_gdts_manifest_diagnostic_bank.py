#!/usr/bin/env python3
"""Export a frozen native GDTS P20 bank on an exact EBJD scene manifest."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from ebjd.diagnostic_bank import (
    BANK_SCHEMA, SEED_MANIFEST_SCHEMA, BankScene, atomic_json, save_scene,
    sha256_file, summarize_scenes,
)


def manifest_scenes(path: Path):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for descriptor in manifest["shards"]:
        shard = path.parent / descriptor["path"]
        if sha256_file(shard) != descriptor["sha256"]:
            raise RuntimeError(f"manifest shard SHA256 mismatch: {shard}")
        with np.load(shard, allow_pickle=True) as payload:
            for index in range(len(payload["observed"])):
                yield {
                    name: np.asarray(payload[name][index])
                    for name in ("observed", "future", "frame_ids", "agent_ids")
                } | {
                    name: str(payload[name][index])
                    for name in ("scene_ids", "window_keys", "source_sequences")
                }


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
        description="Export native GDTS trajectories on an exact EBJD manifest")
    parser.add_argument("--gdts-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[2035])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()

    gdts_root = Path(args.gdts_root).resolve()
    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    manifest_path = Path(args.manifest).resolve()
    for path, expected, label in (
        (checkpoint_path, args.expected_checkpoint_sha256, "checkpoint"),
        (config_path, args.expected_config_sha256, "config"),
        (manifest_path, args.expected_manifest_sha256, "manifest"),
    ):
        if sha256_file(path) != expected:
            raise RuntimeError(f"GDTS {label} SHA256 mismatch")

    sys.path.insert(0, str(gdts_root))
    from src.data_src.dataset_src.dataset_create import create_dataset
    from src.metrics import compute_metric_mask
    from src.models.model import GDTS
    from src.models.model_utils.cnn_big_images_utils import (
        create_CNN_inputs_loop, create_tensor_image,
    )
    from src.utils import isolated_random_seed

    fields = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    fields.update({
        "goal_model_type": "independent", "jdv2_active": False,
        "use_scene_latent": False, "use_dynamic_relation": False,
        "use_joint_energy": False, "use_dependency_corrector": False,
        "training_stage": "baseline", "trajectory_coupling": "none",
        "trajectory_alignment": False, "phase": "test",
        "device": args.device, "shuffle_test_batches": False,
        "data_augmentation": False,
    })
    model_args = SimpleNamespace(**fields)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("native GDTS bank export requires an available CUDA GPU")
    dataset = create_dataset(model_args.dataset, load_visual_data=True)
    model = GDTS(model_args, device, dataset=dataset).to(device).eval()
    checkpoint_before = sha256_file(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    geometry = dataset.scenes[fields["test_set"]]
    tensor_image = create_tensor_image(
        geometry.semantic_map_pred, down_factor=model_args.down_factor)
    tensor_map = create_tensor_image(
        geometry.RGB_image, down_factor=model_args.down_factor)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_scenes = int(manifest["statistics"]["scenes"])
    expected_agents = int(manifest["statistics"]["agents"])
    scene_limit = expected_scenes if args.max_scenes is None else min(
        expected_scenes, int(args.max_scenes))
    if scene_limit < 1:
        raise ValueError("--max-scenes must be positive")

    output_root = Path(args.output_root).resolve() / "gdts"
    output_root.mkdir(parents=True, exist_ok=True)
    seed_manifests = []
    with torch.no_grad():
        for seed in args.seeds:
            seed_dir = output_root / f"seed_{seed}"
            if args.resume and complete_seed(seed_dir, scene_limit):
                seed_manifests.append(seed_dir / "manifest.json")
                print(f"GDTS seed {seed}: verified existing bank", flush=True)
                continue
            seed_dir.mkdir(parents=True, exist_ok=True)
            files, metric_scenes = [], []
            seed_start = time.monotonic()
            with isolated_random_seed(int(seed), use_cuda=True):
                for scene_index, item in enumerate(manifest_scenes(manifest_path)):
                    if scene_index >= scene_limit:
                        break
                    world = torch.from_numpy(np.concatenate(
                        (item["observed"], item["future"]), axis=1
                    ).transpose(1, 0, 2)).float()
                    full_pixel = geometry.make_pixel_coord_torch(world)
                    downsampled = full_pixel / model_args.down_factor
                    maps = create_CNN_inputs_loop(downsampled, tensor_image)
                    n_agents = int(world.shape[1])
                    frames = torch.from_numpy(item["frame_ids"]).long()
                    frame_matrix = frames[:, None].expand(-1, n_agents).clone()
                    raw = {
                        "abs_pixel_coord": downsampled.unsqueeze(0),
                        "seq_list": torch.ones(
                            1, model_args.seq_length, n_agents,
                            dtype=torch.float32),
                        "scene_index": torch.zeros(1, n_agents, dtype=torch.long),
                        "scene_ptr": torch.tensor([[0, n_agents]], dtype=torch.long),
                        "frame_ids": frame_matrix.unsqueeze(0),
                        "batch_format_version": torch.tensor([2], dtype=torch.long),
                        "tensor_image": tensor_image.unsqueeze(0),
                        "input_traj_maps": maps.unsqueeze(0),
                        "tensor_map": tensor_map.unsqueeze(0),
                    }
                    inputs, seq_list = model.prepare_inputs(
                        raw, {"scene_name": [fields["test_set"]]})
                    valid = compute_metric_mask(seq_list)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        predictions, _ = model.forward(inputs, if_test=True)
                    if predictions.shape[0] != 20:
                        raise RuntimeError("native GDTS output is not P20")
                    down = float(model_args.down_factor)
                    pixel_prediction = predictions.detach() * down
                    pixel_truth = inputs["x_augmented"][:, :, 6:8].detach() * down
                    world_prediction = torch.stack([
                        geometry.make_world_coord_torch(value)
                        for value in pixel_prediction])
                    world_truth = geometry.make_world_coord_torch(pixel_truth)
                    positions = valid.nonzero(as_tuple=False).flatten()
                    predicted = world_prediction[
                        :, model_args.obs_length:, positions].permute(
                            0, 2, 1, 3).float().cpu().numpy()
                    history = world_truth[
                        :model_args.obs_length, positions].permute(
                            1, 0, 2).float().cpu().numpy()
                    future = world_truth[
                        model_args.obs_length:, positions].permute(
                            1, 0, 2).float().cpu().numpy()
                    all_agents = np.asarray(item["agent_ids"], dtype=np.str_)
                    agents = all_agents[positions.cpu().numpy()]
                    bank_scene = BankScene(
                        predicted.astype(np.float32), future.astype(np.float32),
                        history.astype(np.float32), item["frame_ids"].astype(np.int64),
                        agents, item["source_sequences"], item["scene_ids"],
                        item["window_keys"], 0.4)
                    relative = Path("scenes") / f"scene_{scene_index:06d}.npz"
                    digest = save_scene(bank_scene, seed_dir / relative)
                    metric_scenes.append(bank_scene)
                    files.append({
                        "path": relative.as_posix(), "sha256": digest,
                        "identity": bank_scene.window_key,
                        "scene_id": bank_scene.scene_id,
                        "source_sequence": bank_scene.source_sequence,
                        "frame_ids": bank_scene.frame_ids.tolist(),
                        "agent_ids": bank_scene.agent_ids.tolist(),
                        "valid_agents": bank_scene.agents,
                        "original_batch_index": scene_index,
                        "original_batch_size": 1,
                        "original_world_order": list(range(20)),
                    })
                    if (scene_index + 1) % 50 == 0:
                        print(f"GDTS seed {seed}: {scene_index + 1}/{scene_limit}", flush=True)
            replay = summarize_scenes(metric_scenes)
            if args.max_scenes is None and (
                    len(files) != expected_scenes
                    or sum(item["valid_agents"] for item in files) != expected_agents):
                raise RuntimeError("GDTS bank coverage mismatch")
            record = {
                "schema": SEED_MANIFEST_SCHEMA, "bank_schema": BANK_SCHEMA,
                "status": "COMPLETED", "model": "GDTS", "fold": "eth",
                "training_seed": "official checkpoint", "inference_seed": seed,
                "worlds": 20, "steps": int(model_args.ddim_step),
                "scene_count": len(files),
                "agent_occurrences": int(sum(x["valid_agents"] for x in files)),
                "unit": "m", "dtype": "float32", "dt_seconds": 0.4,
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_before,
                "config": str(config_path),
                "config_sha256": sha256_file(config_path),
                "source_manifest": str(manifest_path),
                "source_manifest_sha256": sha256_file(manifest_path),
                "execution_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=gdts_root,
                    text=True).strip(),
                "inference_precision": "BF16 CUDA autocast; coordinate conversion FP32",
                "autocast": "torch.autocast(cuda,bfloat16) around model.forward",
                "input_information": "exact EBJD JMM scenes through native GDTS input builder",
                "sampler": "native GDTS TTST + tree/DDIM and isolated global RNG",
                "coordinate_conversion": "native ETH homography; exported in metres",
                "optimizer_updates": 0, "forward_calls": len(files),
                "elapsed_seconds": time.monotonic() - seed_start,
                "replay_metrics": replay, "files": files,
            }
            atomic_json(record, seed_dir / "manifest.json")
            seed_manifests.append(seed_dir / "manifest.json")
            print(f"GDTS seed {seed}: {replay}", flush=True)

    if sha256_file(checkpoint_path) != checkpoint_before:
        raise RuntimeError("GDTS checkpoint changed during export")
    index = {
        "schema": "ebjd-trajectory-bank-index-v1", "model": "GDTS",
        "status": "COMPLETED", "optimizer_updates": 0,
        "seed_manifests": [{
            "path": str(path.relative_to(output_root)),
            "sha256": sha256_file(path),
        } for path in seed_manifests],
        "elapsed_seconds": time.monotonic() - started,
        "cuda_peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "checkpoint_sha256_after": sha256_file(checkpoint_path),
    }
    atomic_json(index, output_root / "index.json")
    print(json.dumps(index, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
