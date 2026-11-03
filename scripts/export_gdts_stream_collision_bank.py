#!/usr/bin/env python3
"""Stream formal manifest scenes through frozen GDTS and save full trajectories."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from export_zara1_endpoint_banks import manifest_identity


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _objects(values: list) -> np.ndarray:
    result = np.empty(len(values), dtype=object)
    result[:] = values
    return result


def _atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _metrics(predictions: list[np.ndarray], targets: list[np.ndarray]) -> dict:
    marginal_ade, marginal_fde, jade, jfde = [], [], [], []
    for prediction, target in zip(predictions, targets):
        distance = np.linalg.norm(prediction - target[None], axis=-1)
        marginal_ade.extend(distance.mean(axis=-1).min(axis=0).tolist())
        marginal_fde.extend(distance[..., -1].min(axis=0).tolist())
        jade.append(float(distance.mean(axis=(1, 2)).min()))
        jfde.append(float(distance[..., -1].mean(axis=1).min()))
    return {
        "minADE": float(np.mean(marginal_ade)),
        "minFDE": float(np.mean(marginal_fde)),
        "JADE": float(np.mean(jade)), "JFDE": float(np.mean(jfde)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    recipe_path = Path(args.recipe).resolve()
    recipe = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
    manifest_path = Path(args.manifest).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    output = Path(args.output).resolve()
    if sha256(manifest_path) != args.manifest_sha256:
        raise ValueError("manifest SHA256 mismatch")
    if sha256(checkpoint) != args.checkpoint_sha256:
        raise ValueError("checkpoint SHA256 mismatch")
    records, manifest = manifest_identity(manifest_path)
    targets_by_source: dict[str, list[dict]] = {}
    for record in records:
        targets_by_source.setdefault(record["scene_id"].split(":")[2], []).append(record)

    legacy = Path(recipe["legacy_gdts"]["root"]).resolve()
    runtime = Path(args.runtime_root).resolve()
    runtime.mkdir(parents=True, exist_ok=True)
    data_link = runtime / "data"
    source_data = Path(recipe["source_data_root"]).resolve()
    if not data_link.exists() and not data_link.is_symlink():
        data_link.symlink_to(source_data, target_is_directory=True)
    if data_link.resolve() != source_data:
        raise RuntimeError("runtime data link target mismatch")
    sys.path.insert(0, str(legacy))
    os.chdir(runtime)
    from src.data_pre_process import Trajectory_Data_Pre_Process
    from src.models.model import GDTS
    from src.parser import check_and_add_additional_args, get_parser

    device = torch.device(args.device)
    cli = [
        "--dataset", "eth5", "--test_set", str(recipe["fold"]),
        "--phase", "pre-process", "--device", str(device),
        "--batch_size", "64", "--num_workers", "0", "--down_factor", "8",
        "--skip_ts_window", "1", "--goal_model_type", "social",
        "--training_stage", "finetune", "--data_augmentation", "False",
        "--shuffle_train_batches", "False", "--shuffle_test_batches", "False",
        "--num_samples", "20", "--ddpm_step", "100", "--ddim_step", "20",
        "--trunk_stage_step", "30", "--reproducibility", "True",
        "--seed", str(recipe["seed"]), "--use_wandb", "False",
    ]
    preprocessor_args = check_and_add_additional_args(get_parser().parse_args(cli))
    model_args = copy.deepcopy(preprocessor_args)
    model_args.goal_model_type = "independent"
    model_args.training_stage = "baseline"
    model_args.use_scene_latent = False
    model_args.use_dynamic_relation = False
    model_args.use_joint_energy = False
    model_args.use_dependency_corrector = False
    model_args.jdv2_active = False
    model_args.joint_goal_enabled = False
    model = GDTS(model_args, device).to(device).eval()
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    state = payload.get("model_state_dict", payload.get("model"))
    if state is None:
        raise RuntimeError("unsupported matched-GDTS checkpoint schema")
    model.load_state_dict(state, strict=True)

    prediction_by_id: dict[str, np.ndarray] = {}
    target_by_id: dict[str, np.ndarray] = {}
    original_add = Trajectory_Data_Pre_Process.add_pre_computed_cnn_maps
    expected_split = "valid"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.synchronize(device)
    started = time.perf_counter()

    def stream(self, scene_data, set_name):
        if set_name != expected_split:
            return
        source = Path(scene_data["file_path"]).stem
        if source not in targets_by_source:
            return
        geometry = self.dataset.scenes[scene_data["scene_name"]]
        for record in targets_by_source[source]:
            full_world = np.concatenate((record["observed"], record["future"]), axis=1)
            full_pixel = geometry.make_pixel_coord_torch(
                torch.from_numpy(full_world).transpose(0, 1)).cpu().numpy()
            batch_data = {
                "abs_pixel_coord": full_pixel,
                "seq_list": np.ones(full_pixel.shape[:2], dtype=np.float32),
            }
            raw_identity = {
                "scene_name": scene_data["scene_name"],
                "data_file_path": scene_data["file_path"],
                "frame_ids": [np.asarray(record["frame_ids"], dtype=np.int64)],
                "agent_ids": [np.asarray([int(value)], dtype=np.int64)
                              for value in record["agent_ids"]],
            }
            batch_data = original_add(self, batch_data, raw_identity)
            batch_id = dict(raw_identity)
            batch_id["scene_name"] = [raw_identity["scene_name"]]
            batch_id["data_file_path"] = [raw_identity["data_file_path"]]
            for key, value in tuple(batch_data.items()):
                if isinstance(value, np.ndarray):
                    batch_data[key] = torch.from_numpy(np.ascontiguousarray(value))
            # ``GDTS.prepare_inputs`` consumes the output of a DataLoader and
            # therefore removes its leading batch dimension.  The streaming
            # path bypasses the DataLoader, so reproduce that collation layer
            # explicitly for every tensor (including maps and trajectory
            # heatmaps) rather than letting prepare_inputs squeeze a semantic
            # dimension.
            for key, value in tuple(batch_data.items()):
                if torch.is_tensor(value):
                    batch_data[key] = value.unsqueeze(0)
            batch_data["abs_pixel_coord"] /= float(model_args.down_factor)
            inputs, _ = model.prepare_inputs(batch_data, batch_id)
            with torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16,
                    enabled=device.type == "cuda"):
                trajectory, _ = model.forward(inputs, if_test=True)
            future_pixel = trajectory[:, model_args.obs_length:].detach() * model_args.down_factor
            future_world = torch.stack([
                geometry.make_world_coord_torch(sample) for sample in future_pixel
            ]).permute(0, 2, 1, 3)
            prediction_by_id[record["scene_id"]] = future_world.float().cpu().numpy()
            target_by_id[record["scene_id"]] = record["future"].astype(np.float32)
            if len(prediction_by_id) % 100 == 0:
                print(f"seed={args.seed} progress={len(prediction_by_id)}/{len(records)}", flush=True)

    Trajectory_Data_Pre_Process.make_scene_window_batches = stream
    with tempfile.TemporaryDirectory(
            prefix=f"gdts_stream_{recipe['fold']}_{args.seed}_", dir=runtime) as scratch:
        preprocessor_args.batch_cache_root = scratch
        Trajectory_Data_Pre_Process(preprocessor_args)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    if len(prediction_by_id) != len(records):
        missing = [item["scene_id"] for item in records
                   if item["scene_id"] not in prediction_by_id]
        raise RuntimeError(f"stream covered {len(prediction_by_id)}/{len(records)}: {missing[:3]}")
    predictions = [prediction_by_id[item["scene_id"]] for item in records]
    targets = [target_by_id[item["scene_id"]] for item in records]
    metadata = {
        "schema": "ebjd-jmm-collision-bank-v1", "method": "GDTS",
        "protocol": "formal_manifest_stream", "fold": recipe["fold"],
        "seed": args.seed, "checkpoint": str(checkpoint),
        "checkpoint_sha256": args.checkpoint_sha256,
        "manifest": str(manifest_path), "manifest_sha256": args.manifest_sha256,
        "recipe": str(recipe_path), "recipe_sha256": sha256(recipe_path),
        "worlds": 20, "timesteps": 12, "coordinate_unit": "m",
        "scene_count": len(records),
        "agent_occurrence_count": int(manifest["statistics"]["agents"]),
        "elapsed_seconds": elapsed,
        "displacement_metrics": _metrics(predictions, targets),
    }
    _atomic_npz(
        output, metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        scene_ids=np.asarray([item["scene_id"] for item in records]),
        agent_ids=_objects([np.asarray(item["agent_ids"]) for item in records]),
        predictions=_objects(predictions), targets=_objects(targets),
    )
    metadata["bank_sha256"] = sha256(output)
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
