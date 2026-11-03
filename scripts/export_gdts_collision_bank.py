#!/usr/bin/env python3
"""Export frozen matched-GDTS worlds without altering its prediction path."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import yaml


METRICS = ("minADE@K", "minFDE@K", "JADE", "JFDE")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_array(values: list) -> np.ndarray:
    result = np.empty(len(values), dtype=object)
    result[:] = values
    return result


def atomic_npz(path: Path, **arrays) -> None:
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


def _unwrap(values):
    while isinstance(values, (list, tuple)) and len(values) == 1 and isinstance(
        values[0], (list, tuple, np.ndarray)
    ):
        values = values[0]
    if torch.is_tensor(values):
        values = values.detach().cpu().numpy()
    return np.asarray(values).reshape(-1).tolist()


def _manifest_records(manifest: Path) -> tuple[list[dict], dict]:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    records = []
    for descriptor in payload["shards"]:
        shard = manifest.parent / descriptor["path"]
        if sha256(shard) != descriptor["sha256"]:
            raise ValueError(f"manifest shard SHA256 mismatch: {shard}")
        with np.load(shard, allow_pickle=True) as data:
            for index in range(len(data["scene_ids"])):
                records.append({
                    "scene_id": str(data["scene_ids"][index]),
                    "agent_ids": [str(value) for value in data["agent_ids"][index]],
                    "frame_ids": [int(value) for value in data["frame_ids"][index]],
                })
    return records, payload


def _scene_identity(batch_id: dict) -> tuple[list[int], list[str]]:
    frames = [int(np.asarray(value).reshape(-1)[0]) for value in batch_id["frame_ids"]]
    agents = [str(int(np.asarray(value).reshape(-1)[0])) for value in batch_id["agent_ids"]]
    return frames, agents


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--scene-cache-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("valid", "test"), default="valid")
    args = parser.parse_args()

    recipe_path = Path(args.recipe).resolve()
    recipe = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
    legacy = Path(recipe["legacy_gdts"]["root"]).resolve()
    runtime = Path(args.runtime_root).resolve()
    runtime.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.checkpoint).resolve()
    manifest = Path(args.manifest).resolve()
    output = Path(args.output).resolve()
    if sha256(checkpoint) != args.checkpoint_sha256:
        raise ValueError("matched GDTS checkpoint SHA256 mismatch")
    manifest_records, manifest_payload = _manifest_records(manifest)
    manifest_lookup = {
        (tuple(record["frame_ids"]), tuple(sorted(record["agent_ids"], key=int))): record
        for record in manifest_records
    }

    sys.path.insert(0, str(legacy))
    os.chdir(runtime)
    from src.models.model import GDTS
    from src.parser import check_and_add_additional_args, get_parser
    from src.trainer import trainer

    cli = [
        "--dataset", "eth5", "--test_set", str(recipe["fold"]),
        "--phase", "test", "--device", "cuda:0", "--batch_size", "64",
        "--num_workers", "0", "--down_factor", "8", "--skip_ts_window", "1",
        # The loader must retain synchronized scene_index metadata.  The model
        # itself is switched to the frozen independent matched-GDTS baseline
        # immediately after trainer construction below.
        "--goal_model_type", "social", "--training_stage", "finetune",
        "--data_augmentation", "False", "--shuffle_train_batches", "False",
        "--shuffle_test_batches", "False", "--num_samples", "20",
        "--ddpm_step", "100", "--ddim_step", "20", "--trunk_stage_step", "30",
        "--reproducibility", "True", "--seed", str(recipe["seed"]),
        "--batch_cache_root", str(Path(args.scene_cache_root).resolve()),
        "--use_wandb", "False",
    ]
    parsed = check_and_add_additional_args(get_parser().parse_args(cli))
    parsed.model_dir = str(output.parent / "gdts_runtime")
    parsed.save_dir = parsed.model_dir
    evaluator = trainer(parsed)
    parsed.goal_model_type = "independent"
    parsed.training_stage = "baseline"
    parsed.use_scene_latent = False
    parsed.use_dynamic_relation = False
    parsed.use_joint_energy = False
    parsed.use_dependency_corrector = False
    parsed.jdv2_active = False
    parsed.joint_goal_enabled = False
    parsed.best_metric = "auto"
    evaluator.args = parsed
    evaluator.net = GDTS(parsed, evaluator.device).to(evaluator.device)
    evaluator._pending_training_state = None
    evaluator._load_state_file(str(checkpoint), weights_only=True)
    evaluator.net.eval()

    # The frozen evaluator owns RNG, preprocessing, sampling and coordinate
    # conversion.  Hooks below only copy its already-produced tensors.
    scenes: OrderedDict[str, dict] = OrderedDict()
    original_prepare = evaluator.net.prepare_inputs
    original_metric = evaluator.net.compute_model_metrics
    capture_pending = {"value": False}

    def prepare_with_identity(batch_data, batch_id):
        inputs, seq_list = original_prepare(batch_data, batch_id)
        evaluator.net._collision_export_batch_id = batch_id
        capture_pending["value"] = True
        return inputs, seq_list

    def metric_with_capture(metric_name, predictions, metric_mask, all_aux_outputs,
                            inputs, obs_length=8):
        if capture_pending["value"]:
            capture_pending["value"] = False
            scaled = predictions.detach() * parsed.down_factor
            truth = inputs["x_augmented"][:, :, 6:8].detach() * parsed.down_factor
            scene_geometry = inputs["scene"]
            pred_world = torch.stack([
                scene_geometry.make_world_coord_torch(sample)
                for sample in scaled
            ])
            truth_world = scene_geometry.make_world_coord_torch(truth)
            batch_id = evaluator.net._collision_export_batch_id
            frames, agents = _scene_identity(batch_id)
            manifest_record = manifest_lookup.get(
                (tuple(frames), tuple(sorted(agents, key=int))))
            if manifest_record is None:
                raise RuntimeError(
                    f"GDTS window not present in manifest: frames={frames}, agents={agents}")
            identity = manifest_record["scene_id"]
            if identity in scenes:
                raise RuntimeError(f"duplicate GDTS scene identity: {identity}")
            cache_index = {agent: index for index, agent in enumerate(agents)}
            order = [cache_index[agent] for agent in manifest_record["agent_ids"]]
            chosen = torch.as_tensor(order, device=pred_world.device)
            scenes[identity] = {
                "agents": manifest_record["agent_ids"],
                "prediction_parts": [
                    pred_world[:, obs_length:, chosen].permute(0, 2, 1, 3).float().cpu()],
                "target_parts": [
                    truth_world[obs_length:, chosen].permute(1, 0, 2).float().cpu()],
            }
        return original_metric(
            metric_name, predictions, metric_mask, all_aux_outputs, inputs,
            obs_length=obs_length)

    evaluator.net.prepare_inputs = prepare_with_identity
    evaluator.net.compute_model_metrics = metric_with_capture
    started = time.perf_counter()
    raw = evaluator._evaluate_epoch(
        0, mode=args.split, evaluation_seed=args.seed, metric_names=METRICS)

    scene_ids, agent_ids, predictions, targets = [], [], [], []
    for identity, record in scenes.items():
        prediction = torch.cat(record["prediction_parts"], dim=1).numpy()
        target = torch.cat(record["target_parts"], dim=0).numpy()
        if prediction.shape[0] != 20 or prediction.shape[2:] != (12, 2):
            raise RuntimeError(f"unexpected GDTS prediction shape for {identity}")
        scene_ids.append(identity)
        agent_ids.append(np.asarray(record["agents"]).astype(str))
        predictions.append(prediction)
        targets.append(target)

    metadata = {
        "schema": "ebjd-jmm-collision-bank-v1",
        "fold": str(recipe["fold"]),
        "method": "GDTS",
        "seed": args.seed,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": args.checkpoint_sha256,
        "manifest": str(manifest),
        "manifest_sha256": sha256(manifest),
        "recipe": str(recipe_path),
        "recipe_sha256": sha256(recipe_path),
        "worlds": 20,
        "timesteps": 12,
        "coordinate_unit": "m",
        "scene_count": len(scene_ids),
        "agent_occurrence_count": sum(len(value) for value in agent_ids),
        "elapsed_seconds": time.perf_counter() - started,
        "displacement_metrics": {
            "minADE": float(raw["valid_minADE@K"]),
            "minFDE": float(raw["valid_minFDE@K"]),
            "JADE": float(raw["valid_JADE"]),
            "JFDE": float(raw["valid_JFDE"]),
        },
        "manifest_declared_statistics": manifest_payload.get("statistics", {}),
    }
    atomic_npz(
        output,
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        scene_ids=np.asarray(scene_ids),
        agent_ids=object_array(agent_ids),
        predictions=object_array(predictions),
        targets=object_array(targets),
    )
    metadata["bank_sha256"] = sha256(output)
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
