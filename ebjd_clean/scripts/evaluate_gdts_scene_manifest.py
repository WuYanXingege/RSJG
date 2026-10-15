#!/usr/bin/env python3
"""Evaluate a frozen GDTS checkpoint on an EBJD synchronized-scene manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml


METRICS = (
    "ADE_world", "FDE_world", "minADE@K", "minFDE@K", "JADE", "JFDE",
)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def scenes(manifest_path: Path):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for descriptor in manifest["shards"]:
        with np.load(
                manifest_path.parent / descriptor["path"],
                allow_pickle=True) as payload:
            for index in range(len(payload["observed"])):
                yield {
                    "observed": np.asarray(payload["observed"][index]),
                    "future": np.asarray(payload["future"][index]),
                    "frame_ids": np.asarray(payload["frame_ids"][index]),
                    "window_key": str(payload["window_keys"][index]),
                }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jdv2-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(2035, 2040)))
    args = parser.parse_args()

    jdv2_root = Path(args.jdv2_root).resolve()
    sys.path.insert(0, str(jdv2_root))
    from src.data_src.dataset_src.dataset_create import create_dataset
    from src.metrics import compute_metric_mask
    from src.models.model import GDTS
    from src.models.model_utils.cnn_big_images_utils import (
        create_CNN_inputs_loop, create_tensor_image)
    from src.utils import isolated_random_seed

    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    manifest_path = Path(args.manifest).resolve()
    output_path = Path(args.output).resolve()
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
    dataset = create_dataset(model_args.dataset, load_visual_data=True)
    model = GDTS(model_args, device, dataset=dataset).to(device).eval()
    checkpoint_before = sha256(checkpoint_path)
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    scene = dataset.scenes[fields["test_set"]]
    tensor_image = create_tensor_image(
        scene.semantic_map_pred, down_factor=model_args.down_factor)
    tensor_map = create_tensor_image(
        scene.RGB_image, down_factor=model_args.down_factor)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_scenes = int(manifest["statistics"]["scenes"])
    expected_agents = int(manifest["statistics"]["agents"])
    protocol_binding = {
        "manifest_sha256": sha256(manifest_path),
        "checkpoint_sha256": checkpoint_before,
        "config_sha256": sha256(config_path),
        "scene_count": expected_scenes,
        "agent_occurrence_count": expected_agents,
    }
    per_seed_dir = output_path.parent / (output_path.stem + "_per_seed")
    per_seed_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    for seed in args.seeds:
        seed_path = per_seed_dir / f"seed_{seed}.json"
        if seed_path.is_file():
            record = json.loads(seed_path.read_text(encoding="utf-8"))
            reusable = (
                record.get("seed") == seed
                and record.get("status") == "complete"
                and record.get("protocol_binding") == protocol_binding
            )
            if reusable:
                runs.append(record)
                print(f"seed={seed} reused", flush=True)
                continue
            print(f"seed={seed} stale record ignored: {seed_path}", flush=True)
        started = time.time()
        values = {name: [] for name in METRICS}
        seen_scenes = seen_agents = 0
        with isolated_random_seed(seed, use_cuda=True), torch.no_grad():
            for item in scenes(manifest_path):
                world = torch.from_numpy(np.concatenate(
                    (item["observed"], item["future"]), axis=1
                ).transpose(1, 0, 2)).float()
                full_pixel = scene.make_pixel_coord_torch(world)
                downsampled = full_pixel / model_args.down_factor
                maps = create_CNN_inputs_loop(downsampled, tensor_image)
                n_agents = int(world.shape[1])
                frame_ids = torch.from_numpy(item["frame_ids"]).long()
                if frame_ids.ndim == 1:
                    frame_ids = frame_ids[:, None].expand(-1, n_agents).clone()
                raw = {
                    "abs_pixel_coord": downsampled.unsqueeze(0),
                    "seq_list": torch.ones(
                        1, model_args.seq_length, n_agents, dtype=torch.float32),
                    "scene_index": torch.zeros(1, n_agents, dtype=torch.long),
                    "scene_ptr": torch.tensor([[0, n_agents]], dtype=torch.long),
                    "frame_ids": frame_ids.unsqueeze(0),
                    "batch_format_version": torch.tensor([2], dtype=torch.long),
                    "tensor_image": tensor_image.unsqueeze(0),
                    "input_traj_maps": maps.unsqueeze(0),
                    "tensor_map": tensor_map.unsqueeze(0),
                }
                inputs, seq_list = model.prepare_inputs(
                    raw, {"scene_name": [fields["test_set"]]})
                metric_mask = compute_metric_mask(seq_list)
                with torch.autocast(
                        device_type=device.type, dtype=torch.bfloat16,
                        enabled=device.type == "cuda"):
                    predictions, auxiliary = model.forward(inputs, if_test=True)
                for name in METRICS:
                    values[name].extend(model.compute_model_metrics(
                        name, predictions, metric_mask, auxiliary, inputs,
                        obs_length=model_args.obs_length))
                seen_scenes += 1
                seen_agents += n_agents
                if seen_scenes % 100 == 0 or seen_scenes == expected_scenes:
                    print(
                        f"seed={seed} progress={seen_scenes}/{expected_scenes} "
                        f"agents={seen_agents}/{expected_agents}",
                        flush=True)
        counts = {name: len(items) for name, items in values.items()}
        if seen_scenes != expected_scenes or seen_agents != expected_agents:
            raise RuntimeError(
                f"coverage mismatch: scenes {seen_scenes}/{expected_scenes}, "
                f"agents {seen_agents}/{expected_agents}")
        record = {
            "status": "complete", "seed": seed,
            "protocol_binding": protocol_binding,
            "elapsed_seconds": time.time() - started,
            "scene_count": seen_scenes, "agent_occurrence_count": seen_agents,
            "counts": counts,
            **{name: float(np.mean(items)) if items else 0.0
               for name, items in values.items()},
        }
        atomic_json(record, seed_path)
        runs.append(record)
        print(
            f"seed={seed} minADE={record['minADE@K']:.6f} "
            f"minFDE={record['minFDE@K']:.6f} "
            f"JADE={record['JADE']:.6f} JFDE={record['JFDE']:.6f} "
            f"elapsed={record['elapsed_seconds']:.1f}s", flush=True)

    summary = {
        name: {
            "mean": statistics.mean(float(run[name]) for run in runs),
            "sample_std": (
                statistics.stdev(float(run[name]) for run in runs)
                if len(runs) > 1 else None),
        }
        for name in METRICS
    }
    result = {
        "schema": "gdts-on-ebjd-synchronized-scene-manifest-v1",
        "status": "complete" if len(runs) == 5 else "partial",
        "protocol": {
            "target": f"eth5/{fields['test_set']}", "split": "validation",
            "manifest": str(manifest_path), "manifest_sha256": sha256(manifest_path),
            "scene_count": expected_scenes,
            "agent_occurrence_count": expected_agents,
            "sample_count": int(model_args.num_samples),
            "seeds": list(args.seeds), "worlds": 20, "steps": 20,
            "metric_implementation": "GDTS.compute_model_metrics",
            "coordinate_conversion": "Scene_eth5.make_pixel_coord_torch",
            "inference_precision": "BF16 CUDA autocast; coordinate conversion FP32",
            "sampler": "native GDTS TTST plus tree/DDIM",
            "cross_method_rng_claim": "same seed labels, not identical random streams",
        },
        "checkpoint": {"path": str(checkpoint_path), "sha256": checkpoint_before},
        "config": {"path": str(config_path), "sha256": sha256(config_path)},
        "runs": runs, "summary": summary,
        "checkpoint_unchanged_during_evaluation": (
            checkpoint_before == sha256(checkpoint_path)),
    }
    if not result["checkpoint_unchanged_during_evaluation"]:
        raise RuntimeError("GDTS checkpoint changed during evaluation")
    atomic_json(result, output_path)


if __name__ == "__main__":
    main()
