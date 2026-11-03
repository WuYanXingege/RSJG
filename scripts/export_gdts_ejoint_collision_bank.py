#!/usr/bin/env python3
"""Capture the frozen E_joint GDTS prediction path used by the formal baseline."""

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
from types import SimpleNamespace

import numpy as np
import torch
import yaml


METRICS = ("ADE_world", "FDE_world", "minADE@K", "minFDE@K", "JADE", "JFDE")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _objects(values: list) -> np.ndarray:
    result = np.empty(len(values), dtype=object)
    result[:] = values
    return result


def _flat(value) -> list:
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value).reshape(-1).tolist()


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.legacy_root).resolve()
    config = Path(args.config).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    manifest = Path(args.manifest).resolve()
    output = Path(args.output).resolve()
    if sha256(checkpoint) != args.checkpoint_sha256:
        raise RuntimeError("matched GDTS checkpoint SHA256 mismatch")

    sys.path.insert(0, str(root))
    os.chdir(root)
    from src.data_loader import get_dataloader
    from src.metrics import compute_metric_mask
    from src.models.model import GDTS
    from src.utils import isolated_random_seed

    fields = yaml.safe_load(config.read_text(encoding="utf-8"))
    loader_args = SimpleNamespace(**fields)
    loader_args.phase = "test"
    loader_args.shuffle_test_batches = False
    loader_args.jdv2_active = True
    loader = get_dataloader(loader_args, set_name="test")

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
    model = GDTS(model_args, device).to(device).eval()
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"], strict=True)

    values = {name: [] for name in METRICS}
    scenes: OrderedDict[str, dict] = OrderedDict()
    started = time.perf_counter()
    with torch.no_grad(), isolated_random_seed(args.seed, use_cuda=True):
        for batch_data, batch_id in loader:
            inputs, seq_list = model.prepare_inputs(batch_data, batch_id)
            metric_mask = compute_metric_mask(seq_list)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                prediction, auxiliary = model.forward(inputs, if_test=True)
            for name in METRICS:
                values[name].extend(model.compute_model_metrics(
                    name, prediction, metric_mask, auxiliary, inputs,
                    obs_length=model_args.obs_length))

            pixel_prediction = prediction.detach() * model_args.down_factor
            pixel_target = inputs["x_augmented"][:, :, 6:8].detach() * model_args.down_factor
            geometry = inputs["scene"]
            world_prediction = torch.stack([
                geometry.make_world_coord_torch(sample)
                for sample in pixel_prediction
            ])
            world_target = geometry.make_world_coord_torch(pixel_target)
            starts = _flat(batch_id["starting_frames"])
            agents = _flat(batch_id["agent_ids"])
            if len(starts) != world_prediction.shape[2] or len(agents) != len(starts):
                raise RuntimeError("identity arrays do not match predicted agents")
            scene_name = batch_id["scene_name"]
            while isinstance(scene_name, (list, tuple)):
                scene_name = scene_name[0]
            for start in dict.fromkeys(starts):
                positions = [i for i, item in enumerate(starts) if item == start]
                identity = f"{scene_name}:{int(start)}"
                if identity in scenes:
                    raise RuntimeError(f"duplicate formal scene identity: {identity}")
                chosen = torch.as_tensor(positions, device=device)
                scenes[identity] = {
                    "agents": np.asarray([str(agents[i]) for i in positions]),
                    "prediction": world_prediction[:, model_args.obs_length:, chosen]
                    .permute(0, 2, 1, 3).float().cpu().numpy(),
                    "target": world_target[model_args.obs_length:, chosen]
                    .permute(1, 0, 2).float().cpu().numpy(),
                }

    metrics = {name: float(np.mean(items)) for name, items in values.items()}
    scene_ids = list(scenes)
    metadata = {
        "schema": "ebjd-jmm-collision-bank-v1",
        "method": "GDTS", "protocol": "E_joint_exact_formal_replay",
        "seed": args.seed, "checkpoint": str(checkpoint),
        "checkpoint_sha256": args.checkpoint_sha256,
        "manifest": str(manifest), "manifest_sha256": sha256(manifest),
        "config": str(config), "config_sha256": sha256(config),
        "worlds": 20, "timesteps": 12, "coordinate_unit": "m",
        "scene_count": len(scene_ids),
        "agent_occurrence_count": sum(len(item["agents"]) for item in scenes.values()),
        "elapsed_seconds": time.perf_counter() - started,
        "displacement_metrics": {
            "minADE": metrics["minADE@K"], "minFDE": metrics["minFDE@K"],
            "JADE": metrics["JADE"], "JFDE": metrics["JFDE"],
        },
    }
    _atomic_npz(
        output, metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        scene_ids=np.asarray(scene_ids),
        agent_ids=_objects([scenes[key]["agents"] for key in scene_ids]),
        predictions=_objects([scenes[key]["prediction"] for key in scene_ids]),
        targets=_objects([scenes[key]["target"] for key in scene_ids]),
    )
    metadata["bank_sha256"] = sha256(output)
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
