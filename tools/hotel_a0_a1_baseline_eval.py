#!/usr/bin/env python3
"""Evaluate the frozen official GDTS parent on synchronized HOTEL windows."""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from src.data_loader import get_dataloader
from src.metrics import compute_metric_mask
from src.models.model import GDTS
from src.utils import isolated_random_seed
from src.joint_dependency_v2_cache import sha256_file


PARENT_SHA256 = (
    "5c101c2474ebb1a3cb3ecf882f0e9db2fbb2489c741c58068b8183b0b663b97b")
SEEDS = (2035, 2036, 2037, 2038, 2039)
METRICS = (
    "ADE_world", "FDE_world", "minADE@K", "minFDE@K", "JADE", "JFDE",
    "Collision_Rate",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with open(args.config) as handle:
        fields = yaml.safe_load(handle)
    loader_args = SimpleNamespace(**fields)
    loader_args.phase = "test"
    loader_args.shuffle_test_batches = False
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
    parent = Path(args.parent).resolve()
    if sha256_file(parent) != PARENT_SHA256:
        raise RuntimeError("Official baseline parent SHA256 mismatch")
    payload = torch.load(parent, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"], strict=True)

    runs = []
    first_support = None
    with torch.no_grad():
        for seed in SEEDS:
            values = {name: [] for name in METRICS}
            with isolated_random_seed(seed, use_cuda=True):
                for batch_data, batch_id in loader:
                    inputs, seq_list = model.prepare_inputs(batch_data, batch_id)
                    metric_mask = compute_metric_mask(seq_list)
                    with torch.autocast(
                            device_type="cuda", dtype=torch.bfloat16):
                        predictions, auxiliary = model.forward(
                            inputs, if_test=True)
                    if first_support is None:
                        first_support = {
                            "goal_point_shape": list(
                                auxiliary["goal_point"].shape),
                            "branch_count": int(model_args.num_samples),
                            "trunk_count": int(
                                auxiliary["goal_point"].shape[0] -
                                model_args.num_samples),
                        }
                    for name in METRICS:
                        values[name].extend(model.compute_model_metrics(
                            name, predictions, metric_mask, auxiliary,
                            inputs, obs_length=model_args.obs_length))
            runs.append({
                "evaluation_seed": seed,
                "metrics": {
                    name: float(np.mean(items)) if items else 0.0
                    for name, items in values.items()},
                "counts": {
                    name: len(items) for name, items in values.items()},
            })
    average = {
        name: float(np.mean([run["metrics"][name] for run in runs]))
        for name in METRICS}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps({
        "status": "COMPLETED", "protocol": "E_joint",
        "parent": str(parent), "parent_sha256": PARENT_SHA256,
        "evaluation_seeds": list(SEEDS), "runs": runs,
        "average": average, "candidate_support_probe": first_support,
        "aggregation": "metric-native values pooled across synchronized windows",
        "official_metric_comparability": False,
    }, indent=2, sort_keys=True))
    os.replace(temporary, output)


if __name__ == "__main__":
    main()
