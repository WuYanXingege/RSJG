#!/usr/bin/env python3
"""Capture full frozen GDTS worlds from the validated JDV2 cache loader."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from export_gdts_collision_bank import (
    METRICS, _manifest_records, _scene_identity, atomic_npz, object_array, sha256,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root, config = Path(args.legacy_root).resolve(), Path(args.config).resolve()
    checkpoint, manifest = Path(args.checkpoint).resolve(), Path(args.manifest).resolve()
    output, runtime = Path(args.output).resolve(), Path(args.runtime_root).resolve()
    runtime.mkdir(parents=True, exist_ok=True)
    if sha256(checkpoint) != args.checkpoint_sha256:
        raise ValueError("checkpoint SHA256 mismatch")
    records, manifest_payload = _manifest_records(manifest)
    lookup = {(tuple(r["frame_ids"]), tuple(sorted(r["agent_ids"], key=int))): r
              for r in records}
    sys.path.insert(0, str(root))
    os.chdir(root)
    from tools.audit_jdv2_stage_a import _baseline_evaluator

    evaluator = _baseline_evaluator(config, checkpoint, runtime, "cuda:0")
    edge_cache = str(Path(yaml.safe_load(config.read_text())["jdv2_cache_root"]).resolve())
    scenes: dict[str, dict] = {}
    original_prepare = evaluator.net.prepare_inputs
    original_metric = evaluator.net.compute_model_metrics
    pending = {"value": False}

    def prepare(batch_data, batch_id):
        inputs, sequence = original_prepare(batch_data, batch_id)
        evaluator.net._cr_batch_id = batch_id
        pending["value"] = True
        return inputs, sequence

    def metric(metric_name, predictions, metric_mask, all_aux_outputs, inputs,
               obs_length=8):
        if pending["value"]:
            pending["value"] = False
            prediction = predictions.detach() * evaluator.args.down_factor
            target = inputs["x_augmented"][:, :, 6:8].detach() * evaluator.args.down_factor
            geometry = inputs["scene"]
            pred_world = torch.stack([geometry.make_world_coord_torch(x) for x in prediction])
            target_world = geometry.make_world_coord_torch(target)
            frames, agents = _scene_identity(evaluator.net._cr_batch_id)
            record = lookup.get((tuple(frames), tuple(sorted(agents, key=int))))
            if record is None:
                raise RuntimeError(f"JDV2 cache window absent from manifest: {frames}/{agents}")
            index = {agent: position for position, agent in enumerate(agents)}
            chosen = torch.as_tensor([index[a] for a in record["agent_ids"]], device=pred_world.device)
            scenes[record["scene_id"]] = {
                "agents": np.asarray(record["agent_ids"]),
                "prediction": pred_world[:, obs_length:, chosen].permute(0, 2, 1, 3).float().cpu().numpy(),
                "target": target_world[obs_length:, chosen].permute(1, 0, 2).float().cpu().numpy(),
            }
        return original_metric(
            metric_name, predictions, metric_mask, all_aux_outputs, inputs,
            obs_length=obs_length)

    evaluator.net.prepare_inputs = prepare
    evaluator.net.compute_model_metrics = metric
    started = time.perf_counter()
    raw = evaluator._evaluate_epoch(
        0, mode="valid", evaluation_seed=args.seed, metric_names=METRICS,
        external_edge_cache_root=edge_cache)
    elapsed = time.perf_counter() - started
    if len(scenes) != len(records):
        raise RuntimeError(f"captured {len(scenes)}/{len(records)} manifest scenes")
    scene_ids = [record["scene_id"] for record in records]
    metadata = {
        "schema": "ebjd-jmm-collision-bank-v1", "method": "GDTS",
        "protocol": "validated_JDV2_cache_formal_replay",
        "fold": str(manifest_payload.get("fold", "univ")), "seed": args.seed,
        "checkpoint": str(checkpoint), "checkpoint_sha256": args.checkpoint_sha256,
        "manifest": str(manifest), "manifest_sha256": sha256(manifest),
        "config": str(config), "config_sha256": sha256(config),
        "worlds": 20, "timesteps": 12, "coordinate_unit": "m",
        "scene_count": len(scene_ids),
        "agent_occurrence_count": sum(len(scenes[x]["agents"]) for x in scene_ids),
        "elapsed_seconds": elapsed,
        "displacement_metrics": {
            "minADE": float(raw["valid_minADE@K"]),
            "minFDE": float(raw["valid_minFDE@K"]),
            "JADE": float(raw["valid_JADE"]), "JFDE": float(raw["valid_JFDE"]),
        },
    }
    atomic_npz(
        output, metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        scene_ids=np.asarray(scene_ids),
        agent_ids=object_array([scenes[x]["agents"] for x in scene_ids]),
        predictions=object_array([scenes[x]["prediction"] for x in scene_ids]),
        targets=object_array([scenes[x]["target"] for x in scene_ids]),
    )
    metadata["bank_sha256"] = sha256(output)
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
