#!/usr/bin/env python3
"""Export the frozen native GDTS proposal branch for the training manifest."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch

from export_zara1_endpoint_banks import (
    configure_legacy_evaluator, gdts_scene_identity, manifest_identity,
    pack_bank, save_bank, sha256,
)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--scene-cache-root", required=True)
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=3101)
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint).resolve()
    manifest = Path(args.manifest).resolve()
    if sha256(checkpoint) != args.checkpoint_sha256:
        raise ValueError("GDTS checkpoint SHA256 mismatch")
    if sha256(manifest) != args.manifest_sha256:
        raise ValueError("training manifest SHA256 mismatch")
    records, manifest_payload = manifest_identity(manifest)
    lookup = {
        (record["scene_id"].split(":")[2],
         tuple(record["frame_ids"]), tuple(record["agent_ids"])): record
        for record in records
    }
    evaluator = configure_legacy_evaluator(args)
    loader = evaluator.data_loaders["train"]
    if len(loader.dataset) != len(records):
        raise RuntimeError(
            f"native cache/manifest scene mismatch: {len(loader.dataset)} != {len(records)}")
    from src.utils import isolated_random_seed

    proposals: dict[str, np.ndarray] = {}
    if evaluator.device.type == "cuda":
        torch.cuda.synchronize(evaluator.device)
    started = time.perf_counter()
    with isolated_random_seed(args.seed, use_cuda=True):
        for batch_data, batch_id in loader:
            inputs, _ = evaluator.net.prepare_inputs(batch_data, batch_id)
            with evaluator._autocast_context():
                _, auxiliary = evaluator.net._legacy_encode(inputs, if_test=True)
            frames, agents = gdts_scene_identity(batch_id)
            source = Path(str(batch_id["data_file_path"])).stem
            record = lookup.get((source, tuple(frames), tuple(agents)))
            if record is None:
                raise RuntimeError(
                    f"native GDTS identity absent from manifest: frames={frames}, agents={agents}")
            proposal_pixel = auxiliary["goal_point"][:20].detach() * float(
                evaluator.args.down_factor)
            proposal_world = inputs["scene"].make_world_coord_torch(proposal_pixel)
            proposals[record["scene_id"]] = proposal_world.float().cpu().numpy()
    if evaluator.device.type == "cuda":
        torch.cuda.synchronize(evaluator.device)
    elapsed = time.perf_counter() - started
    if len(proposals) != len(records):
        raise RuntimeError("native proposal export did not cover the training manifest")
    arrays = pack_bank(records, proposals)
    metadata = {
        "schema": "ebjd-native-proposal-bank-v1",
        "fold": "zara1", "split": "train", "seed": int(args.seed),
        "worlds": 20, "coordinate_unit": "m",
        "endpoint_source": "matched_gdts_native_legacy_encode_goal_point_first20",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": args.checkpoint_sha256,
        "manifest": str(manifest), "manifest_sha256": args.manifest_sha256,
        "scene_count": len(records),
        "agent_occurrence_count": int(manifest_payload["statistics"]["agents"]),
        "gt_used_for_generation": False,
        "gpu_elapsed_seconds": elapsed,
    }
    output = Path(args.output_dir).resolve() / "gdts_proposal_train_seed3101.npz"
    receipt = save_bank(output, arrays, metadata)
    print({"status": "COMPLETE", "path": str(output),
           "sha256": receipt["sha256"], "gpu_seconds": elapsed})


if __name__ == "__main__":
    main()
