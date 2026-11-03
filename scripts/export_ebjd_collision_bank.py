#!/usr/bin/env python3
"""Export frozen unified EBJD G/H full trajectories for JMM CR evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
import sys

import numpy as np
import torch

# Direct execution sets sys.path[0] to ``scripts/``.  Add the paper root so
# the frozen unified runner is imported from this checkout without relying on
# an ambient PYTHONPATH.
PAPER_ROOT = Path(__file__).resolve().parents[1]
if str(PAPER_ROOT) not in sys.path:
    sys.path.insert(0, str(PAPER_ROOT))
SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from ebjd.sampling import differentiable_sample
from run_zara1_unified_coupling_bridge import (
    UnifiedCouplingBridge,
    build_model,
    fit_training_scales,
    gather_candidates,
    load_config,
    loader,
    make_bank,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_array(values: list) -> np.ndarray:
    array = np.empty(len(values), dtype=object)
    array[:] = values
    return array


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--validation-bank", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--fold", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    output = Path(args.output).resolve()
    if sha256(checkpoint) != args.checkpoint_sha256:
        raise ValueError("selected checkpoint SHA256 mismatch")
    config = load_config(config_path)
    manifest = Path(config["paths"]["validation_manifest"]).resolve()
    manifest_sha = config["evaluation"]["matched_gdts_validation"]["manifest_sha256"]
    if sha256(manifest) != manifest_sha:
        raise ValueError("validation manifest SHA256 mismatch")
    provider_sha = config["paths"]["matched_gdts_validation_checkpoint_sha256"]
    bank = make_bank(args.validation_bank, provider_sha, manifest_sha, "validation")

    device = torch.device(args.device)
    bridge = build_model(config, proposal_conditioned=True)
    fit_training_scales(
        bridge, loader(config, "train", torch.Generator().manual_seed(0)))
    model = UnifiedCouplingBridge(bridge).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()

    predictions_g, predictions_h, targets = [], [], []
    scene_ids, agent_ids, source_sequences = [], [], []
    generator = torch.Generator(device=device).manual_seed(args.seed)
    started = time.perf_counter()
    with torch.no_grad():
        for cpu_batch in loader(
            config, "validation", torch.Generator().manual_seed(99173)
        ):
            proposals = bank.batch(cpu_batch).to(device)
            batch = cpu_batch.to(device)
            context = model.encode_context(
                batch.observed, batch.semantic_maps, batch.valid)
            h_goals, permutation, _ = model.coupling.assign(
                context, proposals, model.bridge.representation.goal_scale,
                straight_through=False)
            noise = torch.randn(
                batch.observed.shape[0], 20, batch.observed.shape[1], 12, 2,
                device=device, generator=generator)
            g_path, g_goal, _ = differentiable_sample(
                model.bridge, context, noise, 20,
                proposal_goals=proposals, proposal_eta=0.0)
            h_path, h_goal, _ = differentiable_sample(
                model.bridge, context, gather_candidates(noise, permutation), 20,
                proposal_goals=h_goals, proposal_eta=0.0)
            if not torch.equal(g_goal.sort(1).values, h_goal.sort(1).values):
                raise RuntimeError("G/H endpoint multisets changed")
            for index in range(batch.observed.shape[0]):
                valid = batch.valid[index]
                predictions_g.append(
                    g_path[index, :, valid].float().cpu().numpy())
                predictions_h.append(
                    h_path[index, :, valid].float().cpu().numpy())
                targets.append(batch.future[index, valid].float().cpu().numpy())
                scene_ids.append(str(batch.scene_ids[index]))
                agent_ids.append(np.asarray(batch.agent_ids[index]).astype(str))
                source_sequences.append(str(batch.source_sequences[index]))

    metadata = {
        "schema": "ebjd-jmm-collision-bank-v1",
        "fold": args.fold,
        "method": "G_H",
        "seed": args.seed,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": args.checkpoint_sha256,
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "manifest": str(manifest),
        "manifest_sha256": manifest_sha,
        "proposal_bank": str(Path(args.validation_bank).resolve()),
        "proposal_bank_sha256": bank.sha256,
        "worlds": 20,
        "timesteps": 12,
        "coordinate_unit": "m",
        "scene_count": len(scene_ids),
        "agent_occurrence_count": sum(len(value) for value in agent_ids),
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_npz(
        output,
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        scene_ids=np.asarray(scene_ids),
        agent_ids=object_array(agent_ids),
        source_sequences=np.asarray(source_sequences),
        predictions_G=object_array(predictions_g),
        predictions_H=object_array(predictions_h),
        targets=object_array(targets),
    )
    metadata["bank_sha256"] = sha256(output)
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
