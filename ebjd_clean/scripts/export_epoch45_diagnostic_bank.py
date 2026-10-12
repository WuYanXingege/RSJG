#!/usr/bin/env python3
"""Export the frozen EBJD P20 trajectories without altering inference."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ebjd.config import build_model, load_config
from ebjd.data import SceneManifestDataset, collate_scenes
from ebjd.diagnostic_bank import (
    BANK_SCHEMA, SEED_MANIFEST_SCHEMA, BankScene, atomic_json,
    canonical_window_key, save_scene, sha256_file, summarize_scenes,
)
from ebjd.sampling import predict


EXPECTED_CHECKPOINT_SHA256 = (
    "425988097793b639eb868fcc9ff28e41a3022c08598d13f396383da89268e1b1")
EXPECTED_CONFIG_SHA256 = (
    "a3800984c4876bec69caa62a587ab2db609d2dce1c7670995ddee0ecb6440278")
EXPECTED_MANIFEST_SHA256 = (
    "84c95cb431caf7214d1b85e8e7dbfeaf48c9fe301b1cda7f3201c3e8cf199c78")


def git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()


def complete_seed(path: Path, expected_scenes: int) -> bool:
    manifest = path / "manifest.json"
    if not manifest.is_file():
        return False
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if payload.get("status") != "COMPLETED":
            return False
        if int(payload.get("scene_count", -1)) != expected_scenes:
            return False
        return all(
            (manifest.parent / item["path"]).is_file()
            and sha256_file(manifest.parent / item["path"]) == item["sha256"]
            for item in payload.get("files", []))
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export frozen EBJD epoch-45 predictions as portable banks")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[2035, 2036, 2037, 2038, 2039])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    started = time.monotonic()
    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    if sha256_file(checkpoint_path) != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("EBJD epoch-45 checkpoint SHA256 mismatch")
    if sha256_file(config_path) != EXPECTED_CONFIG_SHA256:
        raise RuntimeError("frozen EBJD config SHA256 mismatch")
    config = load_config(config_path)
    manifest_path = Path(config["paths"]["test_manifest"]).resolve()
    if sha256_file(manifest_path) != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("EBJD test manifest SHA256 mismatch")
    dataset = SceneManifestDataset(manifest_path)
    scene_limit = len(dataset) if args.max_scenes is None else min(
        len(dataset), int(args.max_scenes))
    if scene_limit < 1:
        raise ValueError("--max-scenes must be positive")
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, collate_fn=collate_scenes,
        num_workers=0)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model = build_model(config, "none").to(device)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    parameter_hash_before = sha256_file(checkpoint_path)
    evaluation = config["evaluation"]
    worlds, steps = int(evaluation["worlds"]), int(evaluation["steps"])
    if (worlds, steps) != (20, 20):
        raise RuntimeError("diagnostic export is frozen to P20/DDIM20")
    output_root = Path(args.output_root).resolve() / "ebjd"
    output_root.mkdir(parents=True, exist_ok=True)
    repository = Path(__file__).resolve().parents[2]
    seed_manifests = []

    with torch.no_grad():
        for seed in args.seeds:
            seed_dir = output_root / f"seed_{seed}"
            if args.resume and complete_seed(seed_dir, scene_limit):
                print(f"EBJD seed {seed}: verified existing bank", flush=True)
                seed_manifests.append(seed_dir / "manifest.json")
                continue
            seed_dir.mkdir(parents=True, exist_ok=True)
            generator = torch.Generator(device=device).manual_seed(int(seed))
            files = []
            metric_scenes = []
            seed_start = time.monotonic()
            for scene_index, batch in enumerate(loader):
                if scene_index >= scene_limit:
                    break
                batch = batch.to(device)
                prediction, goal = predict(
                    model, batch.observed, batch.semantic_maps, batch.valid,
                    worlds, steps, generator)
                valid = batch.valid[0]
                predicted = prediction[0, :, valid].float().cpu().numpy()
                explicit_goal = goal[0, :, valid].float().cpu().numpy()
                goal_difference = float(np.max(np.abs(
                    predicted[:, :, -1] - explicit_goal)))
                if goal_difference > 1e-6:
                    raise RuntimeError(
                        f"literal goal differs from final trajectory: {goal_difference}")
                history = batch.observed[0, valid].float().cpu().numpy()
                future = batch.future[0, valid].float().cpu().numpy()
                frames = np.asarray(batch.frame_ids[0], dtype=np.int64)
                agents = np.asarray(batch.agent_ids[0], dtype=np.str_)
                source = str(batch.source_sequences[0])
                identity = canonical_window_key(source, frames)
                scene = BankScene(
                    predicted.astype(np.float32), future.astype(np.float32),
                    history.astype(np.float32), frames, agents, source,
                    str(batch.scene_ids[0]), identity,
                    float(batch.metadata["dt_seconds"]))
                relative = Path("scenes") / f"scene_{scene_index:06d}.npz"
                digest = save_scene(scene, seed_dir / relative)
                metric_scenes.append(scene)
                files.append({
                    "path": relative.as_posix(), "sha256": digest,
                    "identity": identity, "scene_id": scene.scene_id,
                    "source_sequence": source,
                    "frame_ids": frames.tolist(),
                    "agent_ids": agents.tolist(), "valid_agents": scene.agents,
                    "original_batch_index": scene_index,
                    "original_batch_size": 1,
                    "original_world_order": list(range(worlds)),
                    "goal_final_max_abs_m": goal_difference,
                })
                if (scene_index + 1) % 50 == 0:
                    print(
                        f"EBJD seed {seed}: {scene_index + 1}/{scene_limit}",
                        flush=True)
            metrics = summarize_scenes(metric_scenes)
            seed_payload = {
                "schema": SEED_MANIFEST_SCHEMA, "bank_schema": BANK_SCHEMA,
                "status": "COMPLETED", "model": "EBJD", "fold": "hotel",
                "training_seed": 3101, "inference_seed": int(seed),
                "worlds": worlds, "steps": steps, "scene_count": len(files),
                "agent_occurrences": int(sum(x["valid_agents"] for x in files)),
                "unit": "m", "dtype": "float32", "dt_seconds": 0.4,
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
                "config": str(config_path),
                "config_sha256": EXPECTED_CONFIG_SHA256,
                "source_manifest": str(manifest_path),
                "source_manifest_sha256": EXPECTED_MANIFEST_SHA256,
                "execution_commit": git_head(repository),
                "model_definition_commit": config["experiment"]["source_commit"],
                "inference_precision": "model FP32; sampler/geometry FP32",
                "autocast": "disabled",
                "input_information": "8-step trajectories + fixed 6-channel semantic crops",
                "sampler": "native EBJD deterministic DDIM20; B=1; P=20",
                "coordinate_conversion": "none; native metre-space output",
                "optimizer_updates": 0, "forward_calls": len(files),
                "elapsed_seconds": time.monotonic() - seed_start,
                "replay_metrics": metrics, "files": files,
            }
            atomic_json(seed_payload, seed_dir / "manifest.json")
            seed_manifests.append(seed_dir / "manifest.json")
            print(f"EBJD seed {seed}: {metrics}", flush=True)

    if sha256_file(checkpoint_path) != parameter_hash_before:
        raise RuntimeError("checkpoint bytes changed during read-only export")
    if device.type == "cuda":
        peak = int(torch.cuda.max_memory_allocated(device))
    else:
        peak = 0
    index = {
        "schema": "ebjd-trajectory-bank-index-v1", "model": "EBJD",
        "status": "COMPLETED", "optimizer_updates": 0,
        "seed_manifests": [{
            "path": str(path.relative_to(output_root)),
            "sha256": sha256_file(path)} for path in seed_manifests],
        "elapsed_seconds": time.monotonic() - started,
        "cuda_peak_memory_bytes": peak,
        "checkpoint_sha256_after": sha256_file(checkpoint_path),
    }
    atomic_json(index, output_root / "index.json")
    print(json.dumps(index, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
