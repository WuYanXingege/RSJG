"""Command line P20 evaluation entry point: ``python -m ebjd.evaluate``."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import build_model, load_config
from .data import SceneManifestDataset, collate_scenes
from .metrics import MetricAccumulator
from .sampling import predict


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate independent EBJD")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--ablation", default="none")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    path = config.get("paths", {}).get(f"{args.split}_manifest")
    if not path:
        raise ValueError(f"paths.{args.split}_manifest is required")
    dataset = SceneManifestDataset(path)
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        collate_fn=collate_scenes, num_workers=0)
    device = torch.device(args.device)
    model = build_model(config, args.ablation).to(device).float()
    checkpoint_sha256_before = sha256(args.checkpoint)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    model.eval()
    evaluation = config.get("evaluation", {})
    worlds, steps = int(evaluation.get("worlds", 20)), int(evaluation.get("steps", 20))
    results = []
    with torch.no_grad(), torch.autocast(
            device_type=device.type, enabled=False):
        for seed in evaluation.get("inference_seeds", [2035, 2036, 2037, 2038, 2039]):
            generator = torch.Generator(device=device).manual_seed(int(seed))
            accumulator = MetricAccumulator()
            for batch in loader:
                batch = batch.to(device)
                prediction, _ = predict(
                    model, batch.observed, batch.semantic_maps, batch.valid,
                    worlds, steps, generator)
                accumulator.update(prediction, batch.future, batch.valid)
            means = accumulator.as_dict()
            nonfinite = {
                name: value for name, value in means.items()
                if isinstance(value, (int, float)) and not math.isfinite(float(value))}
            if nonfinite:
                raise FloatingPointError(
                    f"non-finite formal metrics for seed {seed}: {nonfinite}")
            if means["scene_count"] != len(dataset):
                raise RuntimeError(
                    f"formal evaluation skipped scenes for seed {seed}: "
                    f"{means['scene_count']} != {len(dataset)}")
            expected_agents = int(dataset.manifest["statistics"]["agents"])
            if means["agent_count"] != expected_agents:
                raise RuntimeError(
                    f"formal evaluation skipped agent occurrences for seed {seed}: "
                    f"{means['agent_count']} != {expected_agents}")
            results.append({"seed": int(seed), "metrics": means})
    metric_names = (
        "minADE", "minFDE", "JADE", "JFDE",
        "scene_weighted_minADE", "scene_weighted_minFDE",
        "coordination_gap_ADE", "coordination_gap_FDE")
    means = {
        name: float(np.mean([x["metrics"][name] for x in results]))
        for name in metric_names}
    sample_standard_deviation = {
        name: float(statistics.stdev([x["metrics"][name] for x in results]))
        for name in metric_names}
    manifest_payload = dataset.manifest
    checkpoint_sha256_after = sha256(args.checkpoint)
    if checkpoint_sha256_after != checkpoint_sha256_before:
        raise RuntimeError("checkpoint bytes changed during formal evaluation")
    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()), "split": args.split,
        "checkpoint_sha256": checkpoint_sha256_before,
        "checkpoint_unchanged_during_evaluation": True,
        "checkpoint_epoch": int(payload["epoch"]),
        "manifest": str(Path(path).resolve()),
        "manifest_sha256": sha256(path),
        "scene_count": int(manifest_payload["statistics"]["scenes"]),
        "agent_occurrence_count": int(manifest_payload["statistics"]["agents"]),
        "validation_test_identity": evaluation["validation_test_identity"],
        "independent_test": False,
        "inference_batch_size": 1, "model_dtype": "fp32",
        "autocast_enabled": False, "ddim_eta": float(config["diffusion"]["ddim_eta"]),
        "per_agent_reordering": bool(evaluation["per_agent_reordering"]),
        "worlds": worlds, "steps": steps, "per_seed": results,
        "mean": means, "sample_standard_deviation": sample_standard_deviation,
    }
    output = Path(args.output)
    atomic_json(summary, output)
    print(json.dumps(summary["mean"], sort_keys=True))


if __name__ == "__main__":
    main()
