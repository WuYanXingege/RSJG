"""Command line P20 evaluation entry point: ``python -m ebjd.evaluate``."""

from __future__ import annotations

import argparse
import hashlib
import json
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


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
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
    parser = argparse.ArgumentParser(
        description="Evaluate EBJD FULL with Raw decoding")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--checkpoint-sha256", required=True,
        help="required frozen checkpoint identity")
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--ablation", choices=("none",), default="none")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    actual_checkpoint_sha256 = _sha256(args.checkpoint)
    if actual_checkpoint_sha256 != args.checkpoint_sha256:
        raise ValueError(
            "checkpoint SHA256 mismatch: " + actual_checkpoint_sha256)
    config = load_config(args.config)
    path = config.get("paths", {}).get(f"{args.split}_manifest")
    if not path:
        raise ValueError(f"paths.{args.split}_manifest is required")
    loader = DataLoader(SceneManifestDataset(path), batch_size=1, shuffle=False,
                        collate_fn=collate_scenes, num_workers=0)
    device = torch.device(args.device)
    model = build_model(config, args.ablation).to(device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    evaluation = config.get("evaluation", {})
    worlds, steps = int(evaluation.get("worlds", 20)), int(evaluation.get("steps", 20))
    results = []
    with torch.no_grad():
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
            results.append({"seed": int(seed), "metrics": means})
    metric_names = (
        "minADE", "minFDE", "JADE", "JFDE",
        "scene_weighted_minADE", "scene_weighted_minFDE",
        "coordination_gap_ADE", "coordination_gap_FDE")
    summary = {
        "schema": "ebjd-paper-full-evaluation-v1",
        "method": "Endpoint--Bridge Joint Diffusion",
        "architecture": "FULL",
        "output_semantics": "Raw",
        "checkpoint": str(Path(args.checkpoint).resolve()), "split": args.split,
        "checkpoint_sha256": actual_checkpoint_sha256,
        "config_sha256": _sha256(args.config),
        "manifest_sha256": _sha256(path),
        "validation_test_identity": evaluation["validation_test_identity"],
        "independent_test": False,
        "worlds": worlds, "steps": steps, "per_seed": results,
        "mean": {
            name: float(np.mean([x["metrics"][name] for x in results]))
            for name in metric_names},
        "sample_standard_deviation": {
            name: float(statistics.stdev(
                [x["metrics"][name] for x in results]))
            if len(results) > 1 else 0.0
            for name in metric_names},
    }
    output = Path(args.output)
    _atomic_json(output, summary)
    print(json.dumps(summary["mean"], sort_keys=True))


if __name__ == "__main__":
    main()
