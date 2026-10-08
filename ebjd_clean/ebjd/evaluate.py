"""Command line P20 evaluation entry point: ``python -m ebjd.evaluate``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import build_model, load_config
from .data import SceneManifestDataset, collate_scenes
from .metrics import MetricAccumulator
from .sampling import predict


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
    loader = DataLoader(SceneManifestDataset(path), batch_size=1, shuffle=False,
                        collate_fn=collate_scenes, num_workers=0)
    device = torch.device(args.device)
    model = build_model(config, args.ablation).to(device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
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
    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()), "split": args.split,
        "validation_test_identity": evaluation["validation_test_identity"],
        "independent_test": False,
        "worlds": worlds, "steps": steps, "per_seed": results,
        "mean": {name: float(np.mean([x["metrics"][name] for x in results]))
                 for name in (
                     "minADE", "minFDE", "JADE", "JFDE",
                     "scene_weighted_minADE", "scene_weighted_minFDE",
                     "coordination_gap_ADE", "coordination_gap_FDE")},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary["mean"], sort_keys=True))


if __name__ == "__main__":
    main()
