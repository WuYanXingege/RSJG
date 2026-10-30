#!/usr/bin/env python3
"""Extract the exact matched-GDTS Goal U-Net into a standalone artifact."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import torch

from ebjd.goal_provider import NativeGoalProvider, file_sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--fold", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    source = Path(args.checkpoint).resolve()
    if file_sha256(source) != args.expected_sha256:
        raise ValueError("matched checkpoint SHA256 mismatch")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    full = payload["model_state_dict"]
    prefix = "goal_module."
    state = {key[len(prefix):]: value.detach().cpu()
             for key, value in full.items() if key.startswith(prefix)}
    if len(state) != 46:
        raise RuntimeError(f"expected 46 Goal U-Net tensors, got {len(state)}")
    output = Path(args.output_dir).resolve(); output.mkdir(parents=True, exist_ok=True)
    weights = output / "matched_goal_provider.pt"
    temporary = weights.with_suffix(".pt.tmp")
    torch.save({"state_dict": state}, temporary); os.replace(temporary, weights)
    manifest = {
        "schema": "ebjd-native-goal-provider-v1", "fold": args.fold,
        "weights": weights.name, "weights_sha256": file_sha256(weights),
        "source_checkpoint": str(source),
        "source_checkpoint_sha256": args.expected_sha256,
        "source_role": "matched_gdts_updated_goal_unet",
        "parameter_count": 613772, "tensor_count": 46,
        "input_channels": {"semantic": 6, "observed_heatmaps": 8},
        "ttst": {"multinomial_samples": 1000, "clusters": 20,
                 "output_count": 20, "ordered": True},
        "down_factor": 8, "coordinate_unit": "m",
        "full_gdts_trajectory_network_required": False,
    }
    manifest_path = output / "provider_manifest.json"
    fd, name = tempfile.mkstemp(prefix=manifest_path.name + ".", dir=output)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True); handle.write("\n")
    os.replace(name, manifest_path)
    NativeGoalProvider.from_artifact(manifest_path)
    print(json.dumps({"status": "COMPLETE", "manifest": str(manifest_path),
                      "weights_sha256": manifest["weights_sha256"]}))


if __name__ == "__main__":
    main()
