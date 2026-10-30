#!/usr/bin/env python3
"""Small GPU A/C parity replay for the formal standalone provider."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

from ebjd.goal_provider import NativeGoalProvider
from export_zara1_endpoint_banks import configure_legacy_evaluator


def goal_state(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    full = payload["model_state_dict"]
    state = {key[len("goal_module."):]: value for key, value in full.items()
             if key.startswith("goal_module.")}
    if len(state) != 46:
        raise RuntimeError("matched checkpoint lacks the exact Goal U-Net state")
    return state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--windows", type=int, default=2)
    args = parser.parse_args()
    specs = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    report = {"schema": "ebjd-goal-provider-gpu-parity-v1", "device": "cuda",
              "dtype": "bfloat16_autocast", "coordinate_tolerance_m": 1e-6,
              "folds": {}}
    for fold, item in specs.items():
        namespace = argparse.Namespace(
            recipe=item["recipe"], runtime_root=item["runtime_root"],
            scene_cache_root=item["scene_cache_root"],
            output_dir=str(Path(args.output).parent / ("runtime_" + fold)),
            checkpoint=item["checkpoint"])
        evaluator = configure_legacy_evaluator(namespace)
        provider = NativeGoalProvider(
            goal_state(Path(item["checkpoint"])),
            {"fold": fold, "source": "matched_gdts_parity_reference"},
        ).to(evaluator.device).eval()
        rows = []
        for index, (batch_data, batch_id) in enumerate(evaluator.data_loaders["valid"]):
            if index >= args.windows:
                break
            inputs, _ = evaluator.net.prepare_inputs(batch_data, batch_id)
            for seed in (3101, 2035):
                random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                python_state, numpy_state = random.getstate(), np.random.get_state()
                cpu_state, cuda_state = torch.get_rng_state(), torch.cuda.get_rng_state_all()
                captured = []
                hook = evaluator.net.goal_module.register_forward_hook(
                    lambda _m, _i, output: captured.append(output.detach().clone()))
                with evaluator._autocast_context():
                    _, reference = evaluator.net._legacy_encode(inputs, if_test=True)
                logits_a = captured.pop(); hook.remove()
                random.setstate(python_state); np.random.set_state(numpy_state)
                torch.set_rng_state(cpu_state); torch.cuda.set_rng_state_all(cuda_state)
                with evaluator._autocast_context():
                    logits_c, map_c = provider.predict_prepared(
                        inputs["tensor_image"], inputs["input_traj_maps"])
                map_a = reference["goal_point"][:20]
                world_a = inputs["scene"].make_world_coord_torch(map_a * 8).float()
                world_c = inputs["scene"].make_world_coord_torch(map_c * 8).float()
                rows.append({
                    "window": index, "seed": seed,
                    "agent_count": int(map_a.shape[1]),
                    "logits_max_abs": float((logits_a.float() - logits_c.float()).abs().max()),
                    "ordered_world_max_euclidean_m": float(torch.linalg.vector_norm(
                        world_a - world_c, dim=-1).max()),
                    "ordered_exact": bool(torch.equal(world_a, world_c)),
                })
        passed = bool(rows) and all(
            row["ordered_world_max_euclidean_m"] <= 1e-6
            for row in rows)
        report["folds"][fold] = {"status": "PASS" if passed else "FAIL",
                                         "comparisons": rows}
        if not passed:
            Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
            raise RuntimeError(f"{fold} standalone provider parity failed")
    report["status"] = "PASS"
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "output": args.output}))


if __name__ == "__main__":
    main()
