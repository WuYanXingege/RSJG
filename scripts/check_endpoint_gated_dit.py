#!/usr/bin/env python3
"""Real-data finite-gradient/depth check; updates temporary models only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import traceback

import torch
import yaml

from ebjd.config import load_config
from ebjd.data import SceneManifestDataset
from ebjd.proposals import file_sha256
from check_acceleration_bridge import run_arm, relative_parameter_difference
from run_zara1_unified_coupling_bridge import atomic_json, make_bank


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-bank", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--updates", type=int, default=2)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--max-seconds", type=float, default=900)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite a validation output")
    if args.updates < 2 or args.max_seconds <= 0:
        raise ValueError("at least two updates are required to open zero-initialized gates")
    result = {"status": "RUNNING", "arms": [], "formal_optimizer_updates": 0,
              "temporary_optimizer_updates": 0, "device": args.device,
              "config": str(Path(args.config).resolve()),
              "config_sha256": file_sha256(args.config),
              "train_bank_sha256": file_sha256(args.train_bank)}
    atomic_json(output / "RESULTS.json", result)
    started = time.perf_counter()
    try:
        torch.set_num_threads(2)
        config = load_config(args.config)
        result["effective_precision"] = ("FP32" if args.device == "cpu" else
                                          config["training"]["mixed_precision"])
        dataset = SceneManifestDataset(config["paths"]["train_manifest"])
        bank = make_bank(args.train_bank,
            config["paths"]["matched_gdts_validation_checkpoint_sha256"],
            file_sha256(config["paths"]["train_manifest"]), "train")
        size = int(config["training"]["batch_scenes"])
        micros = args.updates * int(config["training"]["gradient_accumulation"])
        if size * micros > len(dataset):
            raise ValueError("insufficient unique windows for validation plan")
        order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(3101))
        window_plan = [order[i * size:(i + 1) * size].tolist() for i in range(micros)]
        profiles = yaml.safe_load((Path(__file__).resolve().parents[1] /
            "configs/research/endpoint_gated_dit_v1.yaml").read_text())["arms"]
        reference = None
        for arm in ("legacy6", "legacy6_repeat", "gated6", "gated8"):
            profile = profiles["legacy6" if arm == "legacy6_repeat" else arm]
            receipt, parameters = run_arm(
                config, dataset, bank, window_plan, arm, args.updates,
                torch.device(args.device), started + args.max_seconds,
                architecture_profile=profile)
            if reference is None:
                reference = parameters
            elif arm == "legacy6_repeat":
                receipt["parameter_relative_l2_vs_control"] = relative_parameter_difference(
                    reference, parameters)
                if args.device == "cpu" and receipt["parameter_relative_l2_vs_control"] != 0:
                    raise RuntimeError("CPU legacy repeat changed parameters")
            if result["arms"] and receipt["input_digests"] != result["arms"][0]["input_digests"]:
                raise RuntimeError("sample, augmentation or noise changed across backbone arms")
            result["arms"].append(receipt)
            result["temporary_optimizer_updates"] += args.updates
            atomic_json(output / "RESULTS.json", result)
            print(json.dumps({"arm": arm, "status": "PASS", "updates": args.updates}), flush=True)
        result["status"] = "PASS_INTERFACE_ONLY_NO_ACCURACY_OR_SPEED_CLAIM"
    except BaseException as error:
        result.update(status="FAILED", reason=f"{type(error).__name__}: {error}",
                      traceback=traceback.format_exc())
        raise
    finally:
        result["wall_seconds"] = time.perf_counter() - started
        result["gpu_wall_seconds"] = result["wall_seconds"] if args.device == "cuda" else 0
        atomic_json(output / "RESULTS.json", result)


if __name__ == "__main__":
    main()
