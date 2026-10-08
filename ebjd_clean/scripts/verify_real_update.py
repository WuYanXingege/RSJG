"""Run bounded, identity-recorded EBJD updates/inference on exported real scenes."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ebjd.config import build_model, config_hash, load_config, trainer_options
from ebjd.data import SceneManifestDataset, collate_scenes
from ebjd.metrics import MetricAccumulator
from ebjd.sampling import differentiable_sample
from ebjd.train import fit_training_scales
from ebjd.trainer import EBJDTrainer


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def choose_scene(dataset: SceneManifestDataset, policy: str):
    counts = [dataset[index]["observed"].shape[0] for index in range(len(dataset))]
    if policy == "max":
        index = max(range(len(counts)), key=counts.__getitem__)
    elif policy == "multi":
        eligible = [index for index, count in enumerate(counts) if 1 < count <= 8]
        index = max(eligible, key=counts.__getitem__) if eligible else max(
            range(len(counts)), key=counts.__getitem__)
    else:
        index = 0
    return dataset[index], counts


def parameter_snapshot(model) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.named_parameters()}


def parameter_change(before: dict[str, torch.Tensor], model) -> dict:
    squared = maximum = 0.0
    changed_tensors = changed_values = 0
    for name, value in model.named_parameters():
        delta = value.detach().cpu().float() - before[name].float()
        if torch.count_nonzero(delta):
            changed_tensors += 1
            changed_values += int(torch.count_nonzero(delta))
        squared += float(delta.double().square().sum())
        maximum = max(maximum, float(delta.abs().max()))
    return {
        "l2": squared ** 0.5, "max_abs": maximum,
        "changed_parameter_tensors": changed_tensors,
        "changed_parameter_values": changed_values,
    }


def environment(device: torch.device) -> dict:
    result = {
        "python_source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True,
            text=True).stdout.strip(),
        "torch": torch.__version__, "numpy": np.__version__, "device": str(device),
    }
    if device.type == "cuda":
        result.update({
            "device_name": torch.cuda.get_device_name(device),
            "device_total_bytes": torch.cuda.get_device_properties(device).total_memory,
            "bf16_supported": torch.cuda.is_bf16_supported(),
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--mode", choices=("normal", "rollout", "inference"), required=True)
    parser.add_argument("--scene-policy", choices=("first", "multi", "max"), default="max")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--world-chunk", type=int, default=2)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.world_chunk <= 0:
        raise ValueError("world-chunk must be positive")
    config_path, manifest_path = Path(args.config).resolve(), Path(args.manifest).resolve()
    config = load_config(config_path)
    device = torch.device(args.device)
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    dataset = SceneManifestDataset(manifest_path)
    sample, all_counts = choose_scene(dataset, args.scene_policy)
    batch = collate_scenes([sample]).to(device)
    model = build_model(config).to(device)
    scale_loader = DataLoader(
        dataset, batch_size=1, shuffle=False, collate_fn=collate_scenes, num_workers=0)
    fitted_scales = fit_training_scales(model, scale_loader)
    identity = {
        "mode": args.mode, "fold": config["experiment"]["fold"], "seed": seed,
        "ablation": "none", "scene_id": sample["scene_id"],
        "window_key": sample["scene_id"].split(":", 2)[-1],
        "agent_ids": sample["agent_ids"], "frame_ids": sample["frame_ids"],
        "source_sequence": sample["source_sequence"],
        "manifest": str(manifest_path), "manifest_sha256": sha256(manifest_path),
        "config": str(config_path), "config_sha256": sha256(config_path),
        "resolved_config_hash": config_hash(config),
        "initialization": {
            "path": config["paths"]["accepted_fold_matched_gdts_unet"],
            "sha256": config["paths"]["accepted_fold_matched_gdts_unet_sha256"],
        },
    }
    result = {
        "schema": "ebjd-real-verification-v1", "identity": identity,
        "environment": environment(device),
        "data": {
            "selected_N": int(batch.valid.sum()), "observed_N_values": all_counts,
            "observed_max_N": max(all_counts), "map_shape": list(batch.semantic_maps.shape),
            "fitted_goal_scale_m": fitted_scales[0] if fitted_scales else None,
            "fitted_bridge_scale_m": fitted_scales[1] if fitted_scales else None,
            "encoder_goal_scale_m": float(model.encoder.goal_scale.cpu()),
        },
    }

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    try:
        if args.mode in ("normal", "rollout"):
            options = trainer_options(config)
            if args.mode == "normal":
                options.rollout_start_epoch = 10**9
            else:
                options.rollout_start_epoch = 1
                options.rollout_ramp_end_epoch = 1
                options.rollout_every = 1
                options.rollout_worlds = 4
                options.rollout_steps = 20
            trainer = EBJDTrainer(model, options, config, identity)
            before = parameter_snapshot(model)
            logs = trainer.train_step([batch], epoch=1)
            result["update"] = {
                "logs": logs, "parameter_change": parameter_change(before, model),
                "finite_logs": all(np.isfinite(value) for value in logs.values()),
                "constraint_tolerance": 1e-6,
                "constraints_satisfied": (
                    args.mode == "normal" or (
                        logs["applied_dot_marginal_ade"] >= -1e-6
                        and logs["applied_dot_marginal_fde"] >= -1e-6)),
            }
            result["passed"] = bool(
                result["update"]["finite_logs"]
                and result["update"]["parameter_change"]["l2"] > 0
                and result["update"]["constraints_satisfied"]
                and (args.mode == "normal" or logs["rollout_update"] == 1))
        else:
            model.eval()
            predictions = []
            generator = torch.Generator(device=device).manual_seed(2035)
            with torch.no_grad():
                with torch.autocast(
                    device_type=device.type, dtype=torch.bfloat16,
                    enabled=device.type == "cuda" and torch.cuda.is_bf16_supported()):
                    context = model.encode_context(
                        batch.observed, batch.semantic_maps, batch.valid)
                    for offset in range(0, 20, args.world_chunk):
                        worlds = min(args.world_chunk, 20 - offset)
                        noise = torch.randn(
                            1, worlds, batch.observed.shape[1], 12, 2,
                            device=device, generator=generator)
                        trajectory, _, _ = differentiable_sample(
                            model, context, noise, steps=20, checkpoint_steps=False)
                        predictions.append(trajectory.float().cpu())
            prediction = torch.cat(predictions, dim=1)
            accumulator = MetricAccumulator()
            accumulator.update(prediction, batch.future.cpu(), batch.valid.cpu())
            result["inference"] = {
                "worlds": 20, "steps": 20, "world_chunk": args.world_chunk,
                "prediction_shape": list(prediction.shape),
                "finite": bool(torch.isfinite(prediction).all()),
                "metrics": accumulator.as_dict(),
            }
            result["passed"] = bool(
                result["inference"]["finite"] and prediction.shape[1] == 20)
    except torch.cuda.OutOfMemoryError as error:
        result["passed"] = False
        result["failure"] = {"type": type(error).__name__, "message": str(error)}
        torch.cuda.empty_cache()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    result["resource"] = {"elapsed_seconds": time.perf_counter() - started}
    if device.type == "cuda":
        result["resource"].update({
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "allocated_after_bytes": torch.cuda.memory_allocated(device),
        })
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    if not result.get("passed"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
