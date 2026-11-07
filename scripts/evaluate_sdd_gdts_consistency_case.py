#!/usr/bin/env python3
"""Freeze-evaluate one SDD GDTS checkpoint under one cache layout/seed.

This audit entry point deliberately keeps the deployed independent GDTS model
fixed while selecting either the legacy independent-trajectory cache
(`packed`) or the synchronized scene-window cache (`scene`) at DataLoader
construction time.  It also recomputes pixel minADE/minFDE directly from the
predicted tensors and persists per-agent errors for CPU verification.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
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


def _scalar(value):
    while isinstance(value, (tuple, list)) and len(value) == 1:
        value = value[0]
    if torch.is_tensor(value):
        value = value.detach().cpu().reshape(-1)[0].item()
    elif isinstance(value, np.ndarray):
        value = value.reshape(-1)[0].item()
    return value


def _sequence(value, count: int) -> list:
    if torch.is_tensor(value):
        flat = value.detach().cpu().reshape(-1).tolist()
    elif isinstance(value, np.ndarray):
        flat = value.reshape(-1).tolist()
    elif isinstance(value, (tuple, list)):
        flat = [_scalar(item) for item in value]
    else:
        flat = [value]
    if len(flat) == 1 and count != 1:
        flat = flat * count
    if len(flat) != count:
        raise RuntimeError(
            f"identity field has {len(flat)} values for {count} agents")
    return flat


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    fields = [
        "layout", "seed", "batch_index", "local_agent_index",
        "source_sequence", "starting_frame", "agent_id",
        "manifest_scene_id", "valid", "original_minADE",
        "original_minFDE", "reference_pixel_minADE",
        "reference_pixel_minFDE",
    ]
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--layout", choices=("packed", "scene"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    recipe_path = Path(args.recipe).resolve()
    recipe = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
    legacy = Path(recipe["legacy_gdts"]["root"]).resolve()
    runtime = Path(args.runtime_root).resolve()
    cache_root = Path(args.cache_root).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    manifest = Path(args.manifest).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "RESULT.json"
    if result_path.exists():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if (existing.get("status") == "COMPLETE" and
                existing.get("checkpoint_sha256") == args.checkpoint_sha256 and
                existing.get("manifest_sha256") == sha256(manifest) and
                existing.get("layout") == args.layout and
                existing.get("seed") == args.seed):
            print(f"Reusing complete case {args.layout}/{args.seed}", flush=True)
            return
        raise RuntimeError(f"conflicting existing result: {result_path}")
    if sha256(checkpoint) != args.checkpoint_sha256:
        raise ValueError("checkpoint SHA256 mismatch")

    sys.path.insert(0, str(legacy))
    os.chdir(runtime)
    from src.data_src import data_utils
    source_data_root = str(Path(recipe["source_data_root"]).resolve())
    data_utils.get_data_location = lambda: source_data_root
    from src.data_src.dataset_src.dataset_sdd import Dataset_sdd
    from src.data_src.scene_src.scene_sdd import Scene_sdd
    original_dataset_init = Dataset_sdd.__init__

    def exact_jmm_init(self, verbose=False, load_visual_data=True):
        original_dataset_init(
            self, verbose=verbose, load_visual_data=load_visual_data)
        self.train_scenes.discard("hyang_7")
        self.train_scenes.add("nexus_2")
        self.scenes.pop("hyang_7", None)
        self.scenes["nexus_2"] = Scene_sdd(
            "nexus_2", verbose, load_visual_data=load_visual_data)

    Dataset_sdd.__init__ = exact_jmm_init
    from src.metrics import compute_metric_mask
    from src.models.model import GDTS
    from src.parser import check_and_add_additional_args, get_parser
    from src.trainer import trainer
    from src.utils import isolated_random_seed

    # Cache selection happens while trainer constructs its DataLoaders.
    # The deployed network remains the same independent/baseline model below.
    loader_goal_type = "independent" if args.layout == "packed" else "social"
    loader_stage = "baseline" if args.layout == "packed" else "finetune"
    cli = [
        "--dataset", "sdd", "--test_set", "sdd", "--phase", "test",
        "--device", "cuda:0", "--batch_size", "64", "--num_workers", "0",
        "--down_factor", "8", "--skip_ts_window", "1",
        "--goal_model_type", loader_goal_type,
        "--training_stage", loader_stage,
        "--data_augmentation", "False",
        "--shuffle_train_batches", "False",
        "--shuffle_test_batches", "False",
        "--num_samples", "20", "--ddpm_step", "100",
        "--ddim_step", "20", "--trunk_stage_step", "30",
        "--reproducibility", "True", "--seed", str(recipe["seed"]),
        "--batch_cache_root", str(cache_root),
        "--compress_batch_cache", "True", "--use_wandb", "False",
    ]
    parsed = check_and_add_additional_args(get_parser().parse_args(cli))
    parsed.model_dir = str(output / "runtime")
    parsed.save_dir = parsed.model_dir
    evaluator = trainer(parsed)
    parsed.goal_model_type = "independent"
    parsed.training_stage = "baseline"
    parsed.use_scene_latent = False
    parsed.use_dynamic_relation = False
    parsed.use_joint_energy = False
    parsed.use_dependency_corrector = False
    parsed.jdv2_active = False
    parsed.joint_goal_enabled = False
    parsed.best_metric = "auto"
    evaluator.args = parsed
    evaluator.net = GDTS(parsed, evaluator.device).to(evaluator.device)
    evaluator._pending_training_state = None
    evaluator._load_state_file(str(checkpoint), weights_only=True)
    evaluator.net.eval()

    rows: list[dict] = []
    error_chunks: list[np.ndarray] = []
    original_ade: list[float] = []
    original_fde: list[float] = []
    reference_ade: list[float] = []
    reference_fde: list[float] = []
    joint_ade: list[float] = []
    joint_fde: list[float] = []
    actual_k: set[int] = set()
    identity_digest = hashlib.sha256()
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(evaluator.device)
    with torch.no_grad(), isolated_random_seed(args.seed, use_cuda=True):
        for batch_index, (batch_data, batch_id) in enumerate(
                evaluator.data_loaders["valid"]):
            inputs, seq_list = evaluator.net.prepare_inputs(batch_data, batch_id)
            predictions, auxiliary = evaluator.net.forward(inputs, if_test=True)
            mask = compute_metric_mask(seq_list).bool()
            actual_k.add(int(predictions.shape[0]))

            original_batch_ade = evaluator.net.compute_model_metrics(
                "minADE@K", predictions, mask, auxiliary, inputs,
                obs_length=parsed.obs_length)
            original_batch_fde = evaluator.net.compute_model_metrics(
                "minFDE@K", predictions, mask, auxiliary, inputs,
                obs_length=parsed.obs_length)
            joint_ade.extend(evaluator.net.compute_model_metrics(
                "JADE", predictions, mask, auxiliary, inputs,
                obs_length=parsed.obs_length))
            joint_fde.extend(evaluator.net.compute_model_metrics(
                "JFDE", predictions, mask, auxiliary, inputs,
                obs_length=parsed.obs_length))

            prediction_px = predictions.detach().float() * parsed.down_factor
            target_px = inputs["x_augmented"][:, :, 6:8].detach().float()
            target_px = target_px * parsed.down_factor
            errors = torch.linalg.vector_norm(
                prediction_px[:, parsed.obs_length:] -
                target_px[None, parsed.obs_length:], dim=-1)
            valid_errors = errors[:, :, mask]
            reference_batch_ade = valid_errors.mean(dim=1).min(dim=0).values
            reference_batch_fde = valid_errors[:, -1].min(dim=0).values
            error_chunks.append(valid_errors.permute(2, 0, 1).cpu().numpy())

            count = int(mask.numel())
            source = str(_scalar(batch_id.get(
                "scene_name", batch_id.get("data_file_path", "unknown"))))
            source = Path(source).stem
            starts = _sequence(batch_id.get("starting_frames", -1), count)
            agents = _sequence(batch_id.get("agent_ids", -1), count)
            manifest_scene = str(_scalar(batch_id.get("manifest_scene_id", "")))
            valid_indices = mask.nonzero(as_tuple=False).reshape(-1).tolist()
            for result_index, local_index in enumerate(valid_indices):
                identity = (
                    source, str(int(starts[local_index])),
                    str(agents[local_index]))
                identity_digest.update(("|".join(identity) + "\n").encode())
                rows.append({
                    "layout": args.layout,
                    "seed": args.seed,
                    "batch_index": batch_index,
                    "local_agent_index": local_index,
                    "source_sequence": source,
                    "starting_frame": int(starts[local_index]),
                    "agent_id": str(agents[local_index]),
                    "manifest_scene_id": manifest_scene,
                    "valid": True,
                    "original_minADE": float(original_batch_ade[result_index]),
                    "original_minFDE": float(original_batch_fde[result_index]),
                    "reference_pixel_minADE": float(
                        reference_batch_ade[result_index].cpu()),
                    "reference_pixel_minFDE": float(
                        reference_batch_fde[result_index].cpu()),
                })
            original_ade.extend(map(float, original_batch_ade))
            original_fde.extend(map(float, original_batch_fde))
            reference_ade.extend(reference_batch_ade.cpu().tolist())
            reference_fde.extend(reference_batch_fde.cpu().tolist())
            del batch_data, batch_id, inputs, seq_list, predictions, auxiliary

    if len(actual_k) != 1:
        raise RuntimeError(f"candidate count changed across batches: {actual_k}")
    elapsed = time.perf_counter() - started
    _write_rows(output / "PER_AGENT_ERRORS.csv", rows)
    all_errors = np.concatenate(error_chunks, axis=0)
    temporary = output / "ERROR_BANK.tmp.npz"
    np.savez_compressed(temporary, displacement_errors=all_errors)
    os.replace(temporary, output / "ERROR_BANK.npz")
    original_ade_array = np.asarray(original_ade, dtype=np.float64)
    original_fde_array = np.asarray(original_fde, dtype=np.float64)
    reference_ade_array = np.asarray(reference_ade, dtype=np.float64)
    reference_fde_array = np.asarray(reference_fde, dtype=np.float64)
    payload = {
        "schema": "sdd-gdts-eval-consistency-case-v1",
        "status": "COMPLETE",
        "layout": args.layout,
        "seed": args.seed,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": args.checkpoint_sha256,
        "manifest": str(manifest),
        "manifest_sha256": sha256(manifest),
        "recipe": str(recipe_path),
        "recipe_sha256": sha256(recipe_path),
        "cache_root": str(cache_root),
        "loader_goal_model_type": loader_goal_type,
        "loader_training_stage": loader_stage,
        "active_goal_model_type": evaluator.net.active_goal_model_type,
        "active_training_stage": parsed.training_stage,
        "physical_batch_count": len(evaluator.data_loaders["valid"]),
        "valid_agent_occurrence_count": len(rows),
        "actual_candidate_counts": sorted(actual_k),
        "sampling": {
            "num_samples": int(parsed.num_samples),
            "ddpm_step": int(parsed.ddpm_step),
            "ddim_step": int(parsed.ddim_step),
            "trunk_stage_step": int(parsed.trunk_stage_step),
            "down_factor": int(parsed.down_factor),
            "precision": "fp32",
            "net_eval": not evaluator.net.training,
            "if_test": True,
        },
        "metrics": {
            "original_minADE": float(original_ade_array.mean()),
            "original_minFDE": float(original_fde_array.mean()),
            "reference_pixel_minADE": float(reference_ade_array.mean()),
            "reference_pixel_minFDE": float(reference_fde_array.mean()),
            "original_JADE": float(np.mean(joint_ade)),
            "original_JFDE": float(np.mean(joint_fde)),
        },
        "reference_agreement": {
            "minADE_max_abs": float(np.max(np.abs(
                original_ade_array - reference_ade_array))),
            "minFDE_max_abs": float(np.max(np.abs(
                original_fde_array - reference_fde_array))),
        },
        "identity_stream_sha256": identity_digest.hexdigest(),
        "elapsed_seconds": elapsed,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(
            evaluator.device)),
        "per_agent_errors": str(output / "PER_AGENT_ERRORS.csv"),
        "error_bank": str(output / "ERROR_BANK.npz"),
    }
    atomic_json(result_path, payload)
    print(json.dumps(payload["metrics"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
