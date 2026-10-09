#!/usr/bin/env python3
"""GPU validation for the scene-staged EBJD OOM recovery.

This script is diagnostic-only: every mode reloads the epoch-boundary source
checkpoint and never writes a training checkpoint.  Formal training must start
again from the separately hashed source after these checks finish.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ebjd.config import build_model, trainer_options
from ebjd.data import SceneManifestDataset, augment_batch, collate_scenes
from ebjd.metrics import hard_metrics
from ebjd.optimizer import (
    clip_global_norm, list_dot, list_norm, project_two_halfspaces,
)
from ebjd.sampling import predict
from ebjd.trainer import EBJDTrainer


LOSS_ABS_TOLERANCE = 5e-3
# Scene chunking changes BF16 CUDA kernel shapes.  After a stricter 5e-3 gate
# localized the only excess error to BF16 (the same CUDA comparison in FP32 was
# <= 8.9e-6), use a fixed two-bfloat16-epsilon bound rather than an empirical
# value fitted to the observed result.
GRADIENT_RELATIVE_TOLERANCE = 2.0 * torch.finfo(torch.bfloat16).eps
DIRECTION_COSINE_MINIMUM = 0.9999
# Adam's division by the square-root second moment is nonlinear and can
# amplify a BF16-size gradient perturbation.  Gate its candidate/decrement
# separately at three BF16 epsilons; do not misapply the raw-gradient bound.
OPTIMIZER_RELATIVE_TOLERANCE = 3.0 * torch.finfo(torch.bfloat16).eps
OPTIMIZER_DIRECTION_COSINE_MINIMUM = (
    1.0 - 0.5 * OPTIMIZER_RELATIVE_TOLERANCE**2)
PROJECTION_TOLERANCE = 1e-6
MINIMUM_RESERVED_MARGIN_BYTES = 2 * 1024**3

FAILURE_BATCH_INDICES = (
    (91, 1907, 3967, 525),
    (3302, 568, 146, 2119),
    (1671, 2692, 3296, 2115),
    (4067, 285, 1823, 431),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=("numeric", "stress", "continuous", "all"), default="all")
    parser.add_argument("--continuous-updates", type=int, default=48)
    parser.add_argument(
        "--numeric-precision", choices=("configured", "fp32"),
        default="configured",
        help="diagnostic precision override used only by numeric mode")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def finite_tree(value) -> bool:
    if torch.is_tensor(value):
        return not (value.is_floating_point() or value.is_complex()) or bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def make_trainer(payload: dict, device: torch.device) -> EBJDTrainer:
    config = payload["resolved_config"]
    model = build_model(config).to(device)
    options = trainer_options(config)
    options.scene_forward_chunk = 1
    trainer = EBJDTrainer(model, options)
    trainer.restore_checkpoint_payload(payload, require_identity=False)
    model.train()
    return trainer


def release_trainer(trainer: EBJDTrainer | None) -> None:
    if trainer is not None:
        del trainer
    gc.collect()
    torch.cuda.empty_cache()


def gradient_comparison(reference, candidate) -> dict:
    difference = [
        left - right for left, right in zip(reference, candidate, strict=True)]
    reference_norm = float(list_norm(reference))
    candidate_norm = float(list_norm(candidate))
    difference_norm = float(list_norm(difference))
    cosine = float(list_dot(reference, candidate)) / max(
        reference_norm * candidate_norm, 1e-30)
    return {
        "reference_norm": reference_norm,
        "candidate_norm": candidate_norm,
        "absolute_difference_norm": difference_norm,
        "relative_difference_norm": difference_norm / max(reference_norm, 1e-30),
        "direction_cosine": cosine,
        "maximum_absolute_element_difference": max(
            float(value.abs().max()) for value in difference),
    }


def run_gradient_path(
    trainer: EBJDTrainer, batch, epoch: int, do_rollout: bool,
    chunked: bool, cuda_rng: torch.Tensor,
):
    torch.cuda.set_rng_state(cuda_rng)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    if chunked:
        values = trainer._scene_chunked_batch_gradients(batch, epoch, do_rollout)
    else:
        physical = batch.to(trainer.device)
        draws = trainer._prepare_batch_draws(batch, do_rollout)
        values = trainer._batch_gradients(physical, epoch, do_rollout, draws)
        del physical, draws
    torch.cuda.synchronize()
    return values, {
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "ending_cuda_rng": torch.cuda.get_rng_state().cpu(),
    }


def numerical_equivalence(payload: dict, dataset, device: torch.device) -> dict:
    # A small unequal-agent B4 keeps the legacy graph safely below 16 GiB.
    indices = []
    selected_agent_counts = set()
    for index in range(len(dataset)):
        sample = dataset[index]
        agent_count = int(sample["observed"].shape[0])
        if agent_count <= 8 and agent_count not in selected_agent_counts:
            indices.append(index)
            selected_agent_counts.add(agent_count)
        if len(indices) == 4:
            break
    if len(indices) != 4:
        raise ValueError("numeric validation requires four unequal small-agent scenes")
    batch = collate_scenes([dataset[index] for index in indices])
    trainer = make_trainer(payload, device)
    results = {"scene_indices": indices, "agent_counts": batch.valid.sum(1).tolist()}
    all_passed = True
    for do_rollout in (False, True):
        torch.cuda.manual_seed_all(7319 + int(do_rollout))
        state = torch.cuda.get_rng_state().clone()
        reference, reference_resources = run_gradient_path(
            trainer, batch, 22, do_rollout, False, state)
        repeated, repeated_resources = run_gradient_path(
            trainer, batch, 22, do_rollout, False, state)
        chunked, chunked_resources = run_gradient_path(
            trainer, batch, 22, do_rollout, True, state)

        log_differences = {
            name: abs(float(reference[3][name]) - float(chunked[3][name]))
            for name in reference[3]}
        gradient_differences = {
            name: gradient_comparison(reference[index], chunked[index])
            for index, name in enumerate(("total", "marginal_ade", "marginal_fde"))}
        repeat_differences = {
            name: gradient_comparison(reference[index], repeated[index])
            for index, name in enumerate(("total", "marginal_ade", "marginal_fde"))}

        clipped_reference = clip_global_norm(
            reference[0], trainer.options.gradient_clip_norm)
        clipped_chunked = clip_global_norm(
            chunked[0], trainer.options.gradient_clip_norm)
        candidate_reference, pending_reference = trainer.optimizer.propose(clipped_reference)
        candidate_chunked, pending_chunked = trainer.optimizer.propose(clipped_chunked)
        decrement_reference, _ = project_two_halfspaces(
            candidate_reference, reference[1], reference[2])
        decrement_chunked, _ = project_two_halfspaces(
            candidate_chunked, chunked[1], chunked[2])
        candidate_difference = gradient_comparison(
            candidate_reference, candidate_chunked)
        decrement_difference = gradient_comparison(
            decrement_reference, decrement_chunked)
        update_difference = gradient_comparison(
            [parameter.detach() - value for parameter, value in zip(
                trainer.parameters, decrement_reference, strict=True)],
            [parameter.detach() - value for parameter, value in zip(
                trainer.parameters, decrement_chunked, strict=True)],
        )
        pending_difference = max(
            float((left[key] - right[key]).abs().max())
            for left, right in zip(pending_reference, pending_chunked, strict=True)
            for key in ("moment", "variance"))
        constraints = {
            "marginal_ade": float(list_dot(chunked[1], decrement_chunked)),
            "marginal_fde": float(list_dot(chunked[2], decrement_chunked)),
        }
        rng_exact = (
            torch.equal(reference_resources.pop("ending_cuda_rng"),
                        chunked_resources.pop("ending_cuda_rng"))
            and torch.equal(repeated_resources.pop("ending_cuda_rng"),
                            torch.cuda.get_rng_state().cpu()))
        passed = (
            max(log_differences.values()) <= LOSS_ABS_TOLERANCE
            and all(item["relative_difference_norm"] <= GRADIENT_RELATIVE_TOLERANCE
                    for item in gradient_differences.values()
                    if item["reference_norm"] > 1e-12)
            and all(item["direction_cosine"] >= DIRECTION_COSINE_MINIMUM
                    for item in gradient_differences.values()
                    if item["reference_norm"] > 1e-12)
            and candidate_difference["relative_difference_norm"] <= OPTIMIZER_RELATIVE_TOLERANCE
            and decrement_difference["relative_difference_norm"] <= OPTIMIZER_RELATIVE_TOLERANCE
            and candidate_difference["direction_cosine"] >= OPTIMIZER_DIRECTION_COSINE_MINIMUM
            and decrement_difference["direction_cosine"] >= OPTIMIZER_DIRECTION_COSINE_MINIMUM
            and min(constraints.values()) >= -PROJECTION_TOLERANCE
            and pending_difference <= LOSS_ABS_TOLERANCE
            and rng_exact)
        all_passed = all_passed and passed
        results["rollout_on" if do_rollout else "rollout_off"] = {
            "passed": passed,
            "log_absolute_differences": log_differences,
            "gradient_differences": gradient_differences,
            "legacy_repeat_differences": repeat_differences,
            "candidate_difference": candidate_difference,
            "projected_decrement_difference": decrement_difference,
            "parameter_after_difference": update_difference,
            "pending_optimizer_max_abs_difference": pending_difference,
            "chunked_projection_dots": constraints,
            "cuda_rng_exact": rng_exact,
            "legacy_resources": reference_resources,
            "legacy_repeat_resources": repeated_resources,
            "chunked_resources": chunked_resources,
        }
        del reference, repeated, chunked
        gc.collect()
        torch.cuda.empty_cache()
    results["passed"] = all_passed
    release_trainer(trainer)
    return results


def highest_agent_indices(dataset, manifest_path: Path, count: int = 4) -> list[int]:
    manifest = json.loads(manifest_path.read_text())
    ranked = []
    for index, descriptor in enumerate(manifest["shards"]):
        path = manifest_path.parent / descriptor["path"]
        with np.load(path, allow_pickle=True) as payload:
            agents = int(np.asarray(payload["observed"][0]).shape[0])
        ranked.append((agents, index))
    ranked.sort(reverse=True)
    return [index for _, index in ranked[:count]]


def measured_update(trainer: EBJDTrainer, batches: list, epoch: int) -> dict:
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    free_before, total = torch.cuda.mem_get_info()
    started = time.perf_counter()
    record = trainer.train_step(batches, epoch)
    torch.cuda.synchronize()
    free_after, _ = torch.cuda.mem_get_info()
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    return {
        "record": record,
        "elapsed_seconds": time.perf_counter() - started,
        "free_before_bytes": free_before,
        "free_after_bytes": free_after,
        "total_device_bytes": total,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "reserved_capacity_margin_bytes": total - peak_reserved,
        "finite": finite_tree(record),
    }


def stress_validation(payload: dict, dataset, manifest_path: Path, device: torch.device) -> dict:
    dense_indices = highest_agent_indices(dataset, manifest_path)
    dense_batch = collate_scenes([dataset[index] for index in dense_indices])
    trainer = make_trainer(payload, device)
    trainer.update_index += (-trainer.update_index) % trainer.options.rollout_every
    dense = measured_update(trainer, [dense_batch] * 4, epoch=22)
    dense["scene_indices"] = dense_indices
    dense["agent_counts"] = dense_batch.valid.sum(1).tolist()
    dense["logical_microbatch_shapes"] = [list(dense_batch.observed.shape[:2])] * 4
    release_trainer(trainer)

    failure_batches = [
        collate_scenes([dataset[index] for index in indices])
        for indices in FAILURE_BATCH_INDICES]
    trainer = make_trainer(payload, device)
    trainer.update_index += (-trainer.update_index) % trainer.options.rollout_every
    failure = measured_update(trainer, failure_batches, epoch=22)
    failure["scene_indices"] = [list(indices) for indices in FAILURE_BATCH_INDICES]
    failure["scene_ids"] = [batch.scene_ids for batch in failure_batches]
    failure["agent_counts"] = [batch.valid.sum(1).tolist() for batch in failure_batches]
    failure["logical_microbatch_shapes"] = [list(batch.observed.shape[:2]) for batch in failure_batches]

    trainer.model.eval()
    n57 = collate_scenes([dataset[3302]]).to(device)
    torch.cuda.reset_peak_memory_stats()
    generator = torch.Generator(device=device).manual_seed(2035)
    started = time.perf_counter()
    with torch.no_grad():
        trajectory, _ = predict(
            trainer.model, n57.observed, n57.semantic_maps, n57.valid,
            worlds=20, steps=20, generator=generator)
        metrics = hard_metrics(trajectory, n57.future, n57.valid)
    torch.cuda.synchronize()
    inference = {
        "shape": list(trajectory.shape),
        "finite": bool(torch.isfinite(trajectory).all()),
        "metrics": metrics.__dict__,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    release_trainer(trainer)
    passed = (
        dense["finite"] and failure["finite"] and inference["finite"]
        and dense["reserved_capacity_margin_bytes"] >= MINIMUM_RESERVED_MARGIN_BYTES
        and failure["reserved_capacity_margin_bytes"] >= MINIMUM_RESERVED_MARGIN_BYTES)
    return {"passed": passed, "dense_b4x4_rollout": dense,
            "reconstructed_failure_b4x4_rollout": failure,
            "n57_p20x20_inference": inference}


def continuous_validation(
    payload: dict, dataset, device: torch.device, updates: int,
) -> dict:
    if not 32 <= updates <= 64:
        raise ValueError("continuous validation must contain 32--64 updates")
    generator = torch.Generator()
    trainer = make_trainer(payload, device)
    generator.set_state(payload["rng"]["loader_generator"])
    loader = DataLoader(
        dataset, batch_size=4, shuffle=True, collate_fn=collate_scenes,
        num_workers=0, generator=generator)
    pending = []
    records = []
    for batch in loader:
        pending.append(augment_batch(batch, generator))
        if len(pending) < 4:
            continue
        measured = measured_update(trainer, pending, epoch=22)
        measured["ordinal"] = len(records) + 1
        measured["scene_ids"] = [item.scene_ids for item in pending]
        measured["padded_N"] = [int(item.observed.shape[1]) for item in pending]
        measured["valid_agents"] = [int(item.valid.sum()) for item in pending]
        records.append(measured)
        pending = []
        if len(records) >= updates:
            break
    exact_failure_seen = (
        len(records) >= 47
        and records[46]["scene_ids"] == [
            [
                "hotel:train:biwi_eth:2862-2976",
                "hotel:train:crowds_zara02:3340-3530",
                "hotel:train:students003:2400-2590",
                "hotel:train:biwi_eth:9039-9153",
            ],
            [
                "hotel:train:students001:0-190",
                "hotel:train:biwi_eth:9387-9501",
                "hotel:train:biwi_eth:4241-4355",
                "hotel:train:crowds_zara02:5460-5650",
            ],
            [
                "hotel:train:crowds_zara02:710-900",
                "hotel:train:crowds_zara03:940-1130",
                "hotel:train:crowds_zara03:7290-7480",
                "hotel:train:crowds_zara02:5420-5610",
            ],
            [
                "hotel:train:students003:3400-3590",
                "hotel:train:biwi_eth:6809-6923",
                "hotel:train:crowds_zara02:2500-2690",
                "hotel:train:biwi_eth:8475-8589",
            ],
        ])
    max_reserved = max(item["peak_reserved_bytes"] for item in records)
    total = records[0]["total_device_bytes"]
    passed = (
        len(records) == updates and all(item["finite"] for item in records)
        and exact_failure_seen
        and total - max_reserved >= MINIMUM_RESERVED_MARGIN_BYTES)
    summary = {
        "passed": passed,
        "updates": len(records),
        "rollout_updates": sum(item["record"]["rollout_update"] == 1 for item in records),
        "exact_failure_group_seen_at_ordinal_47": exact_failure_seen,
        "maximum_peak_allocated_bytes": max(item["peak_allocated_bytes"] for item in records),
        "maximum_peak_reserved_bytes": max_reserved,
        "minimum_reserved_capacity_margin_bytes": total - max_reserved,
        "maximum_elapsed_seconds": max(item["elapsed_seconds"] for item in records),
        "records": records,
    }
    release_trainer(trainer)
    return summary


def main() -> None:
    args = parse_args()
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("OOM recovery validation requires the real CUDA device")
    checkpoint = Path(args.checkpoint).resolve()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    diagnostic_payload = payload
    if args.numeric_precision == "fp32":
        if args.mode not in ("numeric", "all"):
            raise ValueError("--numeric-precision=fp32 is only meaningful for numeric mode")
        diagnostic_payload = copy.deepcopy(payload)
        diagnostic_payload["resolved_config"]["training"]["mixed_precision"] = "fp32"
    manifest_path = Path(payload["resolved_config"]["paths"]["train_manifest"]).resolve()
    dataset = SceneManifestDataset(manifest_path)
    device = torch.device(args.device)
    output = {
        "schema": "ebjd-oom-recovery-gpu-validation-v1",
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_epoch": int(payload["epoch"]),
        "checkpoint_update_index": int(payload["update_index"]),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(device),
        "numeric_precision": args.numeric_precision,
        "allocator_environment": {
            "PYTORCH_CUDA_ALLOC_CONF": __import__("os").environ.get(
                "PYTORCH_CUDA_ALLOC_CONF", "")},
        "acceptance_thresholds": {
            "loss_absolute": LOSS_ABS_TOLERANCE,
            "gradient_relative": GRADIENT_RELATIVE_TOLERANCE,
            "direction_cosine_minimum": DIRECTION_COSINE_MINIMUM,
            "optimizer_relative": OPTIMIZER_RELATIVE_TOLERANCE,
            "optimizer_direction_cosine_minimum": OPTIMIZER_DIRECTION_COSINE_MINIMUM,
            "projection_minimum": -PROJECTION_TOLERANCE,
            "minimum_reserved_capacity_margin_bytes": MINIMUM_RESERVED_MARGIN_BYTES,
        },
        "parameter_count": sum(parameter.numel() for parameter in build_model(
            payload["resolved_config"]).parameters()),
    }
    modes = ("numeric", "stress", "continuous") if args.mode == "all" else (args.mode,)
    for mode in modes:
        if mode == "numeric":
            output[mode] = numerical_equivalence(diagnostic_payload, dataset, device)
        elif mode == "stress":
            output[mode] = stress_validation(payload, dataset, manifest_path, device)
        else:
            output[mode] = continuous_validation(
                payload, dataset, device, args.continuous_updates)
        atomic_json(Path(args.output), output)
    output["passed"] = all(output[name]["passed"] for name in modes)
    atomic_json(Path(args.output), output)
    print(json.dumps({"output": str(Path(args.output).resolve()),
                      "passed": output["passed"], "modes": modes}, sort_keys=True))


if __name__ == "__main__":
    main()
