#!/usr/bin/env python3
"""Bounded real-training-data checks; never updates a formal checkpoint.

Runs the existing FULL unified objective and complete optimizer updates on a
fixed training-window plan.  New-prior loss values are diagnostic only: a new
whitening changes their units and they cannot be ranked as forecast metrics.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import time
import traceback

import numpy as np
import torch

from ebjd.config import build_model, load_config
from ebjd.data import SceneManifestDataset, augment_batch_with_points, collate_scenes
from ebjd.proposals import file_sha256
from ebjd.sampling import differentiable_sample
from ebjd.train import fit_training_scales
from ebjd.unified import UnifiedCouplingBridge, gather_candidates
from run_zara1_unified_coupling_bridge import (
    atomic_json, bridge_and_pair_loss, make_bank, optimizer_for, seed_all, set_epoch_lr,
)


def coordinate_batches(manifest: Path):
    """Reuse the exact scale fitter without reading any map or held-out future."""
    payload = json.loads(manifest.read_text())
    for entry in payload["shards"]:
        path = manifest.parent / entry["path"]
        if file_sha256(path) != entry["sha256"]:
            raise ValueError(f"scale-fit shard identity changed: {path}")
        with np.load(path, allow_pickle=True) as shard:
            for observed, future in zip(shard["observed"], shard["future"], strict=True):
                obs = torch.as_tensor(np.asarray(observed, dtype=np.float32))[None]
                yield SimpleNamespace(
                    observed=obs,
                    future=torch.as_tensor(np.asarray(future, dtype=np.float32))[None],
                    valid=torch.ones(obs.shape[:2], dtype=torch.bool))


def tensor_digest(tensors):
    digest = hashlib.sha256()
    for value in tensors:
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def relative_parameter_difference(left, right):
    numerator = denominator = 0.0
    for key, value in left.items():
        numerator += float((value.double() - right[key].double()).square().sum())
        denominator += float(value.double().square().sum())
    return (numerator / max(denominator, 1e-30)) ** 0.5


def run_arm(config, dataset, bank, plan, arm, updates, device, deadline):
    seed_all(3101)
    local = deepcopy(config)
    local["representation"]["type"] = (
        "invertible_endpoint_acceleration_bridge_v1" if arm == "acceleration"
        else "invertible_endpoint_brownian_bridge")
    bridge = build_model(local, proposal_conditioned=True)
    scales = fit_training_scales(
        bridge, coordinate_batches(Path(local["paths"]["train_manifest"])))
    model = UnifiedCouplingBridge(
        bridge, singleton_backward="identity" if arm == "singleton_fix" else "legacy")
    model.training_options = {"coarse": 0.25, "geometry": 0.05, "map": 0.10}
    model.to(device).train()
    optimizer = optimizer_for(model, local)
    set_epoch_lr(optimizer, 1)
    initial = {k: p.detach().cpu().clone() for k, p in model.named_parameters()}
    generator = torch.Generator().manual_seed(3101)
    seed_all(71131)
    accumulation = int(local["training"]["gradient_accumulation"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    rows, seen, input_digests = [], [], []
    gradient_seen = set()
    for update in range(updates):
        if time.perf_counter() > deadline:
            raise TimeoutError("bounded validation wall-clock budget exhausted")
        optimizer.zero_grad(set_to_none=True)
        total_loss = total_bridge = total_pair = 0.0
        for micro in range(accumulation):
            ids = plan[update * accumulation + micro]
            raw = collate_scenes([dataset[index] for index in ids])
            proposal = bank.batch(raw)
            prepared, proposal = augment_batch_with_points(raw, proposal, generator)
            batch_size = prepared.observed.shape[0]
            agent_total = int(prepared.valid.sum())
            times = torch.randint(1, 101, (batch_size,), device=device).float() / 100
            noise = torch.randn(batch_size, 1, prepared.observed.shape[1], 12, 2,
                                device=device, dtype=torch.float32)
            input_digests.append(tensor_digest([
                prepared.observed, prepared.future, prepared.semantic_maps,
                prepared.valid, proposal, times, noise]))
            seen.extend(zip(prepared.scene_ids, prepared.valid.sum(1).tolist()))
            for scene in range(batch_size):
                active = prepared.valid[scene].nonzero(as_tuple=False).flatten()
                physical = prepared.compact_scenes(scene, scene + 1).to(device)
                physical_goals = proposal[scene:scene + 1, :, active].to(device)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=device.type == "cuda" and
                                    local["training"]["mixed_precision"] != "fp32"):
                    _, bridge_loss, pair_loss, agent_part, scene_part = bridge_and_pair_loss(
                        model, physical, physical_goals,
                        time_draw=times[scene:scene + 1],
                        noise=noise[scene:scene + 1, :, active.to(device)])
                    value = agent_part * (active.numel() / agent_total) + scene_part / batch_size
                if not torch.isfinite(value):
                    raise FloatingPointError("non-finite real-data loss")
                (value / accumulation).backward()
                total_loss += float(value.detach()) / accumulation
                total_bridge += float(bridge_loss.detach()) / (batch_size * accumulation)
                total_pair += float(pair_loss.detach()) / (batch_size * accumulation)
            del raw, prepared
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                if not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError(f"non-finite gradient: {name}")
                if parameter.grad.abs().sum() > 0:
                    gradient_seen.add(name)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        rows.append({"update": update + 1, "loss": total_loss,
                     "bridge_loss": total_bridge, "pair_loss": total_pair,
                     "gradient_norm_before_clip": float(norm)})
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    parameters = {k: p.detach().cpu().clone() for k, p in model.named_parameters()}
    if not all(torch.isfinite(p).all() for p in parameters.values()):
        raise FloatingPointError("non-finite updated parameter")
    required = ["bridge.encoder.map_unet", "bridge.encoder.history_gru",
                "bridge.encoder.history_social", "bridge.denoiser.blocks.0",
                "bridge.denoiser.blocks.5", "bridge.denoiser.clean_injection",
                "bridge.denoiser.map_injection", "bridge.denoiser.future_geometry_mlp",
                "bridge.denoiser.proposal_embedding", "coupling"]
    gradients = {prefix: any(k.startswith(prefix) for k in gradient_seen) for prefix in required}
    if not all(gradients.values()):
        raise RuntimeError(f"missing nonzero training-module gradients: {gradients}")
    delta = relative_parameter_difference(initial, parameters)
    if delta == 0:
        raise RuntimeError("optimizer did not update model parameters")
    sampling_receipt = None
    if arm == "acceleration":
        model.eval()
        with torch.no_grad():
            context = model.encode_context(physical.observed, physical.semantic_maps, physical.valid)
            initial_noise = torch.randn(1, 20, physical.observed.shape[1], 12, 2, device=device)
            h, indices, _ = model.coupling.assign(
                context, physical_goals, model.bridge.representation.goal_scale,
                straight_through=False)
            path, g_goal, _ = differentiable_sample(
                model.bridge, context, initial_noise, 20,
                proposal_goals=physical_goals, proposal_eta=0.0)
            h_path, h_goal, _ = differentiable_sample(
                model.bridge, context, gather_candidates(initial_noise, indices), 20,
                proposal_goals=h, proposal_eta=0.0)
            if not torch.isfinite(path).all() or not torch.isfinite(h_path).all():
                raise FloatingPointError("non-finite 20-world/20-step sample")
            if not torch.equal(g_goal, physical_goals) or not torch.equal(h_goal, h):
                raise RuntimeError("fixed endpoint identity changed")
            if not torch.equal(g_goal.sort(1).values, h_goal.sort(1).values):
                raise RuntimeError("coupling changed endpoint candidate multiset")
            sampling_receipt = {"worlds": 20, "steps": 20, "finite": True,
                                "literal_endpoints_exact": True,
                                "G_H_endpoint_multisets_equal": True,
                                "scenes": 1, "agents": int(physical.valid.sum())}
    receipt = {
        "arm": arm, "optimizer_updates": updates,
        "logical_batch_scenes": int(local["training"]["batch_scenes"]),
        "gradient_accumulation": accumulation,
        "map_pixels": list(physical.semantic_maps.shape[-2:]),
        "training_only_scale_fit": {"goal": scales[0], "bridge": scales[1]},
        "finite": True, "required_module_nonzero_gradients": gradients,
        "parameter_relative_l2_change": delta, "updates": rows,
        "window_plan": [{"scene_id": s, "agents": int(n)} for s, n in seen],
        "input_digests": input_digests,
        "diagnostic_wall_seconds": elapsed,
        "not_a_performance_benchmark": True,
        "peak_allocated_mib": (torch.cuda.max_memory_allocated(device) / 2**20
                               if device.type == "cuda" else None),
        "sampling": sampling_receipt,
    }
    del model, bridge, optimizer, initial
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return receipt, parameters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-bank", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--updates", type=int, default=8)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--max-seconds", type=float, default=900)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite check output: {output}")
    if args.updates < 1 or args.max_seconds <= 0:
        raise ValueError("positive updates and budget required")
    report = {"status": "RUNNING", "formal_optimizer_updates": 0,
              "temporary_optimizer_updates": 0, "arms": [],
              "config": str(Path(args.config).resolve()),
              "config_sha256": file_sha256(args.config),
              "train_bank": str(Path(args.train_bank).resolve()),
              "train_bank_sha256": file_sha256(args.train_bank), "device": args.device}
    atomic_json(output / "RESULTS.json", report)
    started = time.perf_counter()
    try:
        torch.set_num_threads(2)
        device = torch.device(args.device)
        config = load_config(args.config)
        report["precision"] = config["training"]["mixed_precision"]
        manifest = Path(config["paths"]["train_manifest"])
        dataset = SceneManifestDataset(manifest)
        bank = make_bank(args.train_bank,
            config["paths"]["matched_gdts_validation_checkpoint_sha256"],
            file_sha256(manifest), "train")
        size = int(config["training"]["batch_scenes"])
        micros = args.updates * int(config["training"]["gradient_accumulation"])
        if micros * size > len(dataset):
            raise ValueError("check would exhaust the fixed unique window plan")
        order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(3101))
        plan = [order[i * size:(i + 1) * size].tolist() for i in range(micros)]
        reference = None
        for arm in ("control", "control_repeat", "acceleration", "singleton_fix"):
            receipt, parameters = run_arm(config, dataset, bank, plan, arm,
                args.updates, device, started + args.max_seconds)
            report["temporary_optimizer_updates"] += args.updates
            if reference is None:
                reference = parameters
            else:
                receipt["parameter_relative_l2_vs_control"] = relative_parameter_difference(
                    reference, parameters)
                if receipt["input_digests"] != report["arms"][0]["input_digests"]:
                    raise RuntimeError(f"paired raw/augmented inputs or RNG changed in {arm}")
            report["arms"].append(receipt)
            atomic_json(output / "RESULTS.json", report)
            print(json.dumps({"arm": arm, "status": "PASS",
                              "temporary_updates": args.updates}), flush=True)
        report["status"] = "PASS_INTERFACE_ONLY_NO_ACCURACY_CLAIM"
    except BaseException as error:
        report.update(status="FAILED", reason=f"{type(error).__name__}: {error}",
                      traceback=traceback.format_exc())
        raise
    finally:
        report["wall_seconds"] = time.perf_counter() - started
        report["gpu_wall_seconds"] = report["wall_seconds"] if args.device == "cuda" else 0
        atomic_json(output / "RESULTS.json", report)


if __name__ == "__main__":
    main()
