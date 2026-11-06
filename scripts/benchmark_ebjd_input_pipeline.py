#!/usr/bin/env python3
"""Paired real-update benchmark for the unified EBJD input pipeline."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import resource
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from ebjd.config import build_model, load_config
from ebjd.data import (
    SceneBatch, apply_scene_rigid_plan, sample_scene_rigid_plan,
)
from ebjd.proposals import NativeProposalBank, file_sha256
from ebjd.objectives import (
    diffusion_loss, gated_geometry_mean, map_loss, relative_motion_loss,
)
from ebjd.representation import cosine_vp
from ebjd.unified import (
    FrozenNativeGoalProvider, UnifiedCouplingBridge, joint_pair_loss,
)
from run_zara1_native_prior_pilot import loader
from run_zara1_unified_coupling_bridge import (
    bridge_and_pair_loss, optimizer_for,
)


MODES = ("legacy_cpu_rigid", "compact_uint8_gpu_rigid_v1")


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compact_map_batch(batch: SceneBatch) -> SceneBatch:
    return SceneBatch(
        batch.observed, batch.future, batch.semantic_maps.to(torch.uint8),
        batch.valid, batch.scene_ids, batch.agent_ids, batch.frame_ids,
        batch.timestamps, batch.source_sequences, batch.metadata,
        batch.paper_metric_scale)


def build_bank(config: dict, path: str) -> NativeProposalBank:
    return NativeProposalBank(
        path,
        expected_checkpoint_sha256=config["paths"][
            "matched_gdts_validation_checkpoint_sha256"],
        expected_manifest_sha256=file_sha256(config["paths"]["train_manifest"]),
        expected_split="train")


def build_runtime(
    config: dict, checkpoint: str, device: torch.device,
) -> tuple[UnifiedCouplingBridge, torch.optim.Optimizer, dict]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    bridge = build_model(config, proposal_conditioned=True)
    model = UnifiedCouplingBridge(bridge).to(device)
    model.training_options = {"coarse": 0.25, "geometry": 0.05, "map": 0.10}
    model.load_state_dict(payload["model"], strict=True)
    optimizer = optimizer_for(model, config)
    optimizer.load_state_dict(payload["optimizer"])
    return model, optimizer, payload


def prepare_batch(
    cpu_batch: SceneBatch,
    proposals: torch.Tensor,
    plan,
    mode: str,
    device: torch.device,
) -> tuple[SceneBatch, torch.Tensor]:
    if mode == "legacy_cpu_rigid":
        augmented, proposals = apply_scene_rigid_plan(cpu_batch, proposals, plan)
        return augmented.to(device), proposals.to(device)
    if mode == "compact_uint8_gpu_rigid_v1":
        compact = compact_map_batch(cpu_batch).to(device)
        proposals = proposals.to(device)
        with torch.no_grad():
            return apply_scene_rigid_plan(compact, proposals, plan)
    raise ValueError(mode)


def logical_backward(
    model, batch, proposals, config, device, *, time_draw=None, noise=None,
) -> tuple[float, dict[str, bool]]:
    logical_batch = int(batch.observed.shape[0])
    total_agents = int(batch.valid.sum())
    if time_draw is None:
        time_draw = torch.randint(
            1, 101, (logical_batch,), device=device).float() / 100
    if noise is None:
        noise = torch.randn(
            logical_batch, 1, batch.observed.shape[1], 12, 2,
            dtype=torch.float32, device=device)
    total_value = 0.0
    groups = range(logical_batch) if config["training"].get(
        "scene_compaction", False) else [None]
    for scene in groups:
        if scene is None:
            physical = batch
            physical_proposals = proposals
            physical_time, physical_noise = time_draw, noise
            agent_weight = scene_weight = 1.0
        else:
            positions = batch.valid[scene].nonzero(as_tuple=False).flatten()
            physical = batch.compact_scenes(scene, scene + 1)
            physical_proposals = proposals[
                scene:scene + 1, :, positions.to(proposals.device)]
            physical_time = time_draw[scene:scene + 1]
            physical_noise = noise[scene:scene + 1, :, positions.to(device)]
            agent_weight = float(positions.numel()) / total_agents
            scene_weight = 1.0 / logical_batch
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16,
            enabled=device.type == "cuda" and torch.cuda.is_bf16_supported(),
        ):
            _, _, _, agent_part, scene_part = bridge_and_pair_loss(
                model, physical, physical_proposals,
                time_draw=physical_time, noise=physical_noise)
            total = agent_weight * agent_part + scene_weight * scene_part
            scaled = total / int(config["training"]["gradient_accumulation"])
        scaled.backward()
        total_value += float(total.detach())
    gradients = {
        "msce_map_unet": any(
            parameter.grad is not None
            for name, parameter in model.named_parameters()
            if "bridge.encoder.map_unet" in name),
        "coupling": any(
            parameter.grad is not None
            for parameter in model.coupling.parameters()),
        "gcjd_ctf": any(
            parameter.grad is not None
            for parameter in model.bridge.denoiser.parameters()),
    }
    return total_value, gradients


def optimizer_update(model, optimizer) -> None:
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def timed_logical_backward(
    model, batch, proposals, config, device,
) -> tuple[float, dict[str, float]]:
    """Low-frequency synchronized timing; never used for speedup claims."""
    logical_batch = int(batch.observed.shape[0])
    total_agents = int(batch.valid.sum())
    time_draw = torch.randint(
        1, 101, (logical_batch,), device=device).float() / 100
    noise = torch.randn(
        logical_batch, 1, batch.observed.shape[1], 12, 2,
        dtype=torch.float32, device=device)
    timings = {"msce_unet": 0.0, "coupling": 0.0,
               "fjab_ctf_and_losses": 0.0, "backward": 0.0}
    total_value = 0.0
    groups = range(logical_batch) if config["training"].get(
        "scene_compaction", False) else [None]
    for scene in groups:
        if scene is None:
            physical, physical_proposals = batch, proposals
            physical_time, physical_noise = time_draw, noise
            agent_weight = scene_weight = 1.0
        else:
            positions = batch.valid[scene].nonzero(as_tuple=False).flatten()
            physical = batch.compact_scenes(scene, scene + 1)
            physical_proposals = proposals[
                scene:scene + 1, :, positions.to(proposals.device)]
            physical_time = time_draw[scene:scene + 1]
            physical_noise = noise[scene:scene + 1, :, positions.to(device)]
            agent_weight = float(positions.numel()) / total_agents
            scene_weight = 1.0 / logical_batch

        torch.cuda.synchronize(device); started = time.perf_counter()
        context = model.encode_context(
            physical.observed, physical.semantic_maps, physical.valid)
        torch.cuda.synchronize(device)
        timings["msce_unet"] += time.perf_counter() - started

        torch.cuda.synchronize(device); started = time.perf_counter()
        coupled, _, _ = model.coupling.assign(
            context, physical_proposals,
            model.bridge.representation.goal_scale, straight_through=True)
        pair = joint_pair_loss(
            coupled, physical.future[..., -1, :], physical.valid,
            model.bridge.representation.goal_scale, temperature=0.1)
        endpoint = physical.future[..., -1, :].detach()
        clean = model.bridge.representation.encode_target_conditioned(
            physical.future.detach(), context.baseline, endpoint,
            eta=0.0)[:, None]
        alpha, sigma = cosine_vp(physical_time)
        alpha = alpha.to(clean).view(-1, 1, 1, 1, 1)
        sigma = sigma.to(clean).view(-1, 1, 1, 1, 1)
        noisy = alpha * clean.float() + sigma * physical_noise
        target = alpha * physical_noise - sigma * clean.float()
        torch.cuda.synchronize(device)
        timings["coupling"] += time.perf_counter() - started

        torch.cuda.synchronize(device); started = time.perf_counter()
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16,
            enabled=device.type == "cuda" and torch.cuda.is_bf16_supported(),
        ):
            final, coarse, coarse_path, _ = model.bridge.denoiser(
                noisy, physical_time, context, proposal_goals=endpoint[:, None],
                proposal_eta=0.0)
            final_clean = alpha * noisy - sigma * final.float()
            final_path, _ = model.bridge.representation.decode_conditioned(
                final_clean, context.baseline, endpoint[:, None], eta=0.0)
            options = model.training_options
            loss_diff = diffusion_loss(
                final, coarse, target, physical.valid, options["coarse"])
            per_scene_geo = 0.5 * (
                relative_motion_loss(
                    coarse_path, physical.future, context, 6.0, 2.0, 4.8)
                + relative_motion_loss(
                    final_path, physical.future, context, 6.0, 2.0, 4.8))
            loss_geo = gated_geometry_mean(per_scene_geo, alpha)
            loss_map = map_loss(physical.future, context, 0.5)
            agent_part = loss_diff + options["map"] * loss_map
            scene_part = options["geometry"] * loss_geo + pair
            total = agent_weight * agent_part + scene_weight * scene_part
            scaled = total / int(config["training"]["gradient_accumulation"])
        torch.cuda.synchronize(device)
        timings["fjab_ctf_and_losses"] += time.perf_counter() - started

        torch.cuda.synchronize(device); started = time.perf_counter()
        scaled.backward()
        torch.cuda.synchronize(device)
        timings["backward"] += time.perf_counter() - started
        total_value += float(total.detach())
    return total_value, timings


def parameter_state(model) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().float().cpu().clone()
        for name, value in model.state_dict().items()
        if torch.is_floating_point(value)
    }


def relative_l2(
    left: dict[str, torch.Tensor], right: dict[str, torch.Tensor], prefix: str,
) -> float:
    numerator = denominator = 0.0
    for name in left:
        if not name.startswith(prefix):
            continue
        numerator += float((left[name] - right[name]).double().square().sum())
        denominator += float(left[name].double().square().sum())
    return math.sqrt(numerator) / max(math.sqrt(denominator), 1e-12)


def fixed_batches(config: dict, count: int, seed: int) -> list[tuple]:
    local = copy.deepcopy(config)
    local["training"]["data_loader_workers"] = 0
    local["training"]["ebjd_input_pipeline"] = "legacy_cpu_rigid"
    generator = torch.Generator().manual_seed(seed)
    iterator = iter(loader(local, "train", generator))
    batches = []
    for _ in range(count):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader(local, "train", generator))
            batch = next(iterator)
        batches.append(batch)
    return batches


def correctness_run(
    config: dict, checkpoint: str, bank: NativeProposalBank,
    batches: list[SceneBatch], mode: str, device: torch.device, seed: int,
) -> dict:
    seed_all(seed)
    model, optimizer, _ = build_runtime(config, checkpoint, device)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    provider = FrozenNativeGoalProvider(bank)
    augmentation_generator = torch.Generator().manual_seed(seed + 11)
    losses, gradient_receipts = [], []
    accumulation = int(config["training"]["gradient_accumulation"])
    for index, raw in enumerate(batches):
        proposals = provider(raw)
        plan = sample_scene_rigid_plan(raw, augmentation_generator)
        batch, proposals = prepare_batch(raw, proposals, plan, mode, device)
        value, gradients = logical_backward(
            model, batch, proposals, config, device)
        losses.append(value)
        gradient_receipts.append(gradients)
        if (index + 1) % accumulation == 0:
            optimizer_update(model, optimizer)
    torch.cuda.synchronize(device)
    finite = all(
        torch.isfinite(value).all().item()
        for value in model.state_dict().values()
        if torch.is_floating_point(value))
    result = {
        "mode": mode, "losses": losses, "parameters_finite": finite,
        "all_required_gradients_present": all(
            all(receipt.values()) for receipt in gradient_receipts),
        "gradient_receipts": gradient_receipts,
    }
    result["state"] = parameter_state(model)
    del model, optimizer
    torch.cuda.empty_cache()
    return result


def process_rss_mib() -> float:
    # Linux reports ru_maxrss in KiB.
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def performance_run(
    config: dict, checkpoint: str, bank: NativeProposalBank, mode: str,
    device: torch.device, seed: int, warmup_updates: int, measured_updates: int,
) -> dict:
    seed_all(seed)
    local = copy.deepcopy(config)
    local["training"]["ebjd_input_pipeline"] = mode
    generator = torch.Generator().manual_seed(seed)
    train_loader = loader(local, "train", generator)
    iterator = iter(train_loader)
    provider = FrozenNativeGoalProvider(bank)
    model, optimizer, _ = build_runtime(local, checkpoint, device)
    model.train(); optimizer.zero_grad(set_to_none=True)
    accumulation = int(local["training"]["gradient_accumulation"])
    logical_target = (warmup_updates + measured_updates) * accumulation
    data_wait = input_prepare = 0.0
    measured_losses = []
    measured_started = None
    torch.cuda.reset_peak_memory_stats(device)
    rss_before = process_rss_mib()
    for logical_index in range(logical_target):
        update_index = logical_index // accumulation
        if update_index == warmup_updates and logical_index % accumulation == 0:
            torch.cuda.synchronize(device)
            measured_started = time.perf_counter()
            data_wait = input_prepare = 0.0
            torch.cuda.reset_peak_memory_stats(device)
            rss_before = process_rss_mib()
        wait_started = time.perf_counter()
        try:
            cpu_batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            cpu_batch = next(iterator)
        data_wait += time.perf_counter() - wait_started
        prep_started = time.perf_counter()
        proposals = provider(cpu_batch)
        plan = sample_scene_rigid_plan(cpu_batch, generator)
        batch, proposals = prepare_batch(
            cpu_batch, proposals, plan, mode, device)
        input_prepare += time.perf_counter() - prep_started
        value, _ = logical_backward(model, batch, proposals, local, device)
        if update_index >= warmup_updates:
            measured_losses.append(value)
        if (logical_index + 1) % accumulation == 0:
            optimizer_update(model, optimizer)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - measured_started
    result = {
        "mode": mode, "seed": seed,
        "measured_updates": measured_updates,
        "seconds": elapsed, "seconds_per_update": elapsed / measured_updates,
        "data_wait_seconds_per_update": data_wait / measured_updates,
        "input_prepare_dispatch_seconds_per_update": input_prepare / measured_updates,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
        "rss_before_mib": rss_before, "rss_peak_mib": process_rss_mib(),
        "mean_logical_loss": statistics.fmean(measured_losses),
    }
    del iterator, train_loader, model, optimizer
    torch.cuda.empty_cache()
    return result


def profile_one_batch(
    config: dict, checkpoint: str, bank: NativeProposalBank,
    mode: str, device: torch.device, seed: int,
) -> dict:
    seed_all(seed)
    local = copy.deepcopy(config)
    local["training"]["ebjd_input_pipeline"] = mode
    generator = torch.Generator().manual_seed(seed)
    train_loader = loader(local, "train", generator)
    iterator = iter(train_loader)
    provider = FrozenNativeGoalProvider(bank)
    model, optimizer, _ = build_runtime(local, checkpoint, device)
    model.train(); optimizer.zero_grad(set_to_none=True)
    rows = []
    accumulation = int(local["training"]["gradient_accumulation"])
    for logical_index in range(8):
        torch.cuda.synchronize(device)
        started = time.perf_counter(); cpu_batch = next(iterator)
        wait = time.perf_counter() - started
        proposals = provider(cpu_batch)
        plan = sample_scene_rigid_plan(cpu_batch, generator)
        torch.cuda.synchronize(device); started = time.perf_counter()
        batch, proposals = prepare_batch(cpu_batch, proposals, plan, mode, device)
        torch.cuda.synchronize(device); prep = time.perf_counter() - started
        value, components = timed_logical_backward(
            model, batch, proposals, local, device)
        optimizer_seconds = 0.0
        if (logical_index + 1) % accumulation == 0:
            torch.cuda.synchronize(device); started = time.perf_counter()
            optimizer_update(model, optimizer)
            torch.cuda.synchronize(device)
            optimizer_seconds = time.perf_counter() - started
        rows.append({
            "data_wait_seconds": wait,
            "augment_transfer_seconds": prep,
            **{name + "_seconds": value for name, value in components.items()},
            "optimizer_seconds": optimizer_seconds,
            "agents": int(batch.valid.sum()),
            "padded_agents": int(batch.valid.numel()),
            "map_host_bytes": int(cpu_batch.semantic_maps.numel()
                                  * cpu_batch.semantic_maps.element_size()),
            "loss": value,
        })
    del model, optimizer, iterator, train_loader
    torch.cuda.empty_cache()
    return {"mode": mode, "samples": rows}


def summarize_correctness(runs: dict[str, dict]) -> dict:
    base_a, base_b = runs["legacy_a"], runs["legacy_b"]
    candidate = runs["candidate"]

    def loss_delta(left, right):
        differences = [abs(a - b) / max(abs(a), abs(b), 1e-8)
                       for a, b in zip(left["losses"], right["losses"], strict=True)]
        return {"max_relative": max(differences),
                "mean_relative": statistics.fmean(differences)}

    result = {
        "legacy_repeat_loss": loss_delta(base_a, base_b),
        "candidate_vs_legacy_loss": loss_delta(base_a, candidate),
        "legacy_repeat_parameter_relative_l2": {
            "goal_unet": relative_l2(
                base_a["state"], base_b["state"], "bridge.encoder.map_unet"),
            "trajectory_and_coupling": relative_l2(
                base_a["state"], base_b["state"], ""),
        },
        "candidate_vs_legacy_parameter_relative_l2": {
            "goal_unet": relative_l2(
                base_a["state"], candidate["state"], "bridge.encoder.map_unet"),
            "trajectory_and_coupling": relative_l2(
                base_a["state"], candidate["state"], ""),
        },
        "finite": all(run["parameters_finite"] for run in runs.values()),
        "all_required_gradients_present": all(
            run["all_required_gradients_present"] for run in runs.values()),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--correctness-updates", type=int, default=8)
    parser.add_argument("--warmup-updates", type=int, default=4)
    parser.add_argument("--measured-updates", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the real-update benchmark requires one visible CUDA GPU")
    spec_path = Path(args.spec).resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    receipt = {
        "schema": "ebjd-input-pipeline-benchmark-v1",
        "spec": str(spec_path), "source_commit": spec["source_commit"],
        "device": torch.cuda.get_device_name(device), "cases": [],
        "optimizer_updates_are_real": True, "rollout": False,
        "actual_step_projection": False, "ddim_sampling_in_timing": False,
    }
    for case_index, case in enumerate(spec["cases"]):
        config = load_config(case["config"])
        bank = build_bank(config, case["train_bank"])
        accumulation = int(config["training"]["gradient_accumulation"])
        batches = fixed_batches(
            config, args.correctness_updates * accumulation,
            7100 + case_index)
        correctness_runs = {
            "legacy_a": correctness_run(
                config, case["checkpoint"], bank, batches,
                "legacy_cpu_rigid", device, 8100 + case_index),
            "legacy_b": correctness_run(
                config, case["checkpoint"], bank, batches,
                "legacy_cpu_rigid", device, 8100 + case_index),
            "candidate": correctness_run(
                config, case["checkpoint"], bank, batches,
                "compact_uint8_gpu_rigid_v1", device, 8100 + case_index),
        }
        correctness = summarize_correctness(correctness_runs)
        profiles = [profile_one_batch(
            config, case["checkpoint"], bank, mode, device,
            9100 + case_index) for mode in MODES]
        performance = []
        for repeat in range(args.repeats):
            for mode in MODES:
                performance.append(performance_run(
                    config, case["checkpoint"], bank, mode, device,
                    10100 + case_index * 100 + repeat,
                    args.warmup_updates, args.measured_updates))
        legacy = [row["seconds_per_update"] for row in performance
                  if row["mode"] == "legacy_cpu_rigid"]
        candidate = [row["seconds_per_update"] for row in performance
                     if row["mode"] == "compact_uint8_gpu_rigid_v1"]
        case_result = {
            "name": case["name"], "config": case["config"],
            "checkpoint": case["checkpoint"],
            "checkpoint_sha256": file_sha256(case["checkpoint"]),
            "train_bank": case["train_bank"],
            "train_bank_sha256": file_sha256(case["train_bank"]),
            "correctness": correctness, "profiles": profiles,
            "performance": performance,
            "paired_speedups": [a / b for a, b in zip(
                legacy, candidate, strict=True)],
            "mean_speedup": statistics.fmean(
                a / b for a, b in zip(legacy, candidate, strict=True)),
        }
        receipt["cases"].append(case_result)
        atomic_json(output / "BENCHMARK_RESULTS.partial.json", receipt)
        for run in correctness_runs.values():
            run.pop("state", None)
    receipt["status"] = "COMPLETE"
    atomic_json(output / "BENCHMARK_RESULTS.json", receipt)
    partial = output / "BENCHMARK_RESULTS.partial.json"
    if partial.exists():
        partial.unlink()


if __name__ == "__main__":
    main()
