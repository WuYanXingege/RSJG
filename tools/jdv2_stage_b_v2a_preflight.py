#!/usr/bin/env python3
"""Read-only production preflight for JDV2 Stage-B V2-A."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from src.jdv2_audit import AUDIT_METRICS
from src.metrics import compute_metric_mask
from src.models import model as model_module
from src.models.joint_dependency_v2 import (
    DependencyCorrector,
    build_component_metadata,
    component_zero_mean_projection,
)
from src.utils import isolated_random_seed
from tools.audit_jdv2_stage_a import _active_evaluator
from tools.jdv2_stage_b_identity_contract_audit import _rng_equal
from tools.jdv2_stage_b_v1_failure_mechanism_audit import (
    _metric_append,
    _metric_finalize,
    checkpoint_contract,
    distribution,
    make_noise_tape,
    replay_branches,
    rng_restore,
    rng_snapshot,
    transform_delta,
    trunk_from_tape,
    velocities_to_predictions,
)


STAGE_A_SHA256 = (
    "699336d49aaecbccfaac8b44f00c0df5a73b521548fc29d3da37d071148fd5bb")
STAGE_B_SHA256 = (
    "e24a1cbfc5760db57bcb5cacd64c66a4afd1aed0f11bb1efefa8549933d21c1a")
DEPENDENCY_SOURCE_SHA256 = (
    "0ff28eb7107e49a0ad6f2bf339a65d8847f83c4a7cf315faa5a80cac4dc00522")
COUNTERFACTUAL_ATOL = 2e-6


def _max_abs(left, right):
    return float((left.float() - right.float()).abs().max().cpu())


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed_call(device, function):
    _sync(device)
    started = time.perf_counter()
    result = function()
    _sync(device)
    return result, time.perf_counter() - started


def _memory_peak(device):
    if device.type != "cuda":
        return {"allocated_bytes": 0, "reserved_bytes": 0}
    return {
        "allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def _component_diagnostics(raw, projected, metadata, store):
    common = raw.float() - projected.float()
    raw_energy = float(raw.float().square().sum().cpu())
    common_energy = float(common.square().sum().cpu())
    store["raw_rms"].append(float(raw.float().square().mean().sqrt().cpu()))
    store["projected_rms"].append(float(
        projected.float().square().mean().sqrt().cpu()))
    store["removed_common_rms"].append(float(
        common.square().mean().sqrt().cpu()))
    store["common_energy_fraction"].append(
        common_energy / max(raw_energy, 1e-30))
    maximum = 0.0
    for component in range(metadata.num_components):
        selected = metadata.component_id.eq(component)
        maximum = max(maximum, float(
            projected[selected].float().sum(0).abs().max().cpu()))
    store["component_zero_mean_max_abs"].append(maximum)
    store["num_active_components"].append(float(metadata.num_components))
    if metadata.num_components:
        counts = metadata.component_count.float().cpu().numpy()
        store["mean_component_size"].append(float(counts.mean()))
        store["max_component_size"].append(float(counts.max()))


@torch.no_grad()
def run_v1_and_counterfactual(evaluator, seed):
    """Run full V1 and production/audit component-centered paired parity."""
    net = evaluator.net
    net.eval()
    original_projection = getattr(
        net.args, "jdv2_residual_projection", "none")
    metrics = {
        name: defaultdict(lambda: defaultdict(list))
        for name in ("v1", "v2a_production", "audit_component_centered")}
    counters = {
        "windows": 0, "e0_windows": 0, "e_gt0_windows": 0,
        "mixed_windows": 0, "all_active_windows": 0,
        "metadata_build_calls": 0, "candidate_id_checks": 0,
        "goal_checks": 0,
    }
    maxima = {
        "v1_production_vs_reference": 0.0,
        "v2a_production_vs_counterfactual": 0.0,
        "projection_vs_counterfactual_residual": 0.0,
        "e0_v2a_vs_stage_a": 0.0,
        "mixed_degree_zero_v2a_vs_stage_a": 0.0,
        "reconstruction": 0.0,
    }
    rng_equal = True
    diagnostics = defaultdict(list)
    runtime = {"v1_seconds": [], "v2a_seconds": [],
               "metadata_seconds": [], "projection_seconds": []}
    memory = {"v1_allocated": [], "v1_reserved": [],
              "v2a_allocated": [], "v2a_reserved": []}

    try:
        with isolated_random_seed(
                int(seed), use_cuda=evaluator.device.type == "cuda"):
            for window_index, (batch_data, batch_id) in enumerate(
                    evaluator.data_loaders["valid"]):
                inputs, sequence = net.prepare_inputs(batch_data, batch_id)
                mask = compute_metric_mask(sequence)
                net.jdv2_sampler.set_sampling_context(seed, window_index)
                net.args.jdv2_residual_projection = "none"
                with evaluator._autocast_context():
                    contexts, auxiliary = net.encode(inputs, if_test=True)
                state = auxiliary["dependency_state"]
                edge = state["edge_index"].long()
                metadata, elapsed = _timed_call(
                    evaluator.device,
                    lambda: build_component_metadata(
                        edge, contexts[-1].size(0), inputs["scene_index"]))
                runtime["metadata_seconds"].append(elapsed)
                counters["metadata_build_calls"] += 1
                state["component_metadata"] = metadata
                active = metadata.active_mask
                counters["windows"] += 1
                if not bool(active.any()):
                    counters["e0_windows"] += 1
                else:
                    counters["e_gt0_windows"] += 1
                    if bool(active.all()):
                        counters["all_active_windows"] += 1
                    else:
                        counters["mixed_windows"] += 1

                ids = auxiliary["joint_candidate_index"][:, :20]
                goals = auxiliary["joint_goal_points_world"][:, :20]
                ids_before, goals_before = ids.clone(), goals.clone()
                pre = rng_snapshot(evaluator.device.type == "cuda")
                tape = make_noise_tape(net, contexts[-1])
                with evaluator._autocast_context():
                    middle = trunk_from_tape(net, contexts, tape)
                    reference_v1, raw_trace = replay_branches(
                        net, contexts, state, tape, "full_v1",
                        inputs["scene_index"], None, seed,
                        middle_result=middle)
                    reference_centered, centered_trace = replay_branches(
                        net, contexts, state, tape, "component_centered",
                        inputs["scene_index"], None, seed,
                        middle_result=middle)
                    reference_stage_a, _ = replay_branches(
                        net, contexts, state, tape, "stage_a",
                        inputs["scene_index"], None, seed,
                        middle_result=middle)

                # Alternate timing order to reduce monotonic thermal bias.
                policy_order = ("v1", "v2a") if window_index % 2 == 0 \
                    else ("v2a", "v1")
                produced = {}
                post_states = {}
                for policy in policy_order:
                    net.args.jdv2_residual_projection = (
                        "none" if policy == "v1" else "component_zero_mean")
                    rng_restore(pre)
                    if evaluator.device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(evaluator.device)
                    with evaluator._autocast_context():
                        produced[policy], elapsed = _timed_call(
                            evaluator.device,
                            lambda: net.ts_sample(contexts, state))
                    runtime[policy + "_seconds"].append(elapsed)
                    peak = _memory_peak(evaluator.device)
                    memory[policy + "_allocated"].append(
                        peak["allocated_bytes"])
                    memory[policy + "_reserved"].append(
                        peak["reserved_bytes"])
                    post_states[policy] = rng_snapshot(
                        evaluator.device.type == "cuda")
                rng_equal = rng_equal and _rng_equal(
                    post_states["v1"], post_states["v2a"])
                rng_restore(post_states["v2a"])

                maxima["v1_production_vs_reference"] = max(
                    maxima["v1_production_vs_reference"],
                    _max_abs(produced["v1"], reference_v1))
                maxima["v2a_production_vs_counterfactual"] = max(
                    maxima["v2a_production_vs_counterfactual"],
                    _max_abs(produced["v2a"], reference_centered))
                if not bool(active.any()):
                    maxima["e0_v2a_vs_stage_a"] = max(
                        maxima["e0_v2a_vs_stage_a"],
                        _max_abs(produced["v2a"], reference_stage_a))
                elif bool((~active).any()):
                    maxima["mixed_degree_zero_v2a_vs_stage_a"] = max(
                        maxima["mixed_degree_zero_v2a_vs_stage_a"],
                        _max_abs(produced["v2a"][:, ~active],
                                 reference_stage_a[:, ~active]))

                for branch, raw in enumerate(raw_trace):
                    projected, elapsed = _timed_call(
                        evaluator.device,
                        lambda raw=raw: component_zero_mean_projection(
                            raw, metadata))
                    runtime["projection_seconds"].append(elapsed)
                    audited_same_raw = transform_delta(
                        raw, "component_centered", edge,
                        inputs["scene_index"], branch)
                    maxima["projection_vs_counterfactual_residual"] = max(
                        maxima["projection_vs_counterfactual_residual"],
                        _max_abs(projected, audited_same_raw))
                    maxima["reconstruction"] = max(
                        maxima["reconstruction"],
                        _max_abs(raw.float(),
                                 projected.float() +
                                 (raw.float() - projected.float())))
                    _component_diagnostics(raw, projected, metadata,
                                           diagnostics)

                prediction = {
                    "v1": velocities_to_predictions(
                        net, inputs, produced["v1"]),
                    "v2a_production": velocities_to_predictions(
                        net, inputs, produced["v2a"]),
                    "audit_component_centered": velocities_to_predictions(
                        net, inputs, reference_centered),
                }
                edge_class = "E>0" if edge.shape[1] else "E=0"
                for name, value in prediction.items():
                    _metric_append(metrics[name], net, value, auxiliary,
                                   inputs, mask, edge_class)
                if not torch.equal(ids, ids_before):
                    raise AssertionError("candidate IDs changed")
                if not torch.equal(goals, goals_before):
                    raise AssertionError("joint goals changed")
                counters["candidate_id_checks"] += ids.numel()
                counters["goal_checks"] += goals.numel()
                del inputs, sequence, contexts, auxiliary, prediction, tape
    finally:
        net.args.jdv2_residual_projection = original_projection

    mean_v1 = float(np.mean(runtime["v1_seconds"]))
    mean_v2a = float(np.mean(runtime["v2a_seconds"]))
    return {
        "seed": int(seed),
        "counters": counters,
        "max_abs_differences": maxima,
        "rng_post_state_equal": rng_equal,
        "metrics": {name: _metric_finalize(value)
                    for name, value in metrics.items()},
        "diagnostics": {name: distribution(values)
                        for name, values in diagnostics.items()},
        "runtime": {
            "v1_end_to_end_seconds": distribution(runtime["v1_seconds"]),
            "v2a_end_to_end_seconds": distribution(runtime["v2a_seconds"]),
            "component_metadata_seconds": distribution(
                runtime["metadata_seconds"]),
            "projection_only_seconds": distribution(
                runtime["projection_seconds"]),
            "mean_end_to_end_overhead_fraction": (
                mean_v2a / mean_v1 - 1.0),
        },
        "cuda_memory": {name: distribution(values)
                        for name, values in memory.items()},
    }


@torch.no_grad()
def run_zero_init_identity(evaluator, seed):
    """Prove a fresh V2-A Stage-A parent is tensor-exact no-harm."""
    net = evaluator.net
    net.eval()
    maxima = {"overall": 0.0, "E=0": 0.0,
              "mixed_degree_zero": 0.0, "all_active": 0.0}
    counters = {"windows": 0, "E=0": 0, "mixed": 0, "all_active": 0}
    rng_equal = True
    with isolated_random_seed(
            int(seed), use_cuda=evaluator.device.type == "cuda"):
        for window_index, (batch_data, batch_id) in enumerate(
                evaluator.data_loaders["valid"]):
            inputs, _ = net.prepare_inputs(batch_data, batch_id)
            net.jdv2_sampler.set_sampling_context(seed, window_index)
            with evaluator._autocast_context():
                contexts, auxiliary = net.encode(inputs, if_test=True)
            state = auxiliary["dependency_state"]
            metadata = state["component_metadata"]
            active = metadata.active_mask
            pre = rng_snapshot(evaluator.device.type == "cuda")
            with evaluator._autocast_context():
                v2a = net.ts_sample(contexts, state)
            post_v2a = rng_snapshot(evaluator.device.type == "cuda")
            old = net.args.use_dependency_corrector
            try:
                net.args.use_dependency_corrector = False
                rng_restore(pre)
                with evaluator._autocast_context():
                    stage_a = net.ts_sample(contexts, state)
            finally:
                net.args.use_dependency_corrector = old
            post_stage_a = rng_snapshot(evaluator.device.type == "cuda")
            rng_equal = rng_equal and _rng_equal(post_v2a, post_stage_a)
            rng_restore(post_v2a)
            difference = _max_abs(v2a, stage_a)
            maxima["overall"] = max(maxima["overall"], difference)
            counters["windows"] += 1
            if not bool(active.any()):
                counters["E=0"] += 1
                maxima["E=0"] = max(maxima["E=0"], difference)
            elif bool(active.all()):
                counters["all_active"] += 1
                maxima["all_active"] = max(
                    maxima["all_active"], difference)
            else:
                counters["mixed"] += 1
                maxima["mixed_degree_zero"] = max(
                    maxima["mixed_degree_zero"],
                    _max_abs(v2a[:, ~active], stage_a[:, ~active]))
            if not torch.equal(v2a, stage_a):
                raise AssertionError("fresh V2-A is not tensor-exact Stage A")
    return {"seed": int(seed), "counters": counters,
            "max_abs_differences": maxima,
            "rng_post_state_equal": rng_equal,
            "tensor_exact": maxima["overall"] == 0.0 and rng_equal}


def run_backward_smoke(evaluator):
    """Run one disposable V2-A forward/backward/optimizer step."""
    net = evaluator.net
    net.train(True)
    parameters = [(name, value) for name, value in net.named_parameters()
                  if value.requires_grad]
    if not parameters or not all(
            name.startswith("jdv2_corrector.") for name, _ in parameters):
        raise AssertionError("trainable parameter isolation failed")
    before = {name: value.detach().clone() for name, value in parameters}
    optimizer = torch.optim.Adam([value for _, value in parameters], lr=1e-4)
    selected = None
    for batch_index, (batch_data, batch_id) in enumerate(
            evaluator.data_loaders["train"]):
        inputs, _ = net.prepare_inputs(batch_data, batch_id)
        if inputs["jdv2_cache"]["edge_index"].shape[1]:
            selected = (batch_index, inputs)
            break
    if selected is None:
        raise RuntimeError("no E>0 training window for backward smoke")
    batch_index, inputs = selected
    net.set_jdv2_training_sampling_context(
        evaluator.args.seed, 1, batch_index,
        len(evaluator.data_loaders["train"]))
    with evaluator._autocast_context():
        losses = net._jdv2_dependency_losses(inputs)
        total = sum(net.set_losses_coeffs()[name] * value
                    for name, value in losses.items())
    optimizer.zero_grad(set_to_none=True)
    total.backward()
    corrector_gradients = {
        name: value.grad for name, value in net.named_parameters()
        if name.startswith("jdv2_corrector.")}
    finite = all(gradient is None or torch.isfinite(gradient).all()
                 for gradient in corrector_gradients.values())
    nonzero = any(gradient is not None and bool(torch.count_nonzero(gradient))
                  for gradient in corrector_gradients.values())
    upstream_none = all(
        value.grad is None for name, value in net.named_parameters()
        if not name.startswith("jdv2_corrector."))
    optimizer.step()
    changed = [name for name, value in parameters
               if not torch.equal(before[name], value.detach())]
    net.eval()
    return {
        "losses": {name: float(value.detach().cpu())
                   for name, value in losses.items()},
        "total_loss": float(total.detach().cpu()),
        "trainable_parameter_count": sum(
            value.numel() for _, value in parameters),
        "only_corrector_trainable": all(
            name.startswith("jdv2_corrector.") for name, _ in parameters),
        "finite_corrector_gradients": bool(finite),
        "nonzero_corrector_gradient": bool(nonzero),
        "stage_a_gradients_all_none": bool(upstream_none),
        "changed_parameters_only_corrector": bool(
            changed and all(name.startswith("jdv2_corrector.")
                            for name in changed)),
        "optimizer_step_disposable": True,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1-config", required=True)
    parser.add_argument("--v2a-config", required=True)
    parser.add_argument("--stage-a-checkpoint", required=True)
    parser.add_argument("--stage-b-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2035)
    return parser.parse_args()


def main():
    args = parse_args()
    output = Path(args.output)
    stage_a = Path(args.stage_a_checkpoint)
    stage_b = Path(args.stage_b_checkpoint)
    contract = checkpoint_contract(stage_a, stage_b)
    if contract["stage_a_sha256"] != STAGE_A_SHA256:
        raise RuntimeError("Stage-A checkpoint hash changed")
    if contract["stage_b_sha256"] != STAGE_B_SHA256:
        raise RuntimeError("Stage-B checkpoint hash changed")
    source_hash = checkpoint_contract(
        Path("src/models/joint_dependency_v2/dependency_corrector.py"),
        Path("src/models/joint_dependency_v2/dependency_corrector.py"))[
            "stage_a_sha256"]
    if source_hash != DEPENDENCY_SOURCE_SHA256:
        raise RuntimeError("DependencyCorrector source changed")

    result = {
        "status": "RUNNING",
        "protocol": {
            "split": "ETH validation", "windows": 139,
            "seed": int(args.seed), "precision": "CUDA/BF16",
            "paired_noise_tape": True,
            "counterfactual_atol": COUNTERFACTUAL_ATOL,
            "formal_training_performed": False,
        },
        "provenance": {
            **contract,
            "dependency_corrector_source_sha256": source_hash,
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True).strip(),
        },
    }

    def persist():
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    persist()
    v1_evaluator, v1_epoch = _active_evaluator(
        args.v1_config, str(stage_b), output.parent / "runtime_v1",
        args.device)
    result["v1_checkpoint_epoch"] = int(v1_epoch)
    result["v1_and_counterfactual"] = run_v1_and_counterfactual(
        v1_evaluator, args.seed)
    persist()

    v2a_evaluator, stage_a_epoch = _active_evaluator(
        args.v2a_config, str(stage_a), output.parent / "runtime_v2a_init",
        args.device)
    result["stage_a_parent_epoch"] = int(stage_a_epoch)
    metadata_calls = 0
    production_builder = model_module.build_component_metadata

    def counted_builder(*builder_args, **builder_kwargs):
        nonlocal metadata_calls
        metadata_calls += 1
        return production_builder(*builder_args, **builder_kwargs)

    model_module.build_component_metadata = counted_builder
    try:
        result["zero_init_identity"] = run_zero_init_identity(
            v2a_evaluator, args.seed)
    finally:
        model_module.build_component_metadata = production_builder
    result["zero_init_identity"]["metadata_build_calls"] = metadata_calls
    result["zero_init_identity"]["metadata_once_per_window"] = (
        metadata_calls == result["zero_init_identity"]["counters"]["windows"])
    result["backward_smoke"] = run_backward_smoke(v2a_evaluator)
    persist()

    resume_rejected = False
    resume_error = None
    try:
        _active_evaluator(
            args.v2a_config, str(stage_b),
            output.parent / "runtime_v1_resume_rejection", args.device)
    except RuntimeError as error:
        resume_rejected = True
        resume_error = str(error)
    result["checkpoint_contract"] = {
        "stage_a_parent_accepted": int(stage_a_epoch) == 13,
        "v1_training_resume_rejected": resume_rejected,
        "v1_rejection_reason": resume_error,
        "future_fields": {
            "training_stage": "joint_trajectory",
            "stage_b_architecture_version": "jdv2-stage-b-v2a",
            "stage_b_residual_projection": "component_zero_mean",
        },
    }

    parity = result["v1_and_counterfactual"]
    overhead = parity["runtime"]["mean_end_to_end_overhead_fraction"]
    gates = {
        "projection_mathematics": (
            parity["max_abs_differences"]["reconstruction"] <=
            COUNTERFACTUAL_ATOL and
            parity["diagnostics"]["component_zero_mean_max_abs"]["max"] <=
            2e-5),
        "corrector_parameter_count_30851": sum(
            value.numel() for value in DependencyCorrector().parameters()) ==
            30851,
        "v1_tensor_exact": parity["max_abs_differences"][
            "v1_production_vs_reference"] == 0.0,
        "stage_a_zero_init_tensor_exact": result[
            "zero_init_identity"]["tensor_exact"],
        "e0_tensor_exact": parity["max_abs_differences"][
            "e0_v2a_vs_stage_a"] == 0.0,
        "mixed_degree_zero_tensor_exact": parity["max_abs_differences"][
            "mixed_degree_zero_v2a_vs_stage_a"] == 0.0,
        "metadata_once_per_window": (
            parity["counters"]["metadata_build_calls"] ==
            parity["counters"]["windows"] and
            result["zero_init_identity"]["metadata_once_per_window"]),
        "rng_unchanged": (parity["rng_post_state_equal"] and result[
            "zero_init_identity"]["rng_post_state_equal"]),
        "stage_a_gradients_absent": result["backward_smoke"][
            "stage_a_gradients_all_none"],
        "v1_resume_rejected": resume_rejected,
        "stage_a_parent_epoch13": int(stage_a_epoch) == 13,
        "cuda_bf16_backward": all((
            result["backward_smoke"]["finite_corrector_gradients"],
            result["backward_smoke"]["nonzero_corrector_gradient"],
            result["backward_smoke"]["changed_parameters_only_corrector"])),
        "counterfactual_parity": parity["max_abs_differences"][
            "v2a_production_vs_counterfactual"] <= COUNTERFACTUAL_ATOL and
            parity["max_abs_differences"][
                "projection_vs_counterfactual_residual"] <=
            COUNTERFACTUAL_ATOL,
        "runtime_overhead_le_5_percent": overhead <= 0.05,
    }
    result["hard_gates"] = gates
    result["status"] = ("STAGE_B_V2A_READY_FOR_TRAINING"
                        if all(gates.values())
                        else "STAGE_B_V2A_IMPLEMENTATION_BLOCKED")
    persist()
    print(json.dumps({"status": result["status"],
                      "hard_gates": gates,
                      "overhead": overhead,
                      "max_abs": parity["max_abs_differences"]},
                     indent=2), flush=True)
    if result["status"] != "STAGE_B_V2A_READY_FOR_TRAINING":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
