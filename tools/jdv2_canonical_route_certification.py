#!/usr/bin/env python3
"""Paired read-only certification of canonical Stage-A numerical routing.

The audit compares three paths under identical Stage-A worlds and an explicit
diffusion NoiseTape:

* A: the frozen Stage-A checkpoint with its canonical corrector=True config;
* B: the same Stage-A checkpoint with correction arithmetic disabled;
* C: the frozen Stage-B V2-A checkpoint with correction arithmetic disabled.

No optimizer is constructed and no checkpoint or production configuration is
modified.  Path C is the literal Stage-A reference used by Stage-B paired
evaluation; it is not the corrected Stage-B prediction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from src.jdv2_audit import AUDIT_METRICS
from src.metrics import compute_metric_mask
from src.utils import isolated_random_seed
from tools.audit_jdv2_stage_a import _active_evaluator
from tools.jdv2_stage_b_identity_contract_audit import _rng_equal
from tools.jdv2_stage_b_v1_failure_mechanism_audit import (
    _metric_append,
    _metric_finalize,
    connected_components,
    make_noise_tape,
    rng_restore,
    rng_snapshot,
    trunk_from_tape,
    velocities_to_predictions,
)


STAGE_A_SHA256 = (
    "699336d49aaecbccfaac8b44f00c0df5a73b521548fc29d3da37d071148fd5bb")
STAGE_B_V2A_SHA256 = (
    "e4c114c729ba8d75ac72fc7d41f0f05e790cf2aec7b5cade563ba792930c7b73")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_comparison(left: torch.Tensor, right: torch.Tensor) -> dict:
    if left.shape != right.shape:
        return {
            "shape_equal": False,
            "left_shape": list(left.shape),
            "right_shape": list(right.shape),
            "left_dtype": str(left.dtype),
            "right_dtype": str(right.dtype),
        }
    left_float = left.detach().float()
    right_float = right.detach().float()
    difference = (left_float - right_float).abs()
    different = left_float.ne(right_float)
    return {
        "shape_equal": True,
        "left_shape": list(left.shape),
        "right_shape": list(right.shape),
        "left_dtype": str(left.dtype),
        "right_dtype": str(right.dtype),
        "dtype_equal": left.dtype == right.dtype,
        "tensor_equal": left.dtype == right.dtype and torch.equal(left, right),
        "max_abs_difference": (
            float(difference.max().cpu()) if difference.numel() else 0.0),
        "different_elements": int(different.sum().cpu()),
        "num_elements": int(different.numel()),
    }


def _empty_parity() -> dict:
    return {
        "max_abs_difference": 0.0,
        "different_elements": 0,
        "compared_elements": 0,
        "nonidentical_windows": 0,
    }


def _update_parity(target: dict, left: torch.Tensor,
                   right: torch.Tensor) -> None:
    result = tensor_comparison(left, right)
    if not result["shape_equal"]:
        raise ValueError("paired tensors have different shapes")
    target["max_abs_difference"] = max(
        target["max_abs_difference"], result["max_abs_difference"])
    target["different_elements"] += result["different_elements"]
    target["compared_elements"] += result["num_elements"]
    target["nonidentical_windows"] += int(not result["tensor_equal"])


def _checkpoint_state_contract(stage_a: Path, stage_b: Path) -> dict:
    checkpoint_a = torch.load(stage_a, map_location="cpu")
    checkpoint_b = torch.load(stage_b, map_location="cpu")
    state_a = checkpoint_a["model_state_dict"]
    state_b = checkpoint_b["model_state_dict"]
    if set(state_a) != set(state_b):
        missing = sorted(set(state_a) - set(state_b))
        extra = sorted(set(state_b) - set(state_a))
        return {"key_sets_equal": False, "missing": missing, "extra": extra}
    changed = [
        name for name in state_a
        if not torch.equal(state_a[name], state_b[name])]
    noncorrector_changed = [
        name for name in changed
        if not name.startswith("jdv2_corrector.")]
    output_weight = state_a["jdv2_corrector.output.2.weight"]
    output_bias = state_a["jdv2_corrector.output.2.bias"]
    return {
        "key_sets_equal": True,
        "stage_a_epoch": int(checkpoint_a["epoch"]),
        "stage_b_epoch": int(checkpoint_b["epoch"]),
        "changed_tensor_count": len(changed),
        "changed_corrector_tensor_count": len(changed) - len(
            noncorrector_changed),
        "noncorrector_changed_tensors": noncorrector_changed,
        "stage_a_output_weight_nonzero": int(
            torch.count_nonzero(output_weight)),
        "stage_a_output_bias_nonzero": int(
            torch.count_nonzero(output_bias)),
        "stage_a_output_layer_exact_zero": bool(
            torch.count_nonzero(output_weight) == 0 and
            torch.count_nonzero(output_bias) == 0),
    }


def _run_ts_sample(evaluator, contexts, state, pre_rng,
                   use_corrector: bool):
    net = evaluator.net
    old_flag = net.args.use_dependency_corrector
    try:
        net.args.use_dependency_corrector = bool(use_corrector)
        rng_restore(pre_rng)
        with evaluator._autocast_context():
            output = net.ts_sample(contexts, state)
        post_rng = rng_snapshot(evaluator.device.type == "cuda")
    finally:
        net.args.use_dependency_corrector = old_flag
    return output, post_rng


@torch.no_grad()
def trace_zero_residual_divergence(evaluator, contexts, state, tape) -> dict:
    """Locate the first dtype and value divergence between paths A and B."""
    net = evaluator.net
    middle = trunk_from_tape(net, contexts, tape)
    edge_index = state["edge_index"].long()
    _components, degree = connected_components(middle.shape[0], edge_index)
    active = degree.gt(0)
    if not active.any():
        raise ValueError("zero-residual divergence trace requires E>0")
    schedule = np.linspace(
        net.var_sched.num_steps, 0, net.args.ddim_step + 1)
    simple_var = net.args.dataset in {"eth5", "ind"}
    eta = 1 if simple_var else 0
    first_dtype = None
    first_value = None
    compact_records = []
    final_a = []
    final_b = []
    for branch in range(net.args.num_samples):
        x_a = middle
        x_b = middle
        context = contexts[branch]
        for noise_index, step in enumerate(range(
                int(net.args.branch_stage_step + 1),
                int(net.args.ddim_step + 1))):
            current_t = int(schedule[step - 1])
            previous_t = int(schedule[step])
            alpha_current = net.var_sched.alpha_bars[current_t]
            alpha_previous = (net.var_sched.alpha_bars[previous_t]
                              if previous_t >= 0 else 1)
            beta = net.var_sched.betas[[current_t] * context.size(0)]
            epsilon_a = net.diffnet(x_a, beta=beta, context=context)
            epsilon_b = net.diffnet(x_b, beta=beta, context=context)

            last_map = state["last_position_map"]
            position_map = last_map[:, None] + torch.cumsum(x_a, dim=1)
            position_world = state["scene"].make_world_coord_torch(
                position_map * float(net.args.down_factor))
            anchor = state["last_position_world"][:, None]
            noisy_world_velocity = torch.diff(
                torch.cat((anchor.float(), position_world.float()), dim=1),
                dim=1) / float(net.args.trajectory_dt)
            delta = net.jdv2_corrector(
                noisy_world_velocity, state["last_position_world"],
                edge_index, state["relation_embedding"][:, branch],
                torch.tensor(current_t, device=x_a.device),
                edge_weight=state["edge_weight"])
            if torch.count_nonzero(delta):
                raise AssertionError("Stage-A corrector residual is not zero")
            epsilon_joint = epsilon_a + delta

            variance = eta * (1 - alpha_previous) / (1 - alpha_current) * \
                (1 - alpha_current / alpha_previous)
            coefficient = ((1 - alpha_previous - variance) ** 0.5 -
                           (alpha_previous * (1 - alpha_current) /
                            alpha_current) ** 0.5)
            first_a = (alpha_previous / alpha_current) ** 0.5 * x_a
            first_b = (alpha_previous / alpha_current) ** 0.5 * x_b
            second_a = coefficient * epsilon_joint
            second_b = coefficient * epsilon_b
            noise = tape.branch_noise[branch][noise_index]
            third = ((1 - alpha_current / alpha_previous) ** 0.5 * noise
                     if simple_var else variance ** 0.5 * noise)
            corrected_next = first_a + second_a + third
            baseline_next = first_b + second_b + third
            if active.all():
                x_a_next = corrected_next
            else:
                same_state_base = first_a + coefficient * epsilon_a + third
                x_a_next = torch.where(
                    active[:, None, None], corrected_next, same_state_base)
            x_b_next = baseline_next

            operators = (
                ("x_t", x_a, x_b),
                ("epsilon_base", epsilon_a, epsilon_b),
                ("epsilon_base_plus_zero", epsilon_joint, epsilon_b),
                ("first_term", first_a, first_b),
                ("second_term", second_a, second_b),
                ("third_term", third, third),
                ("x_next", x_a_next, x_b_next),
            )
            for operator, left, right in operators:
                comparison = tensor_comparison(left[active], right[active])
                record = {
                    "branch": branch,
                    "diffusion_timestep": current_t,
                    "operator": operator,
                    **comparison,
                }
                if first_dtype is None and not comparison["dtype_equal"]:
                    first_dtype = record
                if (first_value is None and
                        comparison["different_elements"] > 0):
                    first_value = record
                if branch == 0 and noise_index == 0:
                    compact_records.append(record)
            x_a, x_b = x_a_next, x_b_next
        final_a.append(x_a)
        final_b.append(x_b)
    return {
        "window_active_agents": int(active.sum().cpu()),
        "window_total_agents": int(active.numel()),
        "first_dtype_divergence": first_dtype,
        "first_value_divergence": first_value,
        "first_branch_step_records": compact_records,
        "final_path_comparison": tensor_comparison(
            torch.stack(final_a), torch.stack(final_b)),
        "trace_path_a": torch.stack(final_a),
        "trace_path_b": torch.stack(final_b),
    }


def _parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-a-config", required=True)
    parser.add_argument("--stage-b-config", required=True)
    parser.add_argument("--stage-a-checkpoint", required=True)
    parser.add_argument("--stage-b-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2035)
    parser.add_argument(
        "--precision", choices=("bf16", "fp32"), default="bf16")
    return parser.parse_args()


def _batch_identity(batch_id) -> str:
    return repr(batch_id)


def classify_status(invariants: dict, parity: dict) -> str:
    """Classify only after the paired input and reference contracts pass."""
    if not all(invariants.values()):
        return "PAIRING_INPUT_CONTRACT_FAILED"
    if parity["B_vs_C"]["all_agents"]["different_elements"] != 0:
        return "STAGE_B_REFERENCE_PATH_MISMATCH"
    if parity["A_vs_B"]["all_agents"]["different_elements"] != 0:
        return "CANONICAL_STAGE_A_ZERO_ADD_DRIFT_CONFIRMED"
    return "CANONICAL_STAGE_A_LITERAL_IDENTITY_CONFIRMED"


def main() -> None:
    cli = _parse_cli()
    stage_a_path = Path(cli.stage_a_checkpoint)
    stage_b_path = Path(cli.stage_b_checkpoint)
    if sha256(stage_a_path) != STAGE_A_SHA256:
        raise RuntimeError("Stage-A checkpoint hash changed")
    if sha256(stage_b_path) != STAGE_B_V2A_SHA256:
        raise RuntimeError("Stage-B V2-A checkpoint hash changed")
    output = Path(cli.output)
    runtime_root = output.parent / "runtime"
    evaluator_a, epoch_a = _active_evaluator(
        cli.stage_a_config, str(stage_a_path), runtime_root / "stage_a",
        cli.device)
    evaluator_c, epoch_c = _active_evaluator(
        cli.stage_b_config, str(stage_b_path), runtime_root / "stage_b",
        cli.device)
    net_a, net_c = evaluator_a.net, evaluator_c.net
    if cli.precision == "fp32":
        evaluator_a.amp_enabled = False
        evaluator_c.amp_enabled = False
        evaluator_a.amp_dtype = torch.float32
        evaluator_c.amp_dtype = torch.float32
    net_a.eval()
    net_c.eval()
    loader_a = evaluator_a.data_loaders["valid"]
    loader_c = evaluator_c.data_loaders["valid"]
    if len(loader_a) != len(loader_c):
        raise RuntimeError("Stage-A and Stage-B validation lengths differ")

    metrics = {
        name: defaultdict(lambda: defaultdict(list))
        for name in ("A_canonical", "B_literal_off", "C_stage_b_reference")}
    parity = {
        pair: {scope: _empty_parity()
               for scope in (
                   "all_agents", "active_agents", "inactive_agents",
                   "E=0_windows", "mixed_windows", "all_active_windows")}
        for pair in ("A_vs_B", "B_vs_C", "A_vs_C")
    }
    independent_relation_parity = _empty_parity()
    counts = {
        "windows": 0, "e0_windows": 0, "e_gt0_windows": 0,
        "mixed_windows": 0, "all_active_windows": 0,
        "candidate_id_elements": 0, "goal_elements": 0,
        "context_elements": 0, "relation_elements": 0,
    }
    invariants = {
        "batch_identity_equal": True,
        "input_tensor_equal": True,
        "candidate_ids_equal": True,
        "joint_goals_equal": True,
        "contexts_equal": True,
        "paired_relation_state_shared": True,
        "encode_rng_post_state_equal": True,
        "diffusion_rng_post_state_equal": True,
        "coverage_20_of_20": True,
    }
    first_trace = None
    started = time.perf_counter()
    if torch.cuda.is_available() and str(cli.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(torch.device(cli.device))

    with isolated_random_seed(
            cli.seed, use_cuda=evaluator_a.device.type == "cuda"):
        for window_index, (row_a, row_c) in enumerate(zip(loader_a, loader_c)):
            batch_a, batch_id_a = row_a
            batch_c, batch_id_c = row_c
            invariants["batch_identity_equal"] &= (
                _batch_identity(batch_id_a) == _batch_identity(batch_id_c))
            inputs_a, seq_a = net_a.prepare_inputs(batch_a, batch_id_a)
            inputs_c, seq_c = net_c.prepare_inputs(batch_c, batch_id_c)
            for key in ("x_augmented", "scene_index", "world_coord"):
                invariants["input_tensor_equal"] &= torch.equal(
                    inputs_a[key], inputs_c[key])
            invariants["input_tensor_equal"] &= torch.equal(seq_a, seq_c)

            net_a.jdv2_sampler.set_sampling_context(cli.seed, window_index)
            net_c.jdv2_sampler.set_sampling_context(cli.seed, window_index)
            pre_encode = rng_snapshot(evaluator_a.device.type == "cuda")
            with evaluator_a._autocast_context():
                contexts_a, auxiliary_a = net_a.encode(
                    inputs_a, if_test=True)
            post_encode_a = rng_snapshot(evaluator_a.device.type == "cuda")
            rng_restore(pre_encode)
            with evaluator_c._autocast_context():
                contexts_c, auxiliary_c = net_c.encode(
                    inputs_c, if_test=True)
            post_encode_c = rng_snapshot(evaluator_a.device.type == "cuda")
            invariants["encode_rng_post_state_equal"] &= _rng_equal(
                post_encode_a, post_encode_c)
            rng_restore(post_encode_a)

            ids_a = auxiliary_a["joint_candidate_index"][:, :20]
            ids_c = auxiliary_c["joint_candidate_index"][:, :20]
            goals_a = auxiliary_a["joint_goal_points_world"][:, :20]
            goals_c = auxiliary_c["joint_goal_points_world"][:, :20]
            relation_a = auxiliary_a["dependency_state"][
                "relation_embedding"]
            relation_c = auxiliary_c["dependency_state"][
                "relation_embedding"]
            invariants["candidate_ids_equal"] &= torch.equal(ids_a, ids_c)
            invariants["joint_goals_equal"] &= torch.equal(goals_a, goals_c)
            invariants["contexts_equal"] &= torch.equal(
                contexts_a, contexts_c)
            _update_parity(
                independent_relation_parity, relation_a, relation_c)
            invariants["coverage_20_of_20"] &= all(
                torch.unique(row).numel() == 20 for row in ids_a)
            counts["candidate_id_elements"] += ids_a.numel()
            counts["goal_elements"] += goals_a.numel()
            counts["context_elements"] += contexts_a.numel()
            counts["relation_elements"] += relation_a.numel()

            state_a = auxiliary_a["dependency_state"]
            state_c = auxiliary_c["dependency_state"]
            _components, degree = connected_components(
                ids_a.shape[0], state_a["edge_index"])
            active = degree.gt(0)
            any_active = bool(active.any())
            all_active = bool(active.all())
            counts["windows"] += 1
            counts["e_gt0_windows" if any_active else "e0_windows"] += 1
            if any_active and not all_active:
                counts["mixed_windows"] += 1
            if all_active:
                counts["all_active_windows"] += 1

            pre_diffusion = rng_snapshot(evaluator_a.device.type == "cuda")
            velocity_a, post_a = _run_ts_sample(
                evaluator_a, contexts_a, state_a, pre_diffusion, True)
            velocity_b, post_b = _run_ts_sample(
                evaluator_a, contexts_a, state_a, pre_diffusion, False)
            velocity_c, post_c = _run_ts_sample(
                evaluator_c, contexts_a, state_a, pre_diffusion, False)
            invariants["diffusion_rng_post_state_equal"] &= (
                _rng_equal(post_a, post_b) and _rng_equal(post_b, post_c))
            rng_restore(post_a)

            paired = {
                "A_vs_B": (velocity_a, velocity_b),
                "B_vs_C": (velocity_b, velocity_c),
                "A_vs_C": (velocity_a, velocity_c),
            }
            for pair, (left, right) in paired.items():
                _update_parity(parity[pair]["all_agents"], left, right)
                if any_active:
                    _update_parity(
                        parity[pair]["active_agents"],
                        left[:, active], right[:, active])
                if not all_active:
                    _update_parity(
                        parity[pair]["inactive_agents"],
                        left[:, ~active], right[:, ~active])
                if not any_active:
                    _update_parity(
                        parity[pair]["E=0_windows"], left, right)
                elif all_active:
                    _update_parity(
                        parity[pair]["all_active_windows"], left, right)
                else:
                    _update_parity(
                        parity[pair]["mixed_windows"], left, right)

            if (first_trace is None and any_active and
                    not torch.equal(velocity_a, velocity_b)):
                rng_restore(pre_diffusion)
                tape = make_noise_tape(net_a, contexts_a[-1])
                with evaluator_a._autocast_context():
                    trace = trace_zero_residual_divergence(
                        evaluator_a, contexts_a, state_a, tape)
                trace_a = trace.pop("trace_path_a")
                trace_b = trace.pop("trace_path_b")
                trace["window_index"] = window_index
                trace["trace_vs_production_A"] = tensor_comparison(
                    trace_a, velocity_a)
                trace["trace_vs_production_B"] = tensor_comparison(
                    trace_b, velocity_b)
                first_trace = trace
                rng_restore(post_a)

            prediction_a = velocities_to_predictions(
                net_a, inputs_a, velocity_a)
            prediction_b = velocities_to_predictions(
                net_a, inputs_a, velocity_b)
            prediction_c = velocities_to_predictions(
                net_c, inputs_a, velocity_c)
            metric_mask = compute_metric_mask(seq_a)
            stratum = "E>0" if any_active else "E=0"
            _metric_append(
                metrics["A_canonical"], net_a, prediction_a, auxiliary_a,
                inputs_a, metric_mask, stratum)
            _metric_append(
                metrics["B_literal_off"], net_a, prediction_b, auxiliary_a,
                inputs_a, metric_mask, stratum)
            _metric_append(
                metrics["C_stage_b_reference"], net_c, prediction_c,
                auxiliary_a, inputs_a, metric_mask, stratum)

    state_contract = _checkpoint_state_contract(stage_a_path, stage_b_path)
    status = classify_status(invariants, parity)
    result = {
        "status": status,
        "scope": "read-only paired numerical-route certification",
        "paths": {
            "A_canonical": (
                "Stage-A checkpoint, canonical use_dependency_corrector=true"),
            "B_literal_off": (
                "same Stage-A checkpoint, correction arithmetic disabled"),
            "C_stage_b_reference": (
                "Stage-B V2-A checkpoint, correction arithmetic disabled"),
        },
        "protocol": {
            "split": "ETH validation", "windows": len(loader_a),
            "seed": cli.seed,
            "precision": f"{str(cli.device).upper()}/{cli.precision.upper()}",
            "paired_goals": True, "paired_noise_tape": True,
            "paired_context_and_relation_state": True,
            "metrics": list(AUDIT_METRICS),
        },
        "provenance": {
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True).strip(),
            "stage_a_config": str(Path(cli.stage_a_config).resolve()),
            "stage_a_config_sha256": sha256(Path(cli.stage_a_config)),
            "stage_b_config": str(Path(cli.stage_b_config).resolve()),
            "stage_b_config_sha256": sha256(Path(cli.stage_b_config)),
            "stage_a_checkpoint": str(stage_a_path.resolve()),
            "stage_a_checkpoint_sha256": sha256(stage_a_path),
            "stage_b_checkpoint": str(stage_b_path.resolve()),
            "stage_b_checkpoint_sha256": sha256(stage_b_path),
            "stage_a_epoch": int(epoch_a),
            "stage_b_epoch": int(epoch_c),
        },
        "checkpoint_state_contract": state_contract,
        "counts": counts,
        "invariants": invariants,
        "independent_recomputation_diagnostic": {
            "relation_embedding_A_vs_C": independent_relation_parity,
            "note": (
                "C receives the literal Stage-A context/relation state. This "
                "field measures a separate CUDA recomputation and is not a "
                "route-input gate because correction is disabled in C."),
        },
        "trajectory_parity": parity,
        "first_divergence_trace": first_trace,
        "metrics": {
            name: _metric_finalize(store) for name, store in metrics.items()},
        "runtime_seconds": float(time.perf_counter() - started),
        "peak_cuda_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(torch.device(cli.device)))
            if torch.cuda.is_available() and str(cli.device).startswith("cuda")
            else 0),
        "source_modified": False,
        "checkpoints_modified": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": status,
        "counts": counts,
        "invariants": invariants,
        "A_vs_B": parity["A_vs_B"]["all_agents"],
        "B_vs_C": parity["B_vs_C"]["all_agents"],
        "first_dtype_divergence": (
            first_trace["first_dtype_divergence"] if first_trace else None),
        "first_value_divergence": (
            first_trace["first_value_divergence"] if first_trace else None),
        "output": str(output),
    }, indent=2), flush=True)
    if status in {"PAIRING_INPUT_CONTRACT_FAILED",
                  "STAGE_B_REFERENCE_PATH_MISMATCH"}:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
