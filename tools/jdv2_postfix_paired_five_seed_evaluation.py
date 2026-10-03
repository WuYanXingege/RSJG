#!/usr/bin/env python3
"""Paired C-reference versus corrected V2-A five-seed evaluation.

Path C loads the V2-A checkpoint but disables correction arithmetic.  Path D
loads the same checkpoint and enables the trained component-relative
corrector.  Consequently C is a Stage-A base reference, while D is the actual
deployed V2-A policy.  The two paths share goals, contexts, relation state,
and the complete diffusion RNG stream.
"""

from __future__ import annotations

import argparse
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
from tools.jdv2_canonical_route_certification import (
    STAGE_A_SHA256,
    STAGE_B_V2A_SHA256,
    _checkpoint_state_contract,
    _empty_parity,
    _run_ts_sample,
    _update_parity,
    sha256,
)
from tools.jdv2_stage_b_identity_contract_audit import _rng_equal
from tools.jdv2_stage_b_v1_failure_mechanism_audit import (
    _metric_append,
    _metric_finalize,
    connected_components,
    rng_restore,
    rng_snapshot,
    velocities_to_predictions,
)


PATHS = ("C_stage_b_reference_off", "D_stage_b_v2a_corrected")


def distribution(values) -> dict:
    array = np.asarray(list(values), dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
        "values": [float(value) for value in array],
    }


def summarize_runs(runs: list[dict]) -> dict:
    return {
        path: {
            stratum: {
                metric: distribution(
                    run["metrics"][path][stratum][metric] for run in runs)
                for metric in AUDIT_METRICS
            }
            for stratum in ("overall", "E=0", "E>0")
        }
        for path in PATHS
    }


def summarize_paired_delta(runs: list[dict]) -> dict:
    return {
        stratum: {
            metric: distribution(
                run["metrics"]["D_stage_b_v2a_corrected"][stratum][metric] -
                run["metrics"]["C_stage_b_reference_off"][stratum][metric]
                for run in runs)
            for metric in AUDIT_METRICS
        }
        for stratum in ("overall", "E=0", "E>0")
    }


@torch.no_grad()
def evaluate_seed(evaluator, seed: int) -> dict:
    net = evaluator.net
    net.eval()
    metrics = {
        path: defaultdict(lambda: defaultdict(list)) for path in PATHS}
    parity = {
        scope: _empty_parity() for scope in (
            "all_agents", "active_agents", "inactive_agents",
            "E=0_windows", "mixed_windows", "all_active_windows")}
    counts = {
        "windows": 0, "E=0_windows": 0, "mixed_windows": 0,
        "all_active_windows": 0, "candidate_id_elements": 0,
        "goal_elements": 0,
    }
    invariants = {
        "diffusion_rng_post_state_equal": True,
        "candidate_ids_unchanged": True,
        "joint_goals_unchanged": True,
        "coverage_20_of_20": True,
        "E=0_tensor_exact": True,
        "mixed_inactive_tensor_exact": True,
    }
    with isolated_random_seed(
            int(seed), use_cuda=evaluator.device.type == "cuda"):
        for window_index, (batch_data, batch_id) in enumerate(
                evaluator.data_loaders["valid"]):
            inputs, sequence = net.prepare_inputs(batch_data, batch_id)
            mask = compute_metric_mask(sequence)
            net.jdv2_sampler.set_sampling_context(seed, window_index)
            with evaluator._autocast_context():
                contexts, auxiliary = net.encode(inputs, if_test=True)
            state = auxiliary["dependency_state"]
            ids = auxiliary["joint_candidate_index"][:, :20].clone()
            goals = auxiliary["joint_goal_points_world"][:, :20].clone()
            _components, degree = connected_components(
                ids.shape[0], state["edge_index"])
            active = degree.gt(0)
            any_active = bool(active.any())
            all_active = bool(active.all())
            counts["windows"] += 1
            if not any_active:
                counts["E=0_windows"] += 1
            elif all_active:
                counts["all_active_windows"] += 1
            else:
                counts["mixed_windows"] += 1

            pre_diffusion = rng_snapshot(evaluator.device.type == "cuda")
            reference, post_reference = _run_ts_sample(
                evaluator, contexts, state, pre_diffusion, False)
            corrected, post_corrected = _run_ts_sample(
                evaluator, contexts, state, pre_diffusion, True)
            invariants["diffusion_rng_post_state_equal"] &= _rng_equal(
                post_reference, post_corrected)
            rng_restore(post_corrected)

            _update_parity(parity["all_agents"], reference, corrected)
            if any_active:
                _update_parity(
                    parity["active_agents"], reference[:, active],
                    corrected[:, active])
            if not all_active:
                _update_parity(
                    parity["inactive_agents"], reference[:, ~active],
                    corrected[:, ~active])
            if not any_active:
                _update_parity(parity["E=0_windows"], reference, corrected)
                invariants["E=0_tensor_exact"] &= torch.equal(
                    reference, corrected)
            elif all_active:
                _update_parity(
                    parity["all_active_windows"], reference, corrected)
            else:
                _update_parity(parity["mixed_windows"], reference, corrected)
                invariants["mixed_inactive_tensor_exact"] &= torch.equal(
                    reference[:, ~active], corrected[:, ~active])

            prediction_reference = velocities_to_predictions(
                net, inputs, reference)
            prediction_corrected = velocities_to_predictions(
                net, inputs, corrected)
            edge_class = "E>0" if any_active else "E=0"
            _metric_append(
                metrics["C_stage_b_reference_off"], net,
                prediction_reference, auxiliary, inputs, mask, edge_class)
            _metric_append(
                metrics["D_stage_b_v2a_corrected"], net,
                prediction_corrected, auxiliary, inputs, mask, edge_class)

            invariants["candidate_ids_unchanged"] &= torch.equal(
                ids, auxiliary["joint_candidate_index"][:, :20])
            invariants["joint_goals_unchanged"] &= torch.equal(
                goals, auxiliary["joint_goal_points_world"][:, :20])
            invariants["coverage_20_of_20"] &= all(
                torch.unique(row).numel() == 20 for row in ids)
            counts["candidate_id_elements"] += ids.numel()
            counts["goal_elements"] += goals.numel()

    if not all(invariants.values()):
        raise RuntimeError(f"paired evaluation invariant failed: {invariants}")
    return {
        "seed": int(seed),
        "counts": counts,
        "invariants": invariants,
        "trajectory_C_vs_D": parity,
        "metrics": {
            path: _metric_finalize(store) for path, store in metrics.items()},
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage-a-checkpoint", required=True)
    parser.add_argument("--stage-b-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--seeds", nargs="+", type=int,
        default=[2035, 2036, 2037, 2038, 2039])
    return parser.parse_args()


def main():
    args = parse_args()
    stage_a = Path(args.stage_a_checkpoint)
    stage_b = Path(args.stage_b_checkpoint)
    if sha256(stage_a) != STAGE_A_SHA256:
        raise RuntimeError("Stage-A checkpoint hash changed")
    if sha256(stage_b) != STAGE_B_V2A_SHA256:
        raise RuntimeError("Stage-B V2-A checkpoint hash changed")
    output = Path(args.output)
    evaluator, epoch = _active_evaluator(
        args.config, str(stage_b), output.parent / "runtime", args.device)
    started = time.perf_counter()
    runs = []
    for seed in args.seeds:
        print(f"PAIRED_C_D seed={seed}", flush=True)
        runs.append(evaluate_seed(evaluator, int(seed)))
    checkpoint_contract = _checkpoint_state_contract(stage_a, stage_b)
    only_corrector_changed = (
        checkpoint_contract["key_sets_equal"] and
        not checkpoint_contract["noncorrector_changed_tensors"])
    if not only_corrector_changed:
        raise RuntimeError("Stage-A/V2-A differ outside the corrector")
    result = {
        "status": "COMPLETE",
        "semantic_boundary": {
            "C_stage_b_reference_off": (
                "V2-A checkpoint with correction arithmetic disabled; "
                "this is the Stage-A base reference, not deployed V2-A"),
            "D_stage_b_v2a_corrected": (
                "same V2-A checkpoint with trained component-relative "
                "correction enabled; this is deployed V2-A"),
        },
        "protocol": {
            "split": "ETH validation", "windows": 139,
            "seeds": [int(seed) for seed in args.seeds],
            "precision": "CUDA/BF16", "paired_rng": True,
            "paired_goals_context_relation": True,
            "all_metrics_lower_is_better": True,
        },
        "provenance": {
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True).strip(),
            "config": str(Path(args.config).resolve()),
            "config_sha256": sha256(Path(args.config)),
            "stage_a_checkpoint": str(stage_a.resolve()),
            "stage_a_checkpoint_sha256": sha256(stage_a),
            "stage_b_checkpoint": str(stage_b.resolve()),
            "stage_b_checkpoint_sha256": sha256(stage_b),
            "stage_b_epoch": int(epoch),
        },
        "checkpoint_state_contract": checkpoint_contract,
        "per_seed": runs,
        "metric_summary": summarize_runs(runs),
        "paired_delta_D_minus_C": summarize_paired_delta(runs),
        "runtime_seconds": float(time.perf_counter() - started),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": result["status"],
        "source_commit": result["provenance"]["source_commit"],
        "metric_summary": result["metric_summary"],
        "output": str(output),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
