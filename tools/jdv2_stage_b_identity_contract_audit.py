#!/usr/bin/env python3
"""Mandatory paired identity audit for the Stage-B BF16 routing fix."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from src.utils import isolated_random_seed
from tools.audit_jdv2_stage_a import _active_evaluator
from tools.jdv2_stage_b_v1_failure_mechanism_audit import (
    checkpoint_contract,
    connected_components,
    make_noise_tape,
    replay_branches,
    rng_restore,
    rng_snapshot,
    trunk_from_tape,
)


def _rng_equal(left, right):
    if left["python"] != right["python"]:
        return False
    if left["numpy"][0] != right["numpy"][0] or \
            not np.array_equal(left["numpy"][1], right["numpy"][1]) or \
            left["numpy"][2:] != right["numpy"][2:]:
        return False
    if not torch.equal(left["torch"], right["torch"]):
        return False
    if left["cuda"] is None or right["cuda"] is None:
        return left["cuda"] is right["cuda"]
    return len(left["cuda"]) == len(right["cuda"]) and all(
        torch.equal(a, b) for a, b in zip(left["cuda"], right["cuda"]))


@torch.no_grad()
def legacy_replay_branches(net, all_context, dependency_state, tape,
                           middle_result=None):
    """Exact pre-fix V1 arithmetic, used only for active-agent regression."""
    if middle_result is None:
        middle_result = trunk_from_tape(net, all_context, tape)
    simple_var = net.args.dataset in ["eth5", "ind"]
    eta = 1 if simple_var else 0
    schedule = np.linspace(net.var_sched.num_steps, 0,
                           net.args.ddim_step + 1)
    outputs = []
    state = dependency_state
    for branch in range(net.args.num_samples):
        x_t = middle_result
        context = all_context[branch]
        for noise_index, step in enumerate(range(
                int(net.args.branch_stage_step + 1),
                int(net.args.ddim_step + 1))):
            cur_t = int(schedule[step - 1])
            prev_t = int(schedule[step])
            ab_cur = net.var_sched.alpha_bars[cur_t]
            ab_prev = net.var_sched.alpha_bars[prev_t] if prev_t >= 0 else 1
            beta = net.var_sched.betas[[cur_t] * context.size(0)]
            epsilon = net.diffnet(x_t, beta=beta, context=context)
            last_map = state["last_position_map"]
            position_map = last_map[:, None] + torch.cumsum(x_t, dim=1)
            position_world = state["scene"].make_world_coord_torch(
                position_map * float(net.args.down_factor))
            anchor = state["last_position_world"][:, None]
            noisy_world_velocity = torch.diff(torch.cat((
                anchor.float(), position_world.float()), dim=1), dim=1) / \
                float(net.args.trajectory_dt)
            delta = net.jdv2_corrector(
                noisy_world_velocity, state["last_position_world"],
                state["edge_index"], state["relation_embedding"][:, branch],
                torch.tensor(cur_t, device=x_t.device),
                edge_weight=state["edge_weight"])
            epsilon = epsilon + delta
            variance = eta * (1 - ab_prev) / (1 - ab_cur) * \
                (1 - ab_cur / ab_prev)
            first = (ab_prev / ab_cur) ** 0.5 * x_t
            coefficient = ((1 - ab_prev - variance) ** 0.5 -
                           (ab_prev * (1 - ab_cur) / ab_cur) ** 0.5)
            noise = tape.branch_noise[branch][noise_index]
            third = ((1 - ab_cur / ab_prev) ** 0.5 * noise
                     if simple_var else variance ** 0.5 * noise)
            x_t = first + coefficient * epsilon + third
        outputs.append(x_t)
    return torch.stack(outputs)


def _max_abs(left, right):
    return float((left.float() - right.float()).abs().max().cpu())


@torch.no_grad()
def run_identity_audit(evaluator, seed=2035):
    net = evaluator.net
    net.eval()
    counters = {
        "windows": 0, "e0_windows": 0, "e_gt0_windows": 0,
        "mixed_windows": 0, "candidate_id_checks": 0,
        "goal_checks": 0, "coverage_checks": 0,
    }
    maxima = {
        "audit_full_vs_production": 0.0,
        "audit_none_vs_production": 0.0,
        "e0_full_vs_stage_a": 0.0,
        "mixed_degree_zero_vs_stage_a": 0.0,
        "degree_positive_fixed_vs_legacy": 0.0,
    }
    rng_pairing = True
    with isolated_random_seed(
            int(seed), use_cuda=evaluator.device.type == "cuda"):
        for window_index, (batch_data, batch_id) in enumerate(
                evaluator.data_loaders["valid"]):
            inputs, _sequence = net.prepare_inputs(batch_data, batch_id)
            net.jdv2_sampler.set_sampling_context(seed, window_index)
            with evaluator._autocast_context():
                contexts, auxiliary = net.encode(inputs, if_test=True)
            state = auxiliary["dependency_state"]
            edge_index = state["edge_index"]
            _components, degree = connected_components(
                contexts[-1].size(0), edge_index)
            active = degree.gt(0)
            counters["windows"] += 1
            counters["e0_windows" if not active.any()
                     else "e_gt0_windows"] += 1
            if active.any() and not active.all():
                counters["mixed_windows"] += 1

            ids = auxiliary["joint_candidate_index"][:, :20]
            goals = auxiliary["joint_goal_points_world"][:, :20]
            assert all(torch.unique(row).numel() == 20 for row in ids)
            counters["coverage_checks"] += ids.shape[0]
            ids_before, goals_before = ids.clone(), goals.clone()

            pre = rng_snapshot(evaluator.device.type == "cuda")
            tape = make_noise_tape(net, contexts[-1])
            with evaluator._autocast_context():
                middle = trunk_from_tape(net, contexts, tape)
                fixed, _ = replay_branches(
                    net, contexts, state, tape, "full_v1",
                    inputs["scene_index"], None, seed, middle_result=middle)
                none, _ = replay_branches(
                    net, contexts, state, tape, "stage_a",
                    inputs["scene_index"], None, seed, middle_result=middle)
                legacy = (legacy_replay_branches(
                    net, contexts, state, tape, middle_result=middle)
                    if active.any() else None)

            rng_restore(pre)
            with evaluator._autocast_context():
                production_fixed = net.ts_sample(contexts, state)
            post_fixed = rng_snapshot(evaluator.device.type == "cuda")
            old_flag = net.args.use_dependency_corrector
            try:
                net.args.use_dependency_corrector = False
                rng_restore(pre)
                with evaluator._autocast_context():
                    production_none = net.ts_sample(contexts, state)
            finally:
                net.args.use_dependency_corrector = old_flag
            post_none = rng_snapshot(evaluator.device.type == "cuda")
            rng_pairing = rng_pairing and _rng_equal(post_fixed, post_none)
            rng_restore(post_fixed)

            maxima["audit_full_vs_production"] = max(
                maxima["audit_full_vs_production"],
                _max_abs(fixed, production_fixed))
            maxima["audit_none_vs_production"] = max(
                maxima["audit_none_vs_production"],
                _max_abs(none, production_none))
            if not active.any():
                maxima["e0_full_vs_stage_a"] = max(
                    maxima["e0_full_vs_stage_a"], _max_abs(fixed, none))
                if not torch.equal(fixed, none):
                    raise AssertionError("E=0 identity contract failed")
            else:
                maxima["degree_positive_fixed_vs_legacy"] = max(
                    maxima["degree_positive_fixed_vs_legacy"],
                    _max_abs(fixed[:, active], legacy[:, active]))
                if not torch.equal(fixed[:, active], legacy[:, active]):
                    raise RuntimeError("ACTIVE_PATH_SEMANTICS_CHANGED")
                if (~active).any():
                    maxima["mixed_degree_zero_vs_stage_a"] = max(
                        maxima["mixed_degree_zero_vs_stage_a"],
                        _max_abs(fixed[:, ~active], none[:, ~active]))
                    if not torch.equal(fixed[:, ~active], none[:, ~active]):
                        raise AssertionError(
                            "mixed degree-zero identity contract failed")
            if not torch.equal(fixed, production_fixed) or \
                    not torch.equal(none, production_none):
                raise AssertionError("audit/production parity failed")
            assert torch.equal(ids, ids_before)
            assert torch.equal(goals, goals_before)
            counters["candidate_id_checks"] += ids.numel()
            counters["goal_checks"] += goals.numel()
    passed = all(value == 0.0 for value in maxima.values()) and rng_pairing
    return {
        "passed": passed, "seed": int(seed), "counters": counters,
        "max_abs_differences": maxima,
        "rng_post_state_equal": rng_pairing,
        "candidate_ids_and_goals_unchanged": True,
        "coverage_20_of_20": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage-a-checkpoint", required=True)
    parser.add_argument("--stage-b-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2035)
    args = parser.parse_args()
    output = Path(args.output)
    stage_a = Path(args.stage_a_checkpoint)
    stage_b = Path(args.stage_b_checkpoint)
    contract = checkpoint_contract(stage_a, stage_b)
    if contract["stage_a_sha256"] != \
            "699336d49aaecbccfaac8b44f00c0df5a73b521548fc29d3da37d071148fd5bb":
        raise RuntimeError("Stage-A checkpoint hash changed")
    if contract["stage_b_sha256"] != \
            "e24a1cbfc5760db57bcb5cacd64c66a4afd1aed0f11bb1efefa8549933d21c1a":
        raise RuntimeError("Stage-B checkpoint hash changed")
    evaluator, epoch = _active_evaluator(
        args.config, str(stage_b), output.parent / "runtime", args.device)
    started = time.perf_counter()
    parity = run_identity_audit(evaluator, args.seed)
    result = {
        "status": ("IDENTITY_CONTRACT_PARITY_PASSED" if parity["passed"]
                   else "IDENTITY_CONTRACT_FIX_FAILED"),
        "protocol": {"split": "ETH validation", "windows": 139,
                     "seed": args.seed, "checkpoint_epoch": int(epoch),
                     "precision": "CUDA/BF16", "paired_noise_tape": True},
        "provenance": {**contract, "source_commit_before_fix":
                       subprocess.check_output(
                           ["git", "rev-parse", "HEAD"], text=True).strip()},
        "parity": parity,
        "runtime_seconds": float(time.perf_counter() - started),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"],
                      "parity": parity}, indent=2), flush=True)
    if not parity["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
