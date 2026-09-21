#!/usr/bin/env python3
"""Inference-only validation of CPSR against categorical and ASSIGN.

The runner stops before structured policies when the protected categorical
baseline does not reproduce. It constructs no optimizer and changes no model
weights.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import ExitStack
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from src.jdv2_audit import AUDIT_METRICS
from src.metrics import compute_metric_mask
from src.utils import isolated_random_seed
from tools.audit_jdv2_stage_a import _active_evaluator
from tools.refinement_coalescence_audit import (
    RefinementAuditController,
    _agent_and_decomposition_records,
    _group_summary,
    _merge_score_statistics,
    _score_statistics,
    _stratified,
    _summarize_decomposition,
    _summarize_metric_runs,
    _summarize_round_records,
    _summarize_score_statistics,
    _summarize_transitions,
    _summarize_worlds,
)
from tools.sampler_coverage_intervention import (
    SEEDS,
    _agent_count_bin,
    _degree_bin,
    _degrees,
    _finite_summary,
    _rank_tensor,
    _world_ground_truth,
    capture_rng_state,
    file_sha256,
    restore_rng_state,
    rng_states_equal,
)


POLICIES = (
    "original_categorical",
    "deterministic_ASSIGN",
    "structured_gumbel_assignment",
)


class SamplerTimer:
    """Synchronizing wall timer around only the sampler module."""

    def __init__(self, sampler, device):
        self.sampler = sampler
        self.device = device
        self.values = []
        self._started = None
        self._pre = None
        self._post = None

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _before(self, _module, _inputs):
        self._sync()
        self._started = time.perf_counter()

    def _after(self, _module, _inputs, _output):
        self._sync()
        self.values.append(time.perf_counter() - self._started)

    def __enter__(self):
        self._pre = self.sampler.register_forward_pre_hook(self._before)
        self._post = self.sampler.register_forward_hook(self._after)
        return self

    def __exit__(self, exc_type, exc, traceback):
        self._pre.remove()
        self._post.remove()


class CPSRController:
    """Run production CPSR while pairing its downstream diffusion RNG."""

    def __init__(self, sampler, reference_states, use_cuda=True):
        self.sampler = sampler
        self.reference_states = reference_states
        self.use_cuda = bool(use_cuda)
        self.original_policy = sampler.refinement_policy
        self.original_callback = sampler.diagnostic_callback
        self.key = None
        self.trace = []
        self.boundary_checks = defaultdict(int)
        self._pre_sampler_state = None
        self._post_hook = None

    def _reference(self):
        reference = self.reference_states.get(self.key)
        if reference is None:
            raise RuntimeError(f"missing categorical RNG reference: {self.key}")
        return reference

    def begin_window(self, seed, window_index, edge_count, device):
        self.key = (int(seed), int(window_index))
        self.edge_count = int(edge_count)
        self.trace = []
        restore_rng_state(self._reference()["pre_initial"])
        self.sampler.set_sampling_context(seed, window_index)

    def _record(self, **record):
        self.trace.append({
            "score": record["score"].detach().float().clone(),
            "mask": record["mask"].detach().bool().clone(),
            "candidate_index": record[
                "candidate_index"].detach().long().clone(),
        })

    def _before(self, _module, _inputs):
        self._pre_sampler_state = capture_rng_state(self.use_cuda)
        if not rng_states_equal(
                self._pre_sampler_state, self._reference()["pre_initial"]):
            raise RuntimeError("CPSR pre-sampler RNG differs from control")

    def _after(self, _module, _inputs, _output):
        current = capture_rng_state(self.use_cuda)
        if not rng_states_equal(current, self._pre_sampler_state):
            raise RuntimeError("CPSR consumed an uncontrolled global RNG")
        self.boundary_checks["sampler_global_rng_unchanged"] += 1
        restore_rng_state(self._reference()["pre_diffusion"])
        if not rng_states_equal(
                capture_rng_state(self.use_cuda),
                self._reference()["pre_diffusion"]):
            raise RuntimeError("failed to pair CPSR pre-diffusion RNG")
        self.boundary_checks["pre_diffusion"] += 1

    def end_window(self):
        expected_rounds = 3 if self.edge_count else 1
        if len(self.trace) != expected_rounds:
            raise RuntimeError(
                f"CPSR trace length {len(self.trace)} != {expected_rounds}")
        if not rng_states_equal(
                capture_rng_state(self.use_cuda),
                self._reference()["window_end"]):
            raise RuntimeError("CPSR window-end RNG differs from control")
        self.boundary_checks["window_end"] += 1
        self.key = None

    def __enter__(self):
        self.sampler.refinement_policy = "structured_gumbel_assignment"
        self.sampler.diagnostic_callback = self._record
        self._pre_hook = self.sampler.register_forward_pre_hook(self._before)
        self._post_hook = self.sampler.register_forward_hook(self._after)
        return self

    def __exit__(self, exc_type, exc, traceback):
        self._pre_hook.remove()
        self._post_hook.remove()
        self.sampler.refinement_policy = self.original_policy
        self.sampler.diagnostic_callback = self.original_callback
        self.sampler._sampling_context = None


def _metadata(num_agents, edge_index, policy, seed, window_index):
    degree = _degrees(num_agents, edge_index)
    edge_class = "E>0" if edge_index.shape[1] else "E=0"
    return [{
        "variant": policy,
        "seed": int(seed),
        "window": int(window_index),
        "agent": int(agent),
        "edge_class": edge_class,
        "agent_count_bin": _agent_count_bin(num_agents),
        "degree_bin": _degree_bin(int(degree[agent])),
    } for agent in range(num_agents)]


def _churn_records(trace, metadata, metric_mask):
    if len(trace) < 3:
        return []
    ids = [item["candidate_index"].long() for item in trace]
    records = []
    for agent in torch.nonzero(metric_mask, as_tuple=False).flatten().tolist():
        r0, r1, r2 = (value[agent] for value in ids)
        records.append({
            **metadata[agent],
            "round0_to_round1_slot_churn": float(
                r0.ne(r1).float().mean().cpu()),
            "round1_to_round2_slot_churn": float(
                r1.ne(r2).float().mean().cpu()),
            "round0_to_round2_slot_churn": float(
                r0.ne(r2).float().mean().cpu()),
            "two_cycle_rate": float(
                (r2.eq(r0) & r1.ne(r0)).float().mean().cpu()),
        })
    return records


def _excluded_records(net, inputs, auxiliary, trace, metadata, metric_mask):
    bank_score = auxiliary["candidate_log_prior"].float()
    unary_score = auxiliary["unary_score"].float()
    bank_rank = _rank_tensor(bank_score)
    unary_rank = _rank_tensor(unary_score)
    bank_probability = F.softmax(bank_score, dim=-1)
    unary_probability = F.softmax(unary_score, dim=-1)
    candidate = auxiliary["goal_candidates_world"].float()
    gt_goal = _world_ground_truth(net, inputs)[-1]
    gt_oracle_id = torch.linalg.vector_norm(
        candidate - gt_goal[:, None], dim=-1).argmin(-1)
    candidate_rows = []
    agent_rows = []
    for round_index, item in enumerate(trace):
        score = item["score"].float()
        mask = item["mask"].bool()
        ids = item["candidate_index"].long()
        if not torch.equal(mask, mask[:, :1].expand_as(mask)):
            raise RuntimeError("audit observed a slot-varying CPSR mask")
        probability = F.softmax(
            score.masked_fill(~mask, float("-inf")) /
            float(net.jdv2_sampler.temperature), dim=-1)
        for agent in torch.nonzero(
                metric_mask, as_tuple=False).flatten().tolist():
            selected = torch.zeros(
                score.shape[-1], dtype=torch.bool, device=score.device)
            selected[torch.unique(ids[agent])] = True
            excluded = mask[agent, 0] & ~selected
            excluded_ids = torch.nonzero(
                excluded, as_tuple=False).flatten()
            agent_rows.append({
                **metadata[agent],
                "round": int(round_index),
                "excluded_count": int(excluded_ids.numel()),
                "gt_oracle_excluded": bool(excluded[gt_oracle_id[agent]]),
            })
            for candidate_id in excluded_ids.tolist():
                candidate_rows.append({
                    **metadata[agent],
                    "round": int(round_index),
                    "candidate": int(candidate_id),
                    "frozen_prior_rank": float(
                        bank_rank[agent, candidate_id].cpu()),
                    "unary_rank": float(
                        unary_rank[agent, candidate_id].cpu()),
                    "frozen_prior_probability": float(
                        bank_probability[agent, candidate_id].cpu()),
                    "unary_probability": float(
                        unary_probability[agent, candidate_id].cpu()),
                    "conditional_score": float(
                        score[agent, :, candidate_id].mean().cpu()),
                    "conditional_probability": float(
                        probability[agent, :, candidate_id].mean().cpu()),
                    "maximum_conditional_probability": float(
                        probability[agent, :, candidate_id].max().cpu()),
                })
    return candidate_rows, agent_rows


def _summarize_churn(rows):
    scalar = (
        "round0_to_round1_slot_churn",
        "round1_to_round2_slot_churn",
        "round0_to_round2_slot_churn",
        "two_cycle_rate",
    )
    return _stratified(rows, lambda group: _group_summary(group, scalar))


def _summarize_excluded(candidate_rows, agent_rows):
    result = {}
    rounds = sorted({row["round"] for row in agent_rows})
    scalar = (
        "frozen_prior_rank", "unary_rank", "frozen_prior_probability",
        "unary_probability", "conditional_score",
        "conditional_probability", "maximum_conditional_probability",
    )
    for round_index in rounds:
        local_candidates = [
            row for row in candidate_rows if row["round"] == round_index]
        local_agents = [
            row for row in agent_rows if row["round"] == round_index]
        result[f"round{round_index}"] = {
            "excluded_candidates": _stratified(
                local_candidates,
                lambda group: _group_summary(group, scalar)),
            "agent_exclusion": _stratified(
                local_agents,
                lambda group: _group_summary(
                    group, ("excluded_count",), ("gt_oracle_excluded",))),
        }
    return result


def _runtime_summary(per_seed, sampler_times, forward_times, device):
    result = {
        "per_seed": per_seed,
        "sampler_seconds": _finite_summary(sampler_times),
        "full_forward_seconds": _finite_summary(forward_times),
        "sampler_total_seconds": float(sum(sampler_times)),
        "full_forward_total_seconds": float(sum(forward_times)),
    }
    if device.type == "cuda":
        result.update({
            "peak_cuda_allocated_bytes": int(
                torch.cuda.max_memory_allocated(device)),
            "peak_cuda_reserved_bytes": int(
                torch.cuda.max_memory_reserved(device)),
        })
    return result


@torch.no_grad()
def evaluate_policy(evaluator, policy, reference_states):
    net = evaluator.net
    net.eval()
    metric_runs = []
    agent_rows = []
    decomposition_rows = []
    transition_rows = []
    world_rows = []
    churn_rows = []
    excluded_candidates = []
    excluded_agents = []
    score_values = defaultdict(lambda: defaultdict(list))
    window_counts = defaultdict(int)
    sampler_times = []
    forward_times = []
    runtime_per_seed = []
    if evaluator.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(evaluator.device)

    if policy == "structured_gumbel_assignment":
        controller = CPSRController(
            net.jdv2_sampler, reference_states,
            use_cuda=evaluator.device.type == "cuda")
    else:
        variant = "iid_control" if policy == "original_categorical" \
            else "ASSIGN"
        controller = RefinementAuditController(
            net.jdv2_sampler, variant, reference_states,
            use_cuda=evaluator.device.type == "cuda")

    for seed in SEEDS:
        print(f"POLICY {policy} seed={seed}", flush=True)
        metric_values = defaultdict(lambda: defaultdict(list))
        seed_sampler_start = len(sampler_times)
        seed_forward_start = len(forward_times)
        with isolated_random_seed(
                seed, use_cuda=evaluator.device.type == "cuda"):
            with ExitStack() as stack:
                timer = stack.enter_context(
                    SamplerTimer(net.jdv2_sampler, evaluator.device))
                stack.enter_context(controller)
                for window_index, (batch_data, batch_id) in enumerate(
                        evaluator.data_loaders["valid"]):
                    inputs, seq_list = net.prepare_inputs(batch_data, batch_id)
                    metric_mask = compute_metric_mask(seq_list)
                    edge_index = inputs["jdv2_cache"]["edge_index"].long()
                    edge_count = int(edge_index.shape[1])
                    controller.begin_window(
                        seed, window_index, edge_count, evaluator.device)
                    if evaluator.device.type == "cuda":
                        torch.cuda.synchronize(evaluator.device)
                    started = time.perf_counter()
                    with evaluator._autocast_context():
                        prediction, auxiliary = net.forward(
                            inputs, if_test=True)
                    if evaluator.device.type == "cuda":
                        torch.cuda.synchronize(evaluator.device)
                    forward_times.append(time.perf_counter() - started)
                    controller.end_window()
                    edge_class = "E>0" if edge_count else "E=0"
                    window_counts[(seed, edge_class)] += 1
                    for metric in AUDIT_METRICS:
                        values = net.compute_model_metrics(
                            metric_name=metric, predictions=prediction,
                            metric_mask=metric_mask,
                            all_aux_outputs=auxiliary, inputs=inputs,
                            obs_length=net.args.obs_length)
                        metric_values["overall"][metric].extend(values)
                        metric_values[edge_class][metric].extend(values)
                    records = _agent_and_decomposition_records(
                        net, inputs, prediction, auxiliary, metric_mask,
                        controller.trace, policy, seed, window_index)
                    agent_rows.extend(records[0])
                    decomposition_rows.extend(records[1])
                    transition_rows.extend(records[2])
                    world_rows.extend(records[3])
                    metadata = _metadata(
                        auxiliary["unary_score"].shape[0],
                        auxiliary["edge_index"].long(), policy, seed,
                        window_index)
                    churn_rows.extend(_churn_records(
                        controller.trace, metadata, metric_mask))
                    excluded = _excluded_records(
                        net, inputs, auxiliary, controller.trace, metadata,
                        metric_mask)
                    excluded_candidates.extend(excluded[0])
                    excluded_agents.extend(excluded[1])
                    _merge_score_statistics(
                        score_values,
                        _score_statistics(
                            controller.trace,
                            net.jdv2_sampler.temperature))
                    del inputs, prediction, auxiliary, seq_list
        sampler_times.extend(timer.values)
        local_sampler = sampler_times[seed_sampler_start:]
        local_forward = forward_times[seed_forward_start:]
        runtime_per_seed.append({
            "seed": int(seed),
            "sampler_total_seconds": float(sum(local_sampler)),
            "full_forward_total_seconds": float(sum(local_forward)),
        })
        metric_runs.append({
            "seed": int(seed),
            **{
                stratum: {
                    metric: float(np.mean(items))
                    for metric, items in values.items()}
                for stratum, values in metric_values.items()
            },
        })

    return {
        "metric_runs": metric_runs,
        "metric_summary": _summarize_metric_runs(metric_runs),
        "round_candidate_summary": _summarize_round_records(agent_rows),
        "error_decomposition": _summarize_decomposition(decomposition_rows),
        "conditional_score_summary": _summarize_score_statistics(
            score_values),
        "transition_summary": _summarize_transitions(transition_rows),
        "world_diversity_summary": _summarize_worlds(world_rows),
        "slot_churn_and_two_cycle": _summarize_churn(churn_rows),
        "excluded_candidate_summary": _summarize_excluded(
            excluded_candidates, excluded_agents),
        "runtime": _runtime_summary(
            runtime_per_seed, sampler_times, forward_times,
            evaluator.device),
        "window_counts": {
            str(seed): {
                name: window_counts[(seed, name)]
                for name in ("E=0", "E>0")}
            for seed in SEEDS
        },
        "paired_rng_checks": dict(controller.boundary_checks),
    }


def reproduction_gate(actual, expected, tolerance=1e-10):
    checks = {}
    expected_runs = {
        int(row["seed"]): row for row in expected["metric_runs"]}
    for row in actual["metric_runs"]:
        target = expected_runs[int(row["seed"])]
        for metric in AUDIT_METRICS:
            value = row["overall"][metric]
            reference = target["overall"][metric]
            checks[f"seed_{row['seed']}_{metric}"] = {
                "actual": value,
                "expected": reference,
                "absolute_error": abs(value - reference),
            }
    actual_r0 = actual["round_candidate_summary"]["round0"][
        "overall"]["scalars"]
    expected_r0 = expected["round_candidate_summary"]["round0"][
        "overall"]["scalars"]
    actual_final = actual["round_candidate_summary"]["final"][
        "overall"]["scalars"]
    expected_final = expected["round_candidate_summary"]["final"][
        "overall"]["scalars"]
    actual_decomp = actual["error_decomposition"]["overall"]["scalars"]
    expected_decomp = expected["error_decomposition"]["overall"]["scalars"]
    scalar_checks = {
        "round0_unique": (
            actual_r0["unique_count"]["mean"],
            expected_r0["unique_count"]["mean"]),
        "final_unique": (
            actual_final["unique_count"]["mean"],
            expected_final["unique_count"]["mean"]),
        "round0_oracle": (
            actual_r0["goal_oracle_error"]["mean"],
            expected_r0["goal_oracle_error"]["mean"]),
        "final_goal_oracle": (
            actual_decomp["final_goal_oracle"]["mean"],
            expected_decomp["final_goal_oracle"]["mean"]),
        "trajectory_minFDE": (
            actual_decomp["trajectory_minFDE"]["mean"],
            expected_decomp["trajectory_minFDE"]["mean"]),
    }
    for name, (value, reference) in scalar_checks.items():
        checks[name] = {
            "actual": value,
            "expected": reference,
            "absolute_error": abs(value - reference),
        }
    return {
        "passed": all(row["absolute_error"] <= tolerance
                      for row in checks.values()),
        "tolerance": tolerance,
        "maximum_absolute_error": max(
            row["absolute_error"] for row in checks.values()),
        "checks": checks,
    }


def adoption_gate(cpsr, categorical, gdts):
    cpsr_mean = {
        name: cpsr["metric_summary"]["overall"][name]["mean"]
        for name in AUDIT_METRICS}
    categorical_mean = {
        name: categorical["metric_summary"]["overall"][name]["mean"]
        for name in AUDIT_METRICS}
    gdts_mean = {
        name: gdts["summary"][name]["mean"] for name in AUDIT_METRICS}
    relative_gdts = {
        name: (cpsr_mean[name] / gdts_mean[name] - 1.0)
        for name in AUDIT_METRICS}
    relative_categorical = {
        name: (cpsr_mean[name] / categorical_mean[name] - 1.0)
        if categorical_mean[name] else 0.0
        for name in AUDIT_METRICS}
    joint_gate_metrics = (
        "JADE", "JFDE", "Joint_Goal_Endpoint_Error",
        "Relative_Motion_Error",
    )
    checks = {
        "minFDE_vs_GDTS_le_plus_2_percent": (
            relative_gdts["minFDE@K"] <= 0.02),
        "joint_metrics_better_than_GDTS": all(
            cpsr_mean[name] < gdts_mean[name]
            for name in joint_gate_metrics),
        "joint_metrics_no_gt_2_percent_degradation_vs_categorical": all(
            relative_categorical[name] <= 0.02
            for name in joint_gate_metrics),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "cpsr_mean": cpsr_mean,
        "relative_change_vs_GDTS": relative_gdts,
        "relative_change_vs_original_categorical": relative_categorical,
        "compatibility_reported_not_used_as_hard_gate": (
            relative_categorical["Joint_Goal_Compatibility"]),
    }


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--refinement-reference", required=True)
    parser.add_argument("--gdts-reference", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    cli = parser.parse_args()

    output = Path(cli.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    evaluator, epoch = _active_evaluator(
        cli.config, cli.checkpoint, output.parent / "runtime", cli.device)
    if epoch != 13 or not evaluator.net.strict_no_z:
        raise RuntimeError("CPSR validation requires strict no-z epoch 13")
    checkpoint = torch.load(cli.checkpoint, map_location="cpu")
    architecture = checkpoint.get("architecture_config", {})
    if architecture.get("architecture_variant") != "strict_no_z":
        raise RuntimeError("checkpoint architecture is not strict_no_z")
    reference_path = Path(cli.refinement_reference).resolve()
    refinement_reference = json.loads(reference_path.read_text())
    if refinement_reference.get("status") != "AUDIT_COMPLETE":
        raise RuntimeError("refinement reference is incomplete")
    gdts_path = Path(cli.gdts_reference).resolve()
    gdts_reference = json.loads(gdts_path.read_text())[
        "interventions"]["gdts"]

    result = {
        "status": "RUNNING_BASELINE_REPRODUCTION",
        "protocol": {
            "split": "valid",
            "windows": 139,
            "seeds": list(SEEDS),
            "candidates": 21,
            "samples": 20,
            "relation_modes": 4,
            "energy_rank": 8,
            "refinement_rounds": 2,
            "temperature": float(evaluator.net.jdv2_sampler.temperature),
            "policies": list(POLICIES),
            "cpsr_mask_contract": (
                "agent-level slot-invariant mask with valid_K>=P"),
            "cpsr_interpretation": (
                "sample-set-level structured allocation of existing "
                "neighbor-conditioned conditional compatibility scores"),
            "rng_pairing": (
                "categorical captures complete pre-initial/pre-round/"
                "pre-diffusion/window-end RNG states; ASSIGN restores the "
                "historical boundaries; CPSR uses explicit independent "
                "init/round streams, consumes no global RNG, and restores "
                "the identical categorical pre-diffusion state."),
        },
        "provenance": {
            "checkpoint": str(Path(cli.checkpoint).resolve()),
            "checkpoint_epoch": int(epoch),
            "checkpoint_sha256": file_sha256(cli.checkpoint),
            "architecture_config": architecture,
            "cache_manifest_hash": checkpoint.get("cache_manifest_hash"),
            "source_checkpoint_hash": checkpoint.get(
                "source_checkpoint_hash"),
            "refinement_reference": str(reference_path),
            "refinement_reference_sha256": file_sha256(reference_path),
            "gdts_reference": str(gdts_path),
            "gdts_reference_sha256": file_sha256(gdts_path),
            "inference_sampler_config": {
                "refinement_policy": "structured_gumbel_assignment",
                "round0": "weighted_gumbel_top_p_without_replacement",
                "round1_round2": (
                    "unit_gumbel_perturbed_injective_assignment"),
                "num_samples": 20,
                "num_candidates": 21,
                "num_refinement_steps": 2,
                "temperature": float(
                    evaluator.net.jdv2_sampler.temperature),
                "solver": "scipy.optimize.linear_sum_assignment",
            },
        },
        "references": {
            "GDTS": gdts_reference,
        },
        "policies": {},
    }
    _write_json(output, result)

    reference_states = {}
    categorical = evaluate_policy(
        evaluator, "original_categorical", reference_states)
    result["policies"]["original_categorical"] = categorical
    expected_categorical = refinement_reference["policies"]["iid_control"]
    gate = reproduction_gate(categorical, expected_categorical)
    result["baseline_reproduction_gate"] = gate
    if not gate["passed"]:
        result["status"] = "STOPPED_BASELINE_REPRODUCTION_FAILED"
        _write_json(output, result)
        raise RuntimeError("categorical baseline reproduction failed")
    result["status"] = "BASELINE_REPRODUCTION_PASSED"
    _write_json(output, result)

    assign = evaluate_policy(
        evaluator, "deterministic_ASSIGN", reference_states)
    result["policies"]["deterministic_ASSIGN"] = assign
    result["assign_reproduction_gate"] = reproduction_gate(
        assign, refinement_reference["policies"]["ASSIGN"])
    if not result["assign_reproduction_gate"]["passed"]:
        result["status"] = "STOPPED_ASSIGN_REPRODUCTION_FAILED"
        _write_json(output, result)
        raise RuntimeError("deterministic ASSIGN reproduction failed")
    result["status"] = "ASSIGN_REPRODUCTION_PASSED"
    _write_json(output, result)

    cpsr = evaluate_policy(
        evaluator, "structured_gumbel_assignment", reference_states)
    result["policies"]["structured_gumbel_assignment"] = cpsr

    expected_windows = len(SEEDS) * 139
    cpsr_rng = cpsr["paired_rng_checks"]
    for name in (
            "sampler_global_rng_unchanged", "pre_diffusion", "window_end"):
        if cpsr_rng.get(name) != expected_windows:
            raise RuntimeError(f"incomplete CPSR RNG check: {name}")
    assign_rng = assign["paired_rng_checks"]
    if (assign_rng.get("pre_diffusion") != expected_windows or
            assign_rng.get("window_end") != expected_windows):
        raise RuntimeError("incomplete ASSIGN downstream RNG pairing")

    result["runtime_relative_to_categorical"] = {
        policy: {
            "sampler_ratio": (
                result["policies"][policy]["runtime"][
                    "sampler_total_seconds"] /
                categorical["runtime"]["sampler_total_seconds"]),
            "full_forward_ratio": (
                result["policies"][policy]["runtime"][
                    "full_forward_total_seconds"] /
                categorical["runtime"]["full_forward_total_seconds"]),
        }
        for policy in ("deterministic_ASSIGN",
                       "structured_gumbel_assignment")
    }
    result["adoption_gate"] = adoption_gate(
        cpsr, categorical, gdts_reference)
    result["status"] = "VALIDATION_COMPLETE"
    _write_json(output, result)
    print(f"WROTE {output}", flush=True)


if __name__ == "__main__":
    main()
