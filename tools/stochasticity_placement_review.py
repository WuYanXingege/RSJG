#!/usr/bin/env python3
"""Paired audit of where stochasticity belongs in JDV2 refinement.

Phase A only.  The production sampler is not modified.  Each policy uses the
same production-derived round-0 Gumbel-Top-P realization and the same
pre-diffusion RNG state.  The sole intervention is whether fresh unit Gumbels
are added before the two injective refinement assignments.
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path
import time

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from src.jdv2_audit import AUDIT_METRICS
from src.metrics import compute_metric_mask
import src.models.joint_dependency_v2.joint_sampler as sampler_module
from src.utils import isolated_random_seed
from tools.audit_jdv2_stage_a import _active_evaluator
from tools.cpsr_inference_validation import (
    SamplerTimer,
    _churn_records,
    _metadata,
    _runtime_summary,
    reproduction_gate,
)
from tools.refinement_coalescence_audit import (
    _agent_and_decomposition_records,
    _merge_score_statistics,
    _score_statistics,
    _summarize_decomposition,
    _summarize_metric_runs,
    _summarize_round_records,
    _summarize_score_statistics,
    _summarize_transitions,
    _summarize_worlds,
)
from tools.sampler_coverage_intervention import (
    SEEDS,
    _degrees,
    _finite_summary,
    capture_rng_state,
    file_sha256,
    restore_rng_state,
    rng_states_equal,
)


POLICIES = (
    "shared_round0_fresh_gumbel",
    "shared_round0_deterministic",
)
PRINCIPAL_JOINT_METRICS = (
    "JADE",
    "JFDE",
    "Joint_Goal_Endpoint_Error",
    "Relative_Motion_Error",
)


class PairedStructuredController:
    """Use shared production round 0 and switch refinement noise only."""

    def __init__(self, sampler, policy, reference_states, shared_round0,
                 use_cuda=True):
        if policy not in POLICIES:
            raise ValueError(f"unknown paired policy: {policy}")
        self.sampler = sampler
        self.policy = policy
        self.reference_states = reference_states
        self.shared_round0 = shared_round0
        self.use_cuda = bool(use_cuda)
        self.original_policy = sampler.refinement_policy
        self.original_callback = sampler.diagnostic_callback
        self.original_initial = sampler_module.weighted_gumbel_top_p
        self.original_refinement = sampler_module.structured_gumbel_assignment
        self.key = None
        self.edge_count = None
        self.trace = []
        self.boundary_checks = defaultdict(int)
        self._pre_sampler_state = None

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
            raise RuntimeError("paired sampler did not start at pre-initial RNG")

    def _after(self, _module, _inputs, _output):
        current = capture_rng_state(self.use_cuda)
        if not rng_states_equal(current, self._pre_sampler_state):
            raise RuntimeError("paired structured sampler consumed global RNG")
        self.boundary_checks["sampler_global_rng_unchanged"] += 1
        restore_rng_state(self._reference()["pre_diffusion"])
        if not rng_states_equal(
                capture_rng_state(self.use_cuda),
                self._reference()["pre_diffusion"]):
            raise RuntimeError("failed to restore paired pre-diffusion RNG")
        self.boundary_checks["pre_diffusion"] += 1

    def end_window(self):
        expected_rounds = 3 if self.edge_count else 1
        if len(self.trace) != expected_rounds:
            raise RuntimeError(
                f"paired trace length {len(self.trace)} != {expected_rounds}")
        if not rng_states_equal(
                capture_rng_state(self.use_cuda),
                self._reference()["window_end"]):
            raise RuntimeError("paired window-end RNG differs from control")
        self.boundary_checks["window_end"] += 1
        self.key = None

    def __enter__(self):
        if sampler_module.weighted_gumbel_top_p is not self.original_initial:
            raise RuntimeError("weighted Gumbel initializer already patched")
        if (sampler_module.structured_gumbel_assignment is not
                self.original_refinement):
            raise RuntimeError("structured assignment already patched")

        def shared_initial(score, mask, temperature, generator,
                           gumbel_noise=None):
            selected = self.original_initial(
                score, mask, temperature, generator,
                gumbel_noise=gumbel_noise)
            cpu_selected = selected.detach().cpu().clone()
            if self.policy == "shared_round0_fresh_gumbel":
                if self.key in self.shared_round0:
                    raise RuntimeError("duplicate shared round-0 key")
                self.shared_round0[self.key] = cpu_selected
            else:
                expected = self.shared_round0.get(self.key)
                if expected is None:
                    raise RuntimeError("deterministic branch lacks shared round 0")
                if not torch.equal(cpu_selected, expected):
                    raise RuntimeError("production initial generator did not pair")
                self.boundary_checks["shared_round0_exact"] += 1
            return selected

        def refinement(score, mask, temperature, generator,
                       gumbel_noise=None):
            if self.policy == "shared_round0_fresh_gumbel":
                return self.original_refinement(
                    score, mask, temperature, generator,
                    gumbel_noise=gumbel_noise)
            before = generator.get_state().clone() if generator is not None \
                else None
            selected = sampler_module.maximum_weight_injective_assignment(
                score.float() / float(temperature), mask)
            if before is not None and not torch.equal(
                    before, generator.get_state()):
                raise RuntimeError("deterministic refinement consumed Gumbel RNG")
            self.boundary_checks["round_generator_unchanged"] += 1
            return selected

        sampler_module.weighted_gumbel_top_p = shared_initial
        sampler_module.structured_gumbel_assignment = refinement
        self.sampler.refinement_policy = "structured_gumbel_assignment"
        self.sampler.diagnostic_callback = self._record
        self._pre_hook = self.sampler.register_forward_pre_hook(self._before)
        self._post_hook = self.sampler.register_forward_hook(self._after)
        return self

    def __exit__(self, exc_type, exc, traceback):
        self._pre_hook.remove()
        self._post_hook.remove()
        sampler_module.weighted_gumbel_top_p = self.original_initial
        sampler_module.structured_gumbel_assignment = self.original_refinement
        self.sampler.refinement_policy = self.original_policy
        self.sampler.diagnostic_callback = self.original_callback
        self.sampler._sampling_context = None


def _support_record(trace, mask, seed, window_index):
    result = []
    for round_name, item in (
            ("round0", trace[0]), ("final", trace[-1])):
        ids = item["candidate_index"].detach().long().cpu()
        valid = mask.detach().bool().cpu()
        for agent in range(ids.shape[0]):
            selected = tuple(sorted(set(ids[agent].tolist())))
            valid_ids = tuple(torch.nonzero(
                valid[agent], as_tuple=False).flatten().tolist())
            excluded = tuple(sorted(set(valid_ids) - set(selected)))
            result.append({
                "round": round_name,
                "seed": int(seed),
                "window": int(window_index),
                "agent": int(agent),
                "support": selected,
                "excluded": excluded,
            })
    return result


def _world_support_record(trace, seed, window_index):
    result = []
    for round_name, item in (
            ("round0", trace[0]), ("final", trace[-1])):
        worlds = item["candidate_index"].detach().long().cpu().transpose(
            0, 1).contiguous()
        result.append({
            "round": round_name,
            "seed": int(seed),
            "window": int(window_index),
            "worlds": worlds,
        })
    return result


def _entropy_from_items(items):
    count = Counter(items)
    probability = np.asarray(list(count.values()), dtype=np.float64)
    probability /= probability.sum()
    return float(-(probability * np.log(probability)).sum())


def summarize_cross_seed_support(support_rows, world_rows):
    agent_groups = defaultdict(list)
    for row in support_rows:
        if row["round"] == "final":
            agent_groups[(row["window"], row["agent"])].append(row)
    distinct_excluded = []
    excluded_entropy = []
    all_same = []
    support_jaccard = []
    for rows in agent_groups.values():
        rows = sorted(rows, key=lambda row: row["seed"])
        excluded = [row["excluded"] for row in rows]
        distinct_excluded.append(len(set(excluded)))
        excluded_entropy.append(_entropy_from_items(excluded))
        all_same.append(len(set(excluded)) == 1)
        for left, right in itertools.combinations(rows, 2):
            a, b = set(left["support"]), set(right["support"])
            support_jaccard.append(len(a & b) / max(len(a | b), 1))

    world_groups = defaultdict(dict)
    for row in world_rows:
        world_groups[(row["round"], row["window"])][row["seed"]] = \
            row["worlds"]
    world_summary = {}
    for round_name in ("round0", "final"):
        matched_hamming = []
        exact_fraction = []
        whole_equal = []
        for (local_round, _window), by_seed in world_groups.items():
            if local_round != round_name:
                continue
            for left_seed, right_seed in itertools.combinations(SEEDS, 2):
                left, right = by_seed[left_seed], by_seed[right_seed]
                cost = left[:, None].ne(right[None]).float().mean(-1).numpy()
                rows, columns = linear_sum_assignment(cost)
                values = cost[rows, columns]
                matched_hamming.append(float(values.mean()))
                exact_fraction.append(float(np.mean(values == 0.0)))
                whole_equal.append(bool(np.all(values == 0.0)))
        world_summary[round_name] = {
            "matched_mean_normalized_hamming": _finite_summary(
                matched_hamming),
            "matched_exact_world_fraction": _finite_summary(exact_fraction),
            "exact_whole_set_equality_rate": float(np.mean(whole_equal)),
            "seed_pair_window_count": len(whole_equal),
        }
    return {
        "agent_count": len(agent_groups),
        "distinct_excluded_candidate_ids": _finite_summary(
            distinct_excluded),
        "excluded_id_entropy": _finite_summary(excluded_entropy),
        "all_five_seeds_same_excluded_id_rate": float(np.mean(all_same)),
        "final_support_jaccard": _finite_summary(support_jaccard),
        "joint_world_set_matching": world_summary,
    }


def _assignment_objective(score, ids):
    return score.gather(-1, ids.unsqueeze(-1)).squeeze(-1).sum(-1)


def synthetic_tie_audit():
    cases = {
        "unique_optimum": torch.tensor([[[9., 1., 0., -1.],
                                           [0., 8., 1., -1.],
                                           [0., 1., 7., -1.]]]),
        "identical_rows": torch.tensor([[[3., 2., 1., 0.],
                                          [3., 2., 1., 0.],
                                          [3., 2., 1., 0.]]]),
        "repeated_ties": torch.tensor([[[2., 2., 0., 0.],
                                         [2., 2., 0., 0.],
                                         [0., 0., 1., 1.]]]),
    }
    permutation = torch.tensor([2, 0, 1])
    result = {}
    for name, score in cases.items():
        mask = torch.ones_like(score, dtype=torch.bool)
        original = sampler_module.maximum_weight_injective_assignment(
            score, mask)
        assigned = sampler_module.maximum_weight_injective_assignment(
            score[:, permutation], mask[:, permutation])
        unpermuted = torch.empty_like(assigned)
        unpermuted[:, permutation] = assigned
        objective = _assignment_objective(score, original)
        permuted_objective = _assignment_objective(score, unpermuted)
        result[name] = {
            "original": original[0].tolist(),
            "unpermuted_after_common_slot_permutation": (
                unpermuted[0].tolist()),
            "objective_equal": bool(torch.allclose(
                objective, permuted_objective, atol=0, rtol=0)),
            "pathwise_equal": bool(torch.equal(original, unpermuted)),
            "unordered_support_equal": bool(torch.equal(
                original.sort(-1).values, unpermuted.sort(-1).values)),
        }
    return result


def real_permutation_records(trace, edge_index, seed, window_index):
    if len(trace) < 3:
        return []
    num_agents, num_slots = trace[1]["candidate_index"].shape
    degree = _degrees(num_agents, edge_index).cpu()
    generator = torch.Generator().manual_seed(
        int(seed) * 1_000_003 + int(window_index) * 97_409 + 0x71E)
    permutations = (
        torch.arange(num_slots).roll(1),
        torch.arange(num_slots - 1, -1, -1),
        torch.randperm(num_slots, generator=generator),
    )
    records = []
    for round_index in (1, 2):
        item = trace[round_index]
        score = item["score"].cpu()
        mask = item["mask"].cpu()
        original = item["candidate_index"].cpu()
        expected = sampler_module.maximum_weight_injective_assignment(
            score, mask)
        if not torch.equal(expected, original):
            raise RuntimeError("deterministic trace differs from exact solver")
        original_objective = _assignment_objective(score, original)
        original_worlds = original.transpose(0, 1).contiguous()
        for permutation_index, permutation in enumerate(permutations):
            assigned = sampler_module.maximum_weight_injective_assignment(
                score[:, permutation], mask[:, permutation])
            unpermuted = torch.empty_like(assigned)
            unpermuted[:, permutation] = assigned
            objective = _assignment_objective(score, unpermuted)
            objective_delta = (objective - original_objective).abs()
            support_equal = original.sort(-1).values.eq(
                unpermuted.sort(-1).values).all(-1)
            path_equal = original.eq(unpermuted).all(-1)
            alternative_worlds = unpermuted.transpose(0, 1).contiguous()
            cost = original_worlds[:, None].ne(
                alternative_worlds[None]).float().mean(-1).numpy()
            rows, columns = linear_sum_assignment(cost)
            matched = cost[rows, columns]
            records.append({
                "seed": int(seed),
                "window": int(window_index),
                "round": int(round_index),
                "permutation": int(permutation_index),
                "maximum_objective_delta": float(objective_delta.max()),
                "all_agent_support_equal": bool(support_equal.all()),
                "all_agent_pathwise_equal": bool(path_equal.all()),
                "degree0_agent_count": int(degree.eq(0).sum()),
                "degree0_pathwise_equal_rate": float(
                    path_equal[degree.eq(0)].float().mean())
                    if bool(degree.eq(0).any()) else None,
                "matched_mean_normalized_hamming": float(matched.mean()),
                "matched_exact_world_fraction": float(
                    np.mean(matched == 0.0)),
                "exact_whole_world_set_equal": bool(
                    np.all(matched == 0.0)),
            })
    return records


def summarize_real_permutation(rows):
    scalar_names = (
        "maximum_objective_delta",
        "degree0_pathwise_equal_rate",
        "matched_mean_normalized_hamming",
        "matched_exact_world_fraction",
    )
    result = {}
    for round_index in (1, 2):
        local = [row for row in rows if row["round"] == round_index]
        result[f"round{round_index}"] = {
            "count": len(local),
            "scalars": {
                name: _finite_summary([
                    row[name] for row in local if row[name] is not None])
                for name in scalar_names
            },
            "all_agent_support_equal_rate": float(np.mean([
                row["all_agent_support_equal"] for row in local])),
            "all_agent_pathwise_equal_rate": float(np.mean([
                row["all_agent_pathwise_equal"] for row in local])),
            "exact_whole_world_set_equal_rate": float(np.mean([
                row["exact_whole_world_set_equal"] for row in local])),
        }
    return result


@torch.no_grad()
def evaluate_paired_policy(evaluator, policy, reference_states,
                           shared_round0):
    net = evaluator.net
    net.eval()
    controller = PairedStructuredController(
        net.jdv2_sampler, policy, reference_states, shared_round0,
        use_cuda=evaluator.device.type == "cuda")
    metric_runs = []
    agent_rows, decomposition_rows = [], []
    transition_rows, world_diversity_rows = [], []
    churn_rows = []
    score_values = defaultdict(lambda: defaultdict(list))
    support_rows, cross_seed_world_rows = [], []
    permutation_rows = []
    window_counts = defaultdict(int)
    sampler_times, forward_times, runtime_per_seed = [], [], []
    if evaluator.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(evaluator.device)

    for seed in SEEDS:
        print(f"PAIRED {policy} seed={seed}", flush=True)
        metric_values = defaultdict(lambda: defaultdict(list))
        sampler_start, forward_start = len(sampler_times), len(forward_times)
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
                        prediction, auxiliary = net.forward(inputs, if_test=True)
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
                    world_diversity_rows.extend(records[3])
                    metadata = _metadata(
                        auxiliary["unary_score"].shape[0],
                        auxiliary["edge_index"].long(), policy, seed,
                        window_index)
                    churn_rows.extend(_churn_records(
                        controller.trace, metadata, metric_mask))
                    _merge_score_statistics(
                        score_values, _score_statistics(
                            controller.trace,
                            net.jdv2_sampler.temperature))
                    support_rows.extend(_support_record(
                        controller.trace,
                        controller.trace[0]["mask"][:, 0],
                        seed, window_index))
                    cross_seed_world_rows.extend(_world_support_record(
                        controller.trace, seed, window_index))
                    if policy == "shared_round0_deterministic":
                        permutation_rows.extend(real_permutation_records(
                            controller.trace, edge_index, seed, window_index))
                    del inputs, prediction, auxiliary, seq_list
        sampler_times.extend(timer.values)
        runtime_per_seed.append({
            "seed": int(seed),
            "sampler_total_seconds": float(sum(
                sampler_times[sampler_start:])),
            "full_forward_total_seconds": float(sum(
                forward_times[forward_start:])),
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
        "conditional_score_summary": _summarize_score_statistics(score_values),
        "transition_summary": _summarize_transitions(transition_rows),
        "world_diversity_summary": _summarize_worlds(
            world_diversity_rows),
        "slot_churn_and_two_cycle": (
            __import__("tools.cpsr_inference_validation", fromlist=[
                "_summarize_churn"])._summarize_churn(churn_rows)),
        "cross_seed_goal_support": summarize_cross_seed_support(
            support_rows, cross_seed_world_rows),
        "real_score_permutation_audit": (
            summarize_real_permutation(permutation_rows)
            if permutation_rows else None),
        "runtime": _runtime_summary(
            runtime_per_seed, sampler_times, forward_times, evaluator.device),
        "window_counts": {
            str(seed): {
                name: window_counts[(seed, name)]
                for name in ("E=0", "E>0")}
            for seed in SEEDS
        },
        "paired_rng_checks": dict(controller.boundary_checks),
    }


def phase_a_gate(fresh, deterministic, gdts, tie_summary):
    fresh_e = fresh["metric_summary"]["E>0"]
    deterministic_e = deterministic["metric_summary"]["E>0"]
    deterministic_overall = deterministic["metric_summary"]["overall"]
    deterministic_final = deterministic["round_candidate_summary"][
        "final"]["overall"]["scalars"]["unique_count"]["mean"]
    relative_minfde = (
        deterministic_overall["minFDE@K"]["mean"] /
        gdts["summary"]["minFDE@K"]["mean"] - 1.0)
    support = deterministic["cross_seed_goal_support"]
    final_world = support["joint_world_set_matching"]["final"]
    variability = (
        support["distinct_excluded_candidate_ids"]["mean"] > 1.0 and
        final_world["matched_mean_normalized_hamming"]["mean"] > 0.0 and
        final_world["exact_whole_set_equality_rate"] < 1.0)
    joint_better = all(
        deterministic_e[name]["mean"] < fresh_e[name]["mean"]
        for name in PRINCIPAL_JOINT_METRICS)
    real_rounds = [tie_summary[f"round{index}"] for index in (1, 2)]
    objective_tolerance = 1e-5
    solver_valid = all(
        row["scalars"]["maximum_objective_delta"]["mean"] <=
        objective_tolerance and
        row["all_agent_support_equal_rate"] == 1.0
        for row in real_rounds)
    # World-set changes from a common row permutation are reported, not hidden.
    # Treat a change in more than 5% of real matrices with nonzero matched
    # Hamming as a material arbitrary-slot pathology for productionization.
    tie_safe = all(
        row["exact_whole_world_set_equal_rate"] >= 0.95 or
        row["scalars"]["matched_mean_normalized_hamming"]["mean"] == 0.0
        for row in real_rounds)
    checks = {
        "deterministic_refinement_better_Egt0_joint_coherence": joint_better,
        "deterministic_final_coverage_is_20": abs(
            deterministic_final - 20.0) <= 1e-12,
        "deterministic_minFDE_vs_GDTS_le_plus_2_percent": (
            relative_minfde <= 0.02),
        "seed_dependent_final_goal_support_confirmed": variability,
        "solver_objective_and_support_valid": solver_valid,
        "no_material_arbitrary_slot_world_set_pathology": tie_safe,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "relative_minFDE_vs_GDTS": relative_minfde,
        "solver_objective_tolerance": objective_tolerance,
        "tie_materiality_rule": (
            "fail when either refinement round changes >5% of unordered "
            "real joint-world sets under common slot permutations and mean "
            "matched normalized Hamming is nonzero"),
    }


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cpsr-reference", required=True)
    parser.add_argument("--gdts-reference", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    cli = parser.parse_args()

    output = Path(cli.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    evaluator, epoch = _active_evaluator(
        cli.config, cli.checkpoint, output.parent / "runtime", cli.device)
    if epoch != 13 or not evaluator.net.strict_no_z:
        raise RuntimeError("review requires strict no-z epoch-13 checkpoint")
    checkpoint = torch.load(cli.checkpoint, map_location="cpu")
    architecture = checkpoint.get("architecture_config", {})
    if architecture.get("architecture_variant") != "strict_no_z":
        raise RuntimeError("checkpoint architecture is not strict_no_z")
    cpsr_path = Path(cli.cpsr_reference).resolve()
    cpsr_reference = json.loads(cpsr_path.read_text())
    if cpsr_reference.get("status") != "VALIDATION_COMPLETE":
        raise RuntimeError("CPSR reference is incomplete")
    gdts_path = Path(cli.gdts_reference).resolve()
    gdts = json.loads(gdts_path.read_text())["interventions"]["gdts"]

    result = {
        "status": "RUNNING_CATEGORICAL_RNG_CONTROL",
        "protocol": {
            "phase": "A_paired_refinement_stochasticity_audit",
            "split": "valid",
            "windows": 139,
            "seeds": list(SEEDS),
            "candidates": 21,
            "samples": 20,
            "refinement_rounds": 2,
            "shared_round0": (
                "production weighted_gumbel_top_p with identical explicit "
                "initial generator and exact candidate-ID assertion"),
            "policy_G": "fresh unit Gumbel before each injective assignment",
            "policy_D": "deterministic injective assignment without round noise",
            "diffusion_rng": "identical categorical pre-diffusion state",
        },
        "source_fact_review": {
            "production_CPSR": {
                "round0": "weighted_gumbel_top_p",
                "round1_round2": (
                    "conditional_score/T + fresh unit Gumbel + exact "
                    "maximum-weight injective assignment"),
            },
            "historical_ASSIGN": {
                "round0": "weighted_without_replacement using +17 seed tag",
                "round1_round2": "deterministic coverage_assignment",
                "full_policy_is_stochastic": True,
                "stochasticity_sources": [
                    "round0 hypothesis-set sampling",
                    "frozen GDTS trajectory diffusion",
                ],
            },
            "round0_rng_was_strictly_paired_in_old_comparison": False,
            "production_initial_seed_tag": "0x5A17",
            "historical_audit_initial_seed_tag": "+17",
        },
        "provenance": {
            "checkpoint": str(Path(cli.checkpoint).resolve()),
            "checkpoint_epoch": int(epoch),
            "checkpoint_sha256": file_sha256(cli.checkpoint),
            "architecture_config": architecture,
            "cache_manifest_hash": checkpoint.get("cache_manifest_hash"),
            "source_checkpoint_hash": checkpoint.get("source_checkpoint_hash"),
            "cpsr_reference": str(cpsr_path),
            "cpsr_reference_sha256": file_sha256(cpsr_path),
            "gdts_reference": str(gdts_path),
            "gdts_reference_sha256": file_sha256(gdts_path),
        },
        "synthetic_tie_audit": synthetic_tie_audit(),
        "policies": {},
    }
    _write_json(output, result)

    # Reuse the accepted categorical run as the complete downstream RNG
    # boundary reference.  It must reproduce before either paired branch runs.
    from tools.cpsr_inference_validation import evaluate_policy
    reference_states = {}
    categorical = evaluate_policy(
        evaluator, "original_categorical", reference_states)
    result["categorical_control"] = categorical
    result["categorical_reproduction_gate"] = reproduction_gate(
        categorical,
        cpsr_reference["policies"]["original_categorical"])
    if not result["categorical_reproduction_gate"]["passed"]:
        result["status"] = "STOPPED_CATEGORICAL_REPRODUCTION_FAILED"
        _write_json(output, result)
        raise RuntimeError("categorical reproduction failed")

    shared_round0 = {}
    fresh = evaluate_paired_policy(
        evaluator, "shared_round0_fresh_gumbel", reference_states,
        shared_round0)
    result["policies"]["shared_round0_fresh_gumbel"] = fresh
    result["fresh_gumbel_CPSR_reproduction_gate"] = reproduction_gate(
        fresh, cpsr_reference["policies"]["structured_gumbel_assignment"])
    if not result["fresh_gumbel_CPSR_reproduction_gate"]["passed"]:
        result["status"] = "STOPPED_CPSR_REPRODUCTION_FAILED"
        _write_json(output, result)
        raise RuntimeError("fresh-Gumbel CPSR reproduction failed")
    result["status"] = "RUNNING_SHARED_ROUND0_DETERMINISTIC"
    _write_json(output, result)

    deterministic = evaluate_paired_policy(
        evaluator, "shared_round0_deterministic", reference_states,
        shared_round0)
    result["policies"]["shared_round0_deterministic"] = deterministic
    expected_windows = len(SEEDS) * 139
    expected_round_calls = len(SEEDS) * 92 * 2
    deterministic_rng = deterministic["paired_rng_checks"]
    if deterministic_rng.get("shared_round0_exact") != expected_windows:
        raise RuntimeError("incomplete shared round-0 assertions")
    if deterministic_rng.get(
            "round_generator_unchanged") != expected_round_calls:
        raise RuntimeError("deterministic rounds did not verify generators")
    for policy in POLICIES:
        checks = result["policies"][policy]["paired_rng_checks"]
        for boundary in (
                "sampler_global_rng_unchanged", "pre_diffusion", "window_end"):
            if checks.get(boundary) != expected_windows:
                raise RuntimeError(
                    f"incomplete {policy} RNG check: {boundary}")

    tie_summary = deterministic["real_score_permutation_audit"]
    result["phase_a_gate"] = phase_a_gate(
        fresh, deterministic, gdts, tie_summary)
    result["status"] = (
        "PHASE_A_GATE_PASSED" if result["phase_a_gate"]["passed"]
        else "PHASE_A_GATE_FAILED_STOPPED_BEFORE_PHASE_B")
    _write_json(output, result)
    print(f"WROTE {output}", flush=True)


if __name__ == "__main__":
    main()
