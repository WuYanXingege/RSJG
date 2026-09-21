#!/usr/bin/env python3
"""Localize deterministic-refinement tie semantics without changing src/.

Policy D is the accepted shared-Round-0 deterministic audit reference. Policy
D0 changes only the audit-time selection primitive: degree-zero agents retain
their previous IDs, while degree-positive agents use the unchanged exact
injective assignment. Both policies keep production Round-0 stochasticity and
paired diffusion RNG.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
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
    _summarize_churn,
    reproduction_gate,
)
from tools.refinement_coalescence_audit import (
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
    _world_ground_truth,
    _world_predictions,
    capture_rng_state,
    file_sha256,
    restore_rng_state,
    rng_states_equal,
)
from tools.stochasticity_placement_review import (
    _support_record,
    _world_support_record,
    summarize_cross_seed_support,
)


POLICIES = ("D_deterministic", "D0_degree0_identity")
ROW_NEAR_TOLERANCE = 1e-6


def graph_components(num_agents, edge_index):
    """Return deterministic undirected connected components, incl. isolates."""
    adjacency = [set() for _ in range(num_agents)]
    for source, destination in edge_index.detach().long().cpu().transpose(
            0, 1).tolist():
        adjacency[source].add(destination)
        adjacency[destination].add(source)
    seen = set()
    result = []
    for root in range(num_agents):
        if root in seen:
            continue
        stack = [root]
        seen.add(root)
        component = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in sorted(adjacency[current], reverse=True):
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        result.append(tuple(sorted(component)))
    return tuple(result)


def graph_class(components):
    if len(components) == 1:
        return "single_connected_component"
    if any(len(component) == 1 for component in components):
        return "multiple_components_with_singleton"
    return "multiple_components_all_size_ge2"


def component_size_bin(size):
    if size == 1:
        return "size=1"
    if size == 2:
        return "size=2"
    if size <= 4:
        return "size=3-4"
    return "size>=5"


class TieLocalizationController:
    """Audit-only D/D0 selector with shared initial and RNG assertions."""

    def __init__(self, sampler, policy, reference_states, baseline_ids,
                 use_cuda=True):
        if policy not in POLICIES:
            raise ValueError(f"unknown tie-localization policy: {policy}")
        self.sampler = sampler
        self.policy = policy
        self.reference_states = reference_states
        self.baseline_ids = baseline_ids
        self.use_cuda = bool(use_cuda)
        self.original_policy = sampler.refinement_policy
        self.original_callback = sampler.diagnostic_callback
        self.original_refinement = sampler_module.structured_gumbel_assignment
        self.key = None
        self.edge_index = None
        self.degree = None
        self.trace = []
        self.boundary_checks = defaultdict(int)
        self._pre_sampler_state = None

    def _reference(self):
        reference = self.reference_states.get(self.key)
        if reference is None:
            raise RuntimeError(f"missing categorical RNG reference: {self.key}")
        return reference

    def begin_window(self, seed, window_index, edge_index, device):
        self.key = (int(seed), int(window_index))
        self.edge_index = edge_index.detach().long().clone()
        num_agents = int(self.edge_index.max().item() + 1) \
            if self.edge_index.numel() else None
        # The exact N is learned from the round-0 callback for E=0. For E>0,
        # the edge maximum can omit trailing isolated agents, so degree is also
        # finalized in _record from the score N axis.
        self.degree = None if num_agents is None else _degrees(
            num_agents, self.edge_index)
        self.trace = []
        restore_rng_state(self._reference()["pre_initial"])
        self.sampler.set_sampling_context(seed, window_index)

    def _record(self, **record):
        item = {
            "score": record["score"].detach().float().clone(),
            "mask": record["mask"].detach().bool().clone(),
            "candidate_index": record[
                "candidate_index"].detach().long().clone(),
        }
        round_index = int(record["round_index"])
        if self.degree is None or self.degree.numel() != item["score"].shape[0]:
            self.degree = _degrees(item["score"].shape[0], self.edge_index)
        baseline_key = (*self.key, round_index)
        local_cpu = item["candidate_index"].detach().cpu()
        if self.policy == "D_deterministic":
            self.baseline_ids[baseline_key] = {
                "candidate_index": local_cpu.clone(),
                "score": item["score"].detach().cpu().clone(),
                "mask": item["mask"].detach().cpu().clone(),
            }
        else:
            baseline = self.baseline_ids.get(baseline_key)
            if baseline is None:
                raise RuntimeError(f"missing Policy-D IDs for {baseline_key}")
            expected = baseline["candidate_index"]
            if round_index == 0:
                if not torch.equal(local_cpu, expected):
                    raise RuntimeError("D0 Round-0 differs from Policy D")
                self.boundary_checks["shared_round0_exact"] += 1
            else:
                positive = self.degree.detach().cpu().gt(0)
                if not torch.equal(local_cpu[positive], expected[positive]):
                    score_cpu = item["score"].detach().cpu()
                    score_delta = (score_cpu[positive] -
                                   baseline["score"][positive]).abs()
                    differing = torch.nonzero(
                        local_cpu.ne(expected).any(-1) & positive,
                        as_tuple=False).flatten().tolist()
                    raise RuntimeError(
                        "D0 changed degree-positive deterministic assignment: "
                        f"key={self.key}, round={round_index}, "
                        f"agents={differing}, "
                        f"score_exact={torch.equal(score_cpu[positive], baseline['score'][positive])}, "
                        f"max_score_delta={float(score_delta.max())}")
                self.boundary_checks[
                    "degree_positive_assignment_exact"] += int(positive.sum())
                zero = ~positive
                previous = self.trace[-1]["candidate_index"].detach().cpu()
                if not torch.equal(local_cpu[zero], previous[zero]):
                    raise RuntimeError("D0 degree-zero identity was not exact")
                self.boundary_checks[
                    "degree_zero_identity_exact"] += int(zero.sum())
        self.trace.append(item)

    def _before(self, _module, _inputs):
        self._pre_sampler_state = capture_rng_state(self.use_cuda)
        if not rng_states_equal(
                self._pre_sampler_state, self._reference()["pre_initial"]):
            raise RuntimeError("tie audit did not start at pre-initial RNG")

    def _after(self, _module, _inputs, _output):
        current = capture_rng_state(self.use_cuda)
        if not rng_states_equal(current, self._pre_sampler_state):
            raise RuntimeError("tie audit sampler consumed global RNG")
        self.boundary_checks["sampler_global_rng_unchanged"] += 1
        restore_rng_state(self._reference()["pre_diffusion"])
        if not rng_states_equal(
                capture_rng_state(self.use_cuda),
                self._reference()["pre_diffusion"]):
            raise RuntimeError("tie audit failed pre-diffusion RNG pairing")
        self.boundary_checks["pre_diffusion"] += 1

    def end_window(self):
        expected_rounds = 3 if self.edge_index.shape[1] else 1
        if len(self.trace) != expected_rounds:
            raise RuntimeError(
                f"tie trace length {len(self.trace)} != {expected_rounds}")
        if not rng_states_equal(
                capture_rng_state(self.use_cuda),
                self._reference()["window_end"]):
            raise RuntimeError("tie audit window-end RNG differs from control")
        self.boundary_checks["window_end"] += 1
        self.key = None

    def __enter__(self):
        if (sampler_module.structured_gumbel_assignment is not
                self.original_refinement):
            raise RuntimeError("structured assignment already patched")

        def refinement(score, mask, temperature, generator,
                       gumbel_noise=None):
            before = generator.get_state().clone() if generator is not None \
                else None
            scaled = score.float() / float(temperature)
            if self.policy == "D_deterministic":
                selected = sampler_module.maximum_weight_injective_assignment(
                    scaled, mask)
            else:
                if not self.trace:
                    raise RuntimeError("D0 refinement lacks previous IDs")
                previous = self.trace[-1]["candidate_index"].long()
                positive = self.degree.gt(0)
                selected = previous.clone()
                if bool(positive.any()):
                    selected[positive] = \
                        sampler_module.maximum_weight_injective_assignment(
                            scaled[positive], mask[positive])
            if before is not None and not torch.equal(
                    before, generator.get_state()):
                raise RuntimeError("deterministic tie audit consumed round RNG")
            self.boundary_checks["round_generator_unchanged"] += 1
            return selected

        sampler_module.structured_gumbel_assignment = refinement
        self.sampler.refinement_policy = "structured_gumbel_assignment"
        self.sampler.diagnostic_callback = self._record
        self._pre_hook = self.sampler.register_forward_pre_hook(self._before)
        self._post_hook = self.sampler.register_forward_hook(self._after)
        return self

    def __exit__(self, exc_type, exc, traceback):
        self._pre_hook.remove()
        self._post_hook.remove()
        sampler_module.structured_gumbel_assignment = self.original_refinement
        self.sampler.refinement_policy = self.original_policy
        self.sampler.diagnostic_callback = self.original_callback
        self.sampler._sampling_context = None


def deterministic_selection(policy, score, mask, previous, degree):
    """Pure audit helper used by both intervention and permutation tests."""
    if policy == "D_deterministic":
        return sampler_module.maximum_weight_injective_assignment(score, mask)
    if policy != "D0_degree0_identity":
        raise ValueError(f"unknown policy: {policy}")
    result = previous.clone()
    positive = degree.gt(0)
    if bool(positive.any()):
        result[positive] = sampler_module.maximum_weight_injective_assignment(
            score[positive], mask[positive])
    return result


def _assignment_objective(score, ids):
    return score.gather(-1, ids.unsqueeze(-1)).squeeze(-1).sum(-1)


def _match_world_sets(left, right):
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("world sets must share shape [P,N]")
    cost = left[:, None].ne(right[None]).float().mean(-1).numpy()
    rows, columns = linear_sum_assignment(cost)
    values = cost[rows, columns]
    return {
        "matched_mean_normalized_hamming": float(values.mean()),
        "matched_exact_world_fraction": float(np.mean(values == 0.0)),
        "exact_world_set_equal": bool(np.all(values == 0.0)),
    }


def permutation_localization_records(policy, trace, edge_index, seed,
                                     window_index):
    """Audit common row permutations with score/mask/previous IDs coupled."""
    if len(trace) < 3:
        return [], [], []
    num_agents, num_slots = trace[1]["candidate_index"].shape
    degree = _degrees(num_agents, edge_index).cpu()
    components = graph_components(num_agents, edge_index)
    local_graph_class = graph_class(components)
    generator = torch.Generator().manual_seed(
        int(seed) * 1_000_003 + int(window_index) * 97_409 + 0x71E)
    permutations = (
        torch.arange(num_slots).roll(1),
        torch.arange(num_slots - 1, -1, -1),
        torch.randperm(num_slots, generator=generator),
    )
    agent_rows, scene_rows, component_rows = [], [], []
    component_id = {}
    for index, component in enumerate(components):
        for agent in component:
            component_id[agent] = index

    for round_index in (1, 2):
        item = trace[round_index]
        previous = trace[round_index - 1]["candidate_index"].cpu()
        score = item["score"].cpu()
        mask = item["mask"].cpu()
        original = item["candidate_index"].cpu()
        expected = deterministic_selection(
            policy, score, mask, previous, degree)
        if not torch.equal(original, expected):
            raise RuntimeError("recorded selection differs from audit policy")
        original_objective = _assignment_objective(score, original)
        original_worlds = original.transpose(0, 1).contiguous()

        for permutation_index, permutation in enumerate(permutations):
            assigned = deterministic_selection(
                policy, score[:, permutation], mask[:, permutation],
                previous[:, permutation], degree)
            unpermuted = torch.empty_like(assigned)
            unpermuted[:, permutation] = assigned
            objective_delta = (_assignment_objective(
                score, unpermuted) - original_objective).abs()
            path_equal = original.eq(unpermuted).all(-1)
            support_equal = original.sort(-1).values.eq(
                unpermuted.sort(-1).values).all(-1)
            slot_change = original.ne(unpermuted).float().mean(-1)
            for agent in range(num_agents):
                size = len(components[component_id[agent]])
                agent_rows.append({
                    "policy": policy,
                    "seed": int(seed),
                    "window": int(window_index),
                    "round": int(round_index),
                    "permutation": int(permutation_index),
                    "agent": int(agent),
                    "degree_bin": _degree_bin(int(degree[agent])),
                    "component_size_bin": component_size_bin(size),
                    "component_size": int(size),
                    "objective_delta": float(objective_delta[agent]),
                    "support_invariant": bool(support_equal[agent]),
                    "pathwise_invariant": bool(path_equal[agent]),
                    "slot_change_rate": float(slot_change[agent]),
                    "tie_failure": bool(not path_equal[agent]),
                })

            scene_match = _match_world_sets(
                original_worlds, unpermuted.transpose(0, 1).contiguous())
            scene_rows.append({
                "policy": policy,
                "seed": int(seed),
                "window": int(window_index),
                "round": int(round_index),
                "permutation": int(permutation_index),
                "graph_class": local_graph_class,
                "num_components": len(components),
                "num_singleton_components": sum(
                    len(value) == 1 for value in components),
                "num_nonsingleton_components": sum(
                    len(value) >= 2 for value in components),
                "largest_component_size": max(map(len, components)),
                "fully_connected": len(components) == 1,
                **scene_match,
            })

            for index, component in enumerate(components):
                agents = torch.tensor(component, dtype=torch.long)
                left = original[agents].transpose(0, 1).contiguous()
                right = unpermuted[agents].transpose(0, 1).contiguous()
                component_rows.append({
                    "policy": policy,
                    "seed": int(seed),
                    "window": int(window_index),
                    "round": int(round_index),
                    "permutation": int(permutation_index),
                    "component": int(index),
                    "graph_class": local_graph_class,
                    "component_size": len(component),
                    "component_size_bin": component_size_bin(len(component)),
                    "is_singleton": len(component) == 1,
                    **_match_world_sets(left, right),
                })
    return agent_rows, scene_rows, component_rows


def row_degeneracy_records(policy, trace, edge_index, seed, window_index):
    if len(trace) < 3:
        return []
    num_agents = trace[1]["score"].shape[0]
    degree = _degrees(num_agents, edge_index).cpu()
    components = graph_components(num_agents, edge_index)
    size_by_agent = {}
    for component in components:
        for agent in component:
            size_by_agent[agent] = len(component)
    rows = []
    for round_index in (1, 2):
        score = trace[round_index]["score"].detach().float().cpu()
        difference = (score - score[:, :1]).abs().amax(dim=(1, 2))
        exact = score.eq(score[:, :1].expand_as(score)).all(dim=(1, 2))
        for agent in range(num_agents):
            size = size_by_agent[agent]
            rows.append({
                "policy": policy,
                "seed": int(seed),
                "window": int(window_index),
                "round": int(round_index),
                "agent": int(agent),
                "degree_bin": _degree_bin(int(degree[agent])),
                "component_size_bin": component_size_bin(size),
                "component_size": int(size),
                "max_row_difference": float(difference[agent]),
                "exact_slot_invariant_rows": bool(exact[agent]),
                "near_slot_invariant_rows": bool(
                    difference[agent] <= ROW_NEAR_TOLERANCE),
            })
    return rows


def marginal_stratum_records(net, inputs, output, auxiliary, metric_mask,
                             policy, seed, window_index):
    prediction = _world_predictions(net, output, inputs)
    ground_truth = _world_ground_truth(net, inputs)
    error = torch.linalg.vector_norm(
        prediction - ground_truth[None], dim=-1)  # [P,T,N]
    min_ade = error.mean(1).min(0).values
    min_fde = error[:, -1].min(0).values
    edge_index = auxiliary["edge_index"].long()
    degree = _degrees(error.shape[-1], edge_index)
    rows = []
    for agent in torch.nonzero(metric_mask, as_tuple=False).flatten().tolist():
        rows.append({
            "policy": policy,
            "seed": int(seed),
            "window": int(window_index),
            "degree_bin": _degree_bin(int(degree[agent])),
            "agent_count_bin": _agent_count_bin(error.shape[-1]),
            "minADE": float(min_ade[agent].cpu()),
            "minFDE": float(min_fde[agent].cpu()),
        })
    return rows


def _summarize_boolean_rows(rows, group_field, scalar_names, bool_names):
    result = {}
    for round_index in (1, 2):
        local = [row for row in rows if row["round"] == round_index]
        groups = defaultdict(list)
        for row in local:
            groups[row[group_field]].append(row)
        result[f"round{round_index}"] = {
            key: {
                "count": len(group),
                "scalars": {
                    name: _finite_summary([row[name] for row in group])
                    for name in scalar_names
                },
                **{
                    name: float(np.mean([row[name] for row in group]))
                    for name in bool_names
                },
            }
            for key, group in sorted(groups.items())
        }
    return result


def summarize_permutation(agent_rows, scene_rows, component_rows):
    agent = _summarize_boolean_rows(
        agent_rows, "degree_bin", ("objective_delta", "slot_change_rate"),
        ("support_invariant", "pathwise_invariant", "tie_failure"))
    by_component_size = _summarize_boolean_rows(
        agent_rows, "component_size_bin",
        ("objective_delta", "slot_change_rate"),
        ("support_invariant", "pathwise_invariant", "tie_failure"))
    scene = _summarize_boolean_rows(
        scene_rows, "graph_class",
        ("matched_mean_normalized_hamming", "matched_exact_world_fraction",
         "num_components", "num_singleton_components",
         "num_nonsingleton_components", "largest_component_size"),
        ("exact_world_set_equal", "fully_connected"))
    component = _summarize_boolean_rows(
        component_rows, "component_size_bin",
        ("matched_mean_normalized_hamming", "matched_exact_world_fraction",
         "component_size"), ("exact_world_set_equal", "is_singleton"))
    overall_scene = {}
    for round_index in (1, 2):
        local = [row for row in scene_rows if row["round"] == round_index]
        overall_scene[f"round{round_index}"] = {
            "count": len(local),
            "matched_mean_normalized_hamming": _finite_summary([
                row["matched_mean_normalized_hamming"] for row in local]),
            "matched_exact_world_fraction": _finite_summary([
                row["matched_exact_world_fraction"] for row in local]),
            "exact_world_set_equal_rate": float(np.mean([
                row["exact_world_set_equal"] for row in local])),
        }
    return {
        "overall_scene": overall_scene,
        "agents_by_degree": agent,
        "agents_by_component_size": by_component_size,
        "scenes_by_graph_class": scene,
        "components_by_size": component,
    }


def summarize_row_degeneracy(rows):
    result = {}
    for grouping in ("degree_bin", "component_size_bin"):
        result[grouping] = _summarize_boolean_rows(
            rows, grouping, ("max_row_difference",),
            ("exact_slot_invariant_rows", "near_slot_invariant_rows"))
    return result


def summarize_marginal_strata(rows):
    result = {}
    for field in ("degree_bin", "agent_count_bin"):
        groups = defaultdict(list)
        for row in rows:
            groups[row[field]].append(row)
        result[field] = {
            key: {
                "count": len(group),
                "minADE": _finite_summary([row["minADE"] for row in group]),
                "minFDE": _finite_summary([row["minFDE"] for row in group]),
            }
            for key, group in sorted(groups.items())
        }
    return result


@torch.no_grad()
def evaluate_policy(evaluator, policy, reference_states, baseline_ids):
    net = evaluator.net
    net.eval()
    controller = TieLocalizationController(
        net.jdv2_sampler, policy, reference_states, baseline_ids,
        use_cuda=evaluator.device.type == "cuda")
    metric_runs = []
    agent_rows, decomposition_rows = [], []
    transition_rows, world_rows, churn_rows = [], [], []
    score_values = defaultdict(lambda: defaultdict(list))
    support_rows, cross_seed_world_rows = [], []
    permutation_agents, permutation_scenes = [], []
    permutation_components, degeneracy_rows, marginal_rows = [], [], []
    window_counts = defaultdict(int)
    sampler_times, forward_times, runtime_per_seed = [], [], []
    if evaluator.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(evaluator.device)

    for seed in SEEDS:
        print(f"TIE {policy} seed={seed}", flush=True)
        metric_values = defaultdict(lambda: defaultdict(list))
        agent_count_values = defaultdict(lambda: defaultdict(list))
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
                        seed, window_index, edge_index, evaluator.device)
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
                    count_bin = _agent_count_bin(
                        auxiliary["unary_score"].shape[0])
                    window_counts[(seed, edge_class)] += 1
                    for metric in AUDIT_METRICS:
                        values = net.compute_model_metrics(
                            metric_name=metric, predictions=prediction,
                            metric_mask=metric_mask,
                            all_aux_outputs=auxiliary, inputs=inputs,
                            obs_length=net.args.obs_length)
                        metric_values["overall"][metric].extend(values)
                        metric_values[edge_class][metric].extend(values)
                        agent_count_values[count_bin][metric].extend(values)
                    records = _agent_and_decomposition_records(
                        net, inputs, prediction, auxiliary, metric_mask,
                        controller.trace, policy, seed, window_index)
                    agent_rows.extend(records[0])
                    decomposition_rows.extend(records[1])
                    transition_rows.extend(records[2])
                    world_rows.extend(records[3])
                    metadata = _metadata(
                        auxiliary["unary_score"].shape[0], edge_index,
                        policy, seed, window_index)
                    churn_rows.extend(_churn_records(
                        controller.trace, metadata, metric_mask))
                    _merge_score_statistics(
                        score_values, _score_statistics(
                            controller.trace, net.jdv2_sampler.temperature))
                    support_rows.extend(_support_record(
                        controller.trace,
                        controller.trace[0]["mask"][:, 0],
                        seed, window_index))
                    cross_seed_world_rows.extend(_world_support_record(
                        controller.trace, seed, window_index))
                    perm = permutation_localization_records(
                        policy, controller.trace, edge_index, seed,
                        window_index)
                    permutation_agents.extend(perm[0])
                    permutation_scenes.extend(perm[1])
                    permutation_components.extend(perm[2])
                    degeneracy_rows.extend(row_degeneracy_records(
                        policy, controller.trace, edge_index, seed,
                        window_index))
                    marginal_rows.extend(marginal_stratum_records(
                        net, inputs, prediction, auxiliary, metric_mask,
                        policy, seed, window_index))
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
            "agent_count_strata": {
                name: {
                    metric: float(np.mean(items))
                    for metric, items in values.items()}
                for name, values in agent_count_values.items()
            },
        })

    return {
        "metric_runs": metric_runs,
        "metric_summary": _summarize_metric_runs(metric_runs),
        "agent_count_metric_summary": {
            name: {
                metric: _finite_summary([
                    row["agent_count_strata"][name][metric]
                    for row in metric_runs
                    if name in row["agent_count_strata"]])
                for metric in AUDIT_METRICS
            }
            for name in sorted({
                name for row in metric_runs
                for name in row["agent_count_strata"]})
        },
        "marginal_degree_and_agent_count": summarize_marginal_strata(
            marginal_rows),
        "round_candidate_summary": _summarize_round_records(agent_rows),
        "error_decomposition": _summarize_decomposition(decomposition_rows),
        "conditional_score_summary": _summarize_score_statistics(score_values),
        "transition_summary": _summarize_transitions(transition_rows),
        "world_diversity_summary": _summarize_worlds(world_rows),
        "slot_churn_and_two_cycle": _summarize_churn(churn_rows),
        "cross_seed_goal_support": summarize_cross_seed_support(
            support_rows, cross_seed_world_rows),
        "permutation_localization": summarize_permutation(
            permutation_agents, permutation_scenes,
            permutation_components),
        "row_degeneracy": summarize_row_degeneracy(degeneracy_rows),
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


def permutation_reproduction_gate(actual, expected, tolerance=1e-10):
    checks = {}
    old = expected["real_score_permutation_audit"]
    new = actual["permutation_localization"]["overall_scene"]
    for round_name in ("round1", "round2"):
        mappings = {
            "exact_world_set_equal_rate": (
                new[round_name]["exact_world_set_equal_rate"],
                old[round_name]["exact_whole_world_set_equal_rate"]),
            "matched_mean_normalized_hamming": (
                new[round_name]["matched_mean_normalized_hamming"]["mean"],
                old[round_name]["scalars"][
                    "matched_mean_normalized_hamming"]["mean"]),
            "matched_exact_world_fraction": (
                new[round_name]["matched_exact_world_fraction"]["mean"],
                old[round_name]["scalars"][
                    "matched_exact_world_fraction"]["mean"]),
        }
        for name, (value, reference) in mappings.items():
            checks[f"{round_name}_{name}"] = {
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


def decide_localization(d_policy, d0_policy):
    d_scene = d_policy["permutation_localization"]["overall_scene"]
    d0_scene = d0_policy["permutation_localization"]["overall_scene"]
    d0_components = d0_policy["permutation_localization"][
        "components_by_size"]
    invariance = [
        d0_scene[f"round{index}"]["exact_world_set_equal_rate"]
        for index in (1, 2)]
    hamming = [
        d0_scene[f"round{index}"][
            "matched_mean_normalized_hamming"]["mean"]
        for index in (1, 2)]
    d_invariance = [
        d_scene[f"round{index}"]["exact_world_set_equal_rate"]
        for index in (1, 2)]
    metric_relative = {
        name: (
            d0_policy["metric_summary"]["overall"][name]["mean"] /
            d_policy["metric_summary"]["overall"][name]["mean"] - 1.0)
        for name in AUDIT_METRICS
    }
    coverage = d0_policy["round_candidate_summary"][
        "final"]["overall"]["scalars"]["unique_count"]["mean"]
    metric_preserved = all(
        metric_relative[name] <= 0.02 for name in (
            "JADE", "JFDE", "Joint_Goal_Endpoint_Error",
            "Relative_Motion_Error"))
    case_a = (
        all(value >= 0.95 for value in invariance) and
        all(value <= 1e-6 for value in hamming) and
        metric_preserved and abs(coverage - 20.0) <= 1e-12)
    improvement = float(np.mean(invariance) - np.mean(d_invariance))
    if case_a:
        case = "CASE_A_ISOLATED_AGENT_PRIMARY_BLOCKER"
        recommendation = (
            "design and formally validate interaction-aware deterministic "
            "structured refinement with an explicit degree-zero identity bypass")
    elif improvement > 0.05:
        non_singleton_rates = []
        for round_index in (1, 2):
            for name, row in d0_components[f"round{round_index}"].items():
                if name != "size=1":
                    non_singleton_rates.append(row["exact_world_set_equal"])
        component_local_safe = (
            non_singleton_rates and np.mean(non_singleton_rates) >= 0.95)
        if component_local_safe:
            case = "CASE_B1_CROSS_COMPONENT_RELATIVE_SLOT_AMBIGUITY"
            recommendation = (
                "study disconnected-component relative slot semantics; do "
                "not modify per-agent Hungarian yet")
        else:
            case = "CASE_B2_WITHIN_COMPONENT_ASSIGNMENT_DEGENERACY"
            recommendation = (
                "design multiple-optimum tie resolution with explicit "
                "lexicographic continuity semantics in a separate review")
    else:
        case = "CASE_C_DEGREE_ZERO_NOT_PRIMARY"
        recommendation = (
            "stop the degree-zero identity direction and localize the "
            "remaining deterministic assignment degeneracy")
    return {
        "case": case,
        "unique_next_recommendation": recommendation,
        "D0_scene_invariance": invariance,
        "D0_scene_hamming": hamming,
        "mean_invariance_improvement_vs_D": improvement,
        "coverage": coverage,
        "relative_metrics_D0_vs_D": metric_relative,
        "metric_preservation_within_2_percent": metric_preserved,
        "productionization_authorized": False,
    }


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--stochasticity-reference", required=True)
    parser.add_argument("--cpsr-reference", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    cli = parser.parse_args()

    output = Path(cli.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    evaluator, epoch = _active_evaluator(
        cli.config, cli.checkpoint, output.parent / "runtime", cli.device)
    if epoch != 13 or not evaluator.net.strict_no_z:
        raise RuntimeError("tie audit requires strict no-z epoch-13")
    checkpoint = torch.load(cli.checkpoint, map_location="cpu")
    architecture = checkpoint.get("architecture_config", {})
    if architecture.get("architecture_variant") != "strict_no_z":
        raise RuntimeError("checkpoint architecture is not strict_no_z")
    stochasticity_path = Path(cli.stochasticity_reference).resolve()
    stochasticity_reference = json.loads(stochasticity_path.read_text())
    if stochasticity_reference.get("status") != \
            "PHASE_A_GATE_FAILED_STOPPED_BEFORE_PHASE_B":
        raise RuntimeError("stochasticity reference has unexpected status")
    cpsr_path = Path(cli.cpsr_reference).resolve()
    cpsr_reference = json.loads(cpsr_path.read_text())

    result = {
        "status": "RUNNING_CATEGORICAL_RNG_CONTROL",
        "protocol": {
            "split": "valid",
            "windows": 139,
            "seeds": list(SEEDS),
            "policies": list(POLICIES),
            "D0_semantics": (
                "degree==0 keeps previous IDs tensor-exact; degree>0 uses "
                "unchanged deterministic injective assignment"),
            "row_near_tolerance": ROW_NEAR_TOLERANCE,
            "permutation_contract": (
                "common permutation applied to conditional-score rows, mask "
                "rows, and previous candidate IDs"),
        },
        "source_fact_review": {
            "conditional_score": "unary[:,None,:] - accumulated_pair_energy",
            "degree_zero_accumulated_pair_energy": 0,
            "degree_zero_rows_theoretically_slot_invariant": True,
            "current_refinement_guard": "window edge_count > 0",
            "isolated_agents_in_Egt0_windows_are_currently_assigned": True,
        },
        "provenance": {
            "checkpoint": str(Path(cli.checkpoint).resolve()),
            "checkpoint_epoch": int(epoch),
            "checkpoint_sha256": file_sha256(cli.checkpoint),
            "architecture_config": architecture,
            "cache_manifest_hash": checkpoint.get("cache_manifest_hash"),
            "source_checkpoint_hash": checkpoint.get("source_checkpoint_hash"),
            "stochasticity_reference": str(stochasticity_path),
            "stochasticity_reference_sha256": file_sha256(stochasticity_path),
            "cpsr_reference": str(cpsr_path),
            "cpsr_reference_sha256": file_sha256(cpsr_path),
        },
        "policies": {},
    }
    _write_json(output, result)

    from tools.cpsr_inference_validation import evaluate_policy as evaluate_cpsr
    reference_states = {}
    categorical = evaluate_cpsr(
        evaluator, "original_categorical", reference_states)
    result["categorical_control"] = categorical
    result["categorical_reproduction_gate"] = reproduction_gate(
        categorical,
        cpsr_reference["policies"]["original_categorical"])
    if not result["categorical_reproduction_gate"]["passed"]:
        result["status"] = "STOPPED_CATEGORICAL_REPRODUCTION_FAILED"
        _write_json(output, result)
        raise RuntimeError("categorical reproduction failed")

    baseline_ids = {}
    policy_d = evaluate_policy(
        evaluator, "D_deterministic", reference_states, baseline_ids)
    result["policies"]["D_deterministic"] = policy_d
    previous_d = stochasticity_reference["policies"][
        "shared_round0_deterministic"]
    result["policy_D_prediction_reproduction_gate"] = reproduction_gate(
        policy_d, previous_d)
    result["policy_D_permutation_reproduction_gate"] = \
        permutation_reproduction_gate(policy_d, previous_d)
    if not (result["policy_D_prediction_reproduction_gate"]["passed"] and
            result["policy_D_permutation_reproduction_gate"]["passed"]):
        result["status"] = "STOPPED_POLICY_D_REPRODUCTION_FAILED"
        _write_json(output, result)
        raise RuntimeError("Policy D reproduction failed")
    result["status"] = "POLICY_D_REPRODUCED_RUNNING_D0"
    _write_json(output, result)

    d0 = evaluate_policy(
        evaluator, "D0_degree0_identity", reference_states, baseline_ids)
    result["policies"]["D0_degree0_identity"] = d0
    expected_windows = len(SEEDS) * 139
    expected_rounds = len(SEEDS) * 92 * 2
    for policy in POLICIES:
        checks = result["policies"][policy]["paired_rng_checks"]
        for boundary in (
                "sampler_global_rng_unchanged", "pre_diffusion", "window_end"):
            if checks.get(boundary) != expected_windows:
                raise RuntimeError(f"incomplete {policy} RNG: {boundary}")
        if checks.get("round_generator_unchanged") != expected_rounds:
            raise RuntimeError(f"incomplete {policy} round RNG checks")
    d0_checks = d0["paired_rng_checks"]
    if d0_checks.get("shared_round0_exact") != expected_windows:
        raise RuntimeError("D0 did not pair every Round 0")
    if d0_checks.get("degree_zero_identity_exact", 0) == 0:
        raise RuntimeError("D0 observed no degree-zero identity checks")
    if d0_checks.get("degree_positive_assignment_exact", 0) == 0:
        raise RuntimeError("D0 observed no degree-positive comparisons")

    result["decision"] = decide_localization(policy_d, d0)
    result["status"] = "AUDIT_COMPLETE_NO_PRODUCTION_CHANGE"
    _write_json(output, result)
    print(f"WROTE {output}", flush=True)


if __name__ == "__main__":
    main()
