#!/usr/bin/env python3
"""Audit persistent stochastic resolution inside exact LCPSR equivalence.

The fourth objective in this file is *not* score noise.  It is used only when
the exact deterministic tuple ``(J, C_stay, C_geom)`` is non-unique.  No
production source is modified.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
import itertools
import json
import math
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch

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
from tools.lexicographic_continuity_audit import (
    TIE_LEVELS,
    _component_sizes,
    _exact_uniqueness,
    _summarize_metric_strata,
    assignment_objective_tuple,
    audit_d0_assignment,
    exact_max_weight_assignment,
    solve_exact_lexicographic,
)
from tools.refinement_coalescence_audit import (
    _agent_and_decomposition_records,
    _summarize_decomposition,
    _summarize_metric_runs,
    _summarize_round_records,
)
from tools.sampler_coverage_intervention import (
    SEEDS,
    _agent_count_bin,
    _degree_bin,
    _degrees,
    _finite_summary,
    capture_rng_state,
    file_sha256,
    restore_rng_state,
    rng_states_equal,
)
from tools.tie_semantics_localization_audit import (
    _match_world_sets,
    component_size_bin,
    marginal_stratum_records,
    summarize_marginal_strata,
)


TIE_REPLICAS = (0, 1, 2, 3, 4)
TIE_NAMESPACE = 0x524553494455414C  # ASCII-ish "RESIDUAL", fixed by protocol.
POLICIES = ("D0_reference", "persistent_exchangeable_exact_tie")


def stable_tie_seed(eval_seed, window_index, tie_replica, agent_index):
    """Stable SplitMix64-style seed mixer; never uses Python ``hash``."""
    mask = (1 << 64) - 1
    value = TIE_NAMESPACE & mask
    for item in (eval_seed, window_index, tie_replica, agent_index):
        value ^= (int(item) + 0x9E3779B97F4A7C15) & mask
        value = (value + 0x9E3779B97F4A7C15) & mask
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
        value ^= value >> 31
    return value & mask


def exchangeable_edge_priorities(mask, seed):
    """Assign a distinct exact binary bit to every valid assignment edge."""
    mask = mask.detach().to(device="cpu", dtype=torch.bool)
    if mask.ndim != 2:
        raise ValueError("mask must have shape [P,K]")
    edges = [(row, column) for row in range(mask.shape[0])
             for column in range(mask.shape[1]) if bool(mask[row, column])]
    local_rng = random.Random(int(seed))
    shuffled = list(edges)
    local_rng.shuffle(shuffled)
    priorities = [[0 for _ in range(mask.shape[1])]
                  for _ in range(mask.shape[0])]
    for rank, (row, column) in enumerate(shuffled):
        priorities[row][column] = 1 << rank
    return tuple(tuple(row) for row in priorities), len(edges)


def random_objective(assignment, priorities):
    return sum(priorities[row][column]
               for row, column in enumerate(assignment))


def extend_exact_full_weight(full_weight, priorities, valid_edge_count):
    """Encode strict ``(deterministic triple, R)`` lexicographic semantics."""
    radix = 1 << int(valid_edge_count)
    extended = tuple(tuple(
        int(full_weight[row][column]) * radix +
        int(priorities[row][column])
        for column in range(len(full_weight[row])))
        for row in range(len(full_weight)))
    return extended, radix


def resolve_residual_equivalence(
        score, mask, previous_ids, goal_candidates, *, priority_seed=None,
        priorities=None, certify_extended=False):
    """Resolve one matrix, invoking R only for a full deterministic tie."""
    deterministic, internals = solve_exact_lexicographic(
        score, mask, previous_ids, goal_candidates)
    return resolve_from_exact(
        deterministic, internals, mask, priority_seed=priority_seed,
        priorities=priorities, certify_extended=certify_extended)


def resolve_from_exact(deterministic, internals, mask, *,
                       priority_seed=None, priorities=None,
                       certify_extended=False):
    """Append R to an already solved exact deterministic triple."""
    base = {
        "deterministic": deterministic,
        "internals": internals,
        "r_invoked": False,
        "r_objective": None,
        "priority_seed": None,
        "extended_unique": deterministic.unique,
        "extended_certificate": None,
    }
    if deterministic.unique:
        return {**base, "assignment": deterministic.assignment,
                "objective": (*deterministic.objective, None)}
    if priorities is None:
        if priority_seed is None:
            raise ValueError("ambiguous exact tuple requires a tie priority")
        priorities, valid_edge_count = exchangeable_edge_priorities(
            mask, priority_seed)
    else:
        priorities = tuple(tuple(int(item) for item in row)
                           for row in priorities)
        valid_edge_count = sum(
            bool(item) for row in mask.detach().cpu().tolist() for item in row)
    rows = len(internals["full_weight"])
    columns = len(internals["full_weight"][0])
    if any(len(row) != columns for row in priorities) or \
            len(priorities) != rows:
        raise ValueError("priority shape differs from score")
    nonzero = [priorities[row][column]
               for row in range(rows)
               for column in range(columns)
               if bool(mask[row, column])]
    if len(set(nonzero)) != valid_edge_count or any(
            value <= 0 or value & (value - 1) for value in nonzero):
        raise ValueError("valid priorities must be distinct powers of two")
    extended_weight, radix = extend_exact_full_weight(
        internals["full_weight"], priorities, valid_edge_count)
    assignment, extended_objective = exact_max_weight_assignment(
        extended_weight, internals["mask"])
    triple = assignment_objective_tuple(
        assignment, internals["primary"], internals["stay"],
        internals["geometry"])
    if triple != deterministic.objective:
        raise AssertionError("R changed the exact deterministic optimum tuple")
    r_value = random_objective(assignment, priorities)
    expected_extended = sum(
        internals["full_weight"][row][column] * radix +
        priorities[row][column]
        for row, column in enumerate(assignment))
    if extended_objective != expected_extended:
        raise AssertionError("extended scalar objective is inconsistent")
    certificate = None
    if certify_extended:
        unique, certificate, _ = _exact_uniqueness(
            extended_weight, internals["mask"], assignment,
            extended_objective)
        if not unique:
            raise AssertionError("powers-of-two R failed to uniqueify assignment")
    return {
        **base,
        "assignment": assignment,
        "objective": (*triple, r_value),
        "r_invoked": True,
        "r_objective": r_value,
        "priority_seed": int(priority_seed) if priority_seed is not None else None,
        "extended_unique": True,
        "extended_certificate": certificate,
        "priorities": priorities,
        "extended_weight": extended_weight,
        "radix": radix,
    }


def transform_residual_problem(score, mask, previous, goals, priorities,
                               slot_permutation=None,
                               candidate_permutation=None):
    """Solve a consistently relabeled frozen problem and map IDs back."""
    score = score.detach().cpu()
    mask = mask.detach().cpu()
    previous = previous.detach().cpu()
    goals = goals.detach().cpu()
    slots, candidates = score.shape
    slot_permutation = (torch.arange(slots) if slot_permutation is None
                        else slot_permutation.cpu())
    candidate_permutation = (
        torch.arange(candidates) if candidate_permutation is None
        else candidate_permutation.cpu())
    inverse = torch.empty_like(candidate_permutation)
    inverse[candidate_permutation] = torch.arange(candidates)
    transformed_priority = tuple(tuple(
        priorities[int(slot_permutation[row])][
            int(candidate_permutation[column])]
        for column in range(candidates)) for row in range(slots))
    result = resolve_residual_equivalence(
        score[slot_permutation][:, candidate_permutation],
        mask[slot_permutation][:, candidate_permutation],
        inverse[previous[slot_permutation]], goals[candidate_permutation],
        priorities=transformed_priority)
    mapped = candidate_permutation[torch.tensor(result["assignment"])]
    restored = torch.empty_like(mapped)
    restored[slot_permutation] = mapped
    return result, tuple(int(item) for item in restored.tolist())


class ResidualAuditController:
    """Audit-only D0 or persistent exact-equivalence stochastic resolver."""

    def __init__(self, sampler, policy, reference_states, tie_replica=0,
                 use_cuda=True):
        if policy not in POLICIES:
            raise ValueError(f"unknown residual audit policy: {policy}")
        self.sampler = sampler
        self.policy = policy
        self.reference_states = reference_states
        self.tie_replica = int(tie_replica)
        self.use_cuda = bool(use_cuda)
        self.original_policy = sampler.refinement_policy
        self.original_callback = sampler.diagnostic_callback
        self.original_refinement = sampler_module.structured_gumbel_assignment
        self.key = None
        self.edge_index = None
        self.degree = None
        self.goal_candidates = None
        self.trace = []
        self.resolution_trace = []
        self.priority_cache = {}
        self.priority_masks = {}
        self.pending_resolution = None
        self.boundary_checks = defaultdict(int)
        self.solver_seconds = []
        self._pre_sampler_state = None

    def _reference(self):
        reference = self.reference_states.get(self.key)
        if reference is None:
            raise RuntimeError(f"missing categorical RNG reference: {self.key}")
        return reference

    def begin_window(self, seed, window_index, edge_index):
        self.key = (int(seed), int(window_index))
        self.edge_index = edge_index.detach().long().clone()
        self.degree = None
        self.goal_candidates = None
        self.trace = []
        self.resolution_trace = []
        self.priority_cache = {}
        self.priority_masks = {}
        self.pending_resolution = None
        restore_rng_state(self._reference()["pre_initial"])
        self.sampler.set_sampling_context(seed, window_index)

    def _record(self, **record):
        item = {
            "score": record["score"].detach().float().clone(),
            "mask": record["mask"].detach().bool().clone(),
            "candidate_index": record[
                "candidate_index"].detach().long().clone(),
        }
        if self.degree is None:
            self.degree = _degrees(
                item["score"].shape[0], self.edge_index).to(
                    item["score"].device)
        self.trace.append(item)
        round_index = int(record["round_index"])
        if round_index == 0:
            self.resolution_trace.append(None)
        else:
            if self.pending_resolution is None:
                raise RuntimeError("refinement callback lacks resolution trace")
            self.resolution_trace.append(self.pending_resolution)
            self.pending_resolution = None

    def _before(self, _module, inputs):
        self._pre_sampler_state = capture_rng_state(self.use_cuda)
        if not rng_states_equal(
                self._pre_sampler_state, self._reference()["pre_initial"]):
            raise RuntimeError("residual audit pre-sampler RNG mismatch")
        self.goal_candidates = inputs[1].detach().float().clone()

    def _after(self, _module, _inputs, _output):
        if not rng_states_equal(
                capture_rng_state(self.use_cuda), self._pre_sampler_state):
            raise RuntimeError("residual resolver consumed global RNG")
        self.boundary_checks["sampler_global_rng_unchanged"] += 1
        restore_rng_state(self._reference()["pre_diffusion"])
        if not rng_states_equal(
                capture_rng_state(self.use_cuda),
                self._reference()["pre_diffusion"]):
            raise RuntimeError("residual audit failed diffusion RNG pairing")
        self.boundary_checks["pre_diffusion"] += 1

    def end_window(self):
        expected = 3 if self.edge_index.shape[1] else 1
        if len(self.trace) != expected:
            raise RuntimeError("residual audit trace has wrong round count")
        if not rng_states_equal(
                capture_rng_state(self.use_cuda),
                self._reference()["window_end"]):
            raise RuntimeError("residual audit window-end RNG mismatch")
        self.boundary_checks["window_end"] += 1
        self.key = None

    def _priority(self, agent, mask):
        if agent not in self.priority_cache:
            seed = stable_tie_seed(
                self.key[0], self.key[1], self.tie_replica, agent)
            priority, _ = exchangeable_edge_priorities(mask, seed)
            self.priority_cache[agent] = priority
            self.priority_masks[agent] = mask.detach().cpu().clone()
            self.boundary_checks["tie_state_created"] += 1
        elif not torch.equal(
                self.priority_masks[agent], mask.detach().cpu()):
            raise RuntimeError("candidate mask changed across refinement rounds")
        return self.priority_cache[agent]

    def __enter__(self):
        def refinement(score, mask, temperature, generator,
                       gumbel_noise=None):
            if not self.trace or self.goal_candidates is None:
                raise RuntimeError("refinement lacks previous worlds")
            before = generator.get_state().clone() if generator is not None \
                else None
            previous = self.trace[-1]["candidate_index"]
            selected = previous.clone()
            local_resolution = []
            for agent in range(score.shape[0]):
                if self.degree[agent] == 0:
                    local_resolution.append({
                        "tie_level": "DEGREE_ZERO_STRUCTURAL_IDENTITY",
                        "r_invoked": False,
                        "assignment": tuple(int(item) for item in
                                            previous[agent].tolist()),
                    })
                    self.boundary_checks["degree_zero_identity"] += 1
                    continue
                if self.policy == "D0_reference":
                    assigned = sampler_module.maximum_weight_injective_assignment(
                        score[agent:agent + 1], mask[agent:agent + 1])[0]
                    selected[agent] = assigned
                    local_resolution.append({
                        "tie_level": "D0_FLOATING_REFERENCE",
                        "r_invoked": False,
                        "assignment": tuple(int(item) for item in
                                            assigned.tolist()),
                    })
                else:
                    started = time.perf_counter()
                    # Generate R lazily only if the deterministic exact tuple
                    # is ambiguous. The stable cache makes it persistent.
                    deterministic, internals = solve_exact_lexicographic(
                        score[agent], mask[agent], previous[agent],
                        self.goal_candidates[agent])
                    if deterministic.unique:
                        resolved = {
                            "deterministic": deterministic,
                            "assignment": deterministic.assignment,
                            "objective": (*deterministic.objective, None),
                            "r_invoked": False,
                            "r_objective": None,
                            "extended_unique": True,
                        }
                    else:
                        priority = self._priority(agent, mask[agent])
                        resolved = resolve_from_exact(
                            deterministic, internals, mask[agent],
                            priorities=priority)
                    self.solver_seconds.append(time.perf_counter() - started)
                    selected[agent] = torch.tensor(
                        resolved["assignment"], dtype=torch.long,
                        device=selected.device)
                    local_resolution.append({
                        "tie_level": deterministic.tie_level,
                        "r_invoked": bool(resolved["r_invoked"]),
                        "assignment": resolved["assignment"],
                        "objective": resolved["objective"],
                    })
                    self.boundary_checks["exact_interacting_assignment"] += 1
                    self.boundary_checks["r_invoked"] += int(
                        resolved["r_invoked"])
            if before is not None and not torch.equal(
                    before, generator.get_state()):
                raise RuntimeError("residual resolver consumed round RNG")
            self.boundary_checks["round_generator_unchanged"] += 1
            self.pending_resolution = local_resolution
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


def _priority_or_zeros(mask, seed, invoked):
    if invoked:
        return exchangeable_edge_priorities(mask, seed)[0]
    return tuple(tuple(0 for _ in range(mask.shape[1]))
                 for _ in range(mask.shape[0]))


def analyze_historical_scene(trace, goals, edge_index, seed, window_index,
                             certify_budget):
    """Phase-1 audit on frozen D0 matrices for all five tie replicas."""
    if len(trace) != 3:
        return [], [], []
    num_agents, slots = trace[1]["candidate_index"].shape
    candidates = trace[1]["score"].shape[-1]
    degree = _degrees(num_agents, edge_index).cpu()
    component_sizes = _component_sizes(num_agents, edge_index)
    generator = torch.Generator().manual_seed(
        int(seed) * 1_000_003 + int(window_index) * 97_409 + 0x5EED)
    slot_permutation = torch.randperm(slots, generator=generator)
    candidate_permutation = torch.randperm(candidates, generator=generator)
    agent_permutation = torch.randperm(num_agents, generator=generator)
    matrix_rows, scene_rows, component_rows = [], [], []
    components = []
    adjacency = [set() for _ in range(num_agents)]
    for source, destination in edge_index.detach().cpu().t().tolist():
        adjacency[source].add(destination)
        adjacency[destination].add(source)
    seen = set()
    for root in range(num_agents):
        if root in seen:
            continue
        stack, component = [root], []
        seen.add(root)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in adjacency[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        components.append(tuple(sorted(component)))

    for round_index in (1, 2):
        item = trace[round_index]
        previous = trace[round_index - 1]["candidate_index"].cpu()
        d0_ids = item["candidate_index"].cpu()
        output_by_replica = {
            replica: previous.clone() for replica in TIE_REPLICAS}
        transformed_outputs = {
            name: previous.clone() for name in (
                "slot", "candidate", "slot_candidate",
                "slot_agent_candidate")}
        for agent in range(num_agents):
            common = {
                "seed": int(seed),
                "window": int(window_index),
                "round": int(round_index),
                "agent": int(agent),
                "degree_bin": _degree_bin(int(degree[agent])),
                "component_size_bin": component_size_bin(
                    component_sizes[agent]),
                "agent_count_bin": _agent_count_bin(num_agents),
            }
            if degree[agent] == 0:
                matrix_rows.append({
                    **common,
                    "tie_level": "DEGREE_ZERO_STRUCTURAL_IDENTITY",
                    "r_invoked": False,
                    "unique_unchanged_all_replicas": True,
                    "triple_preserved_all_replicas": True,
                    "extended_unique_all_replicas": True,
                    "valid_injective_all_replicas": (
                        torch.unique(previous[agent]).numel() == slots),
                    "same_state_reproducible": True,
                    "persistent_state_equal_across_rounds": True,
                    "distinct_representatives": 1,
                    "pairwise_assignment_hamming": 0.0,
                    "slot_equivariant": True,
                    "candidate_equivariant": True,
                    "combined_equivariant": True,
                    "d0_primary_optimal": None,
                    "primary_regret": None,
                })
                continue
            deterministic, internals = solve_exact_lexicographic(
                item["score"][agent], item["mask"][agent], previous[agent],
                goals[agent])
            d0 = audit_d0_assignment(d0_ids[agent], deterministic, internals)
            assignments = []
            objectives = []
            same_state = True
            persistent = True
            for replica in TIE_REPLICAS:
                tie_seed = stable_tie_seed(
                    seed, window_index, replica, agent)
                priority = _priority_or_zeros(
                    item["mask"][agent], tie_seed,
                    not deterministic.unique)
                certify = (not deterministic.unique and certify_budget[0] > 0)
                resolved = resolve_from_exact(
                    deterministic, internals, item["mask"][agent],
                    priorities=priority, certify_extended=certify)
                if certify:
                    certify_budget[0] -= 1
                repeated = resolve_from_exact(
                    deterministic, internals, item["mask"][agent],
                    priorities=priority)
                same_state &= repeated["assignment"] == resolved["assignment"]
                # Re-generation with the same four-key seed must reproduce R;
                # round is intentionally absent from the key.
                persistent &= priority == _priority_or_zeros(
                    item["mask"][agent], tie_seed,
                    not deterministic.unique)
                assignments.append(resolved["assignment"])
                objectives.append(resolved["objective"])
                output_by_replica[replica][agent] = torch.tensor(
                    resolved["assignment"])
            triple_preserved = all(
                objective[:3] == deterministic.objective
                for objective in objectives)
            unique_unchanged = (
                not deterministic.unique or all(
                    assignment == deterministic.assignment
                    for assignment in assignments))
            valid = all(
                len(set(assignment)) == slots and all(
                    bool(item["mask"][agent, row, column])
                    for row, column in enumerate(assignment))
                for assignment in assignments)
            pairs = list(itertools.combinations(assignments, 2))
            hamming = [
                np.mean([left[index] != right[index]
                         for index in range(slots)])
                for left, right in pairs]
            priority0 = _priority_or_zeros(
                item["mask"][agent], stable_tie_seed(
                    seed, window_index, 0, agent),
                not deterministic.unique)
            original = assignments[0]
            slot_result, slot_restored = transform_residual_problem(
                item["score"][agent], item["mask"][agent], previous[agent],
                goals[agent], priority0,
                slot_permutation=slot_permutation)
            candidate_result, candidate_restored = transform_residual_problem(
                item["score"][agent], item["mask"][agent], previous[agent],
                goals[agent], priority0,
                candidate_permutation=candidate_permutation)
            combined_result, combined_restored = transform_residual_problem(
                item["score"][agent], item["mask"][agent], previous[agent],
                goals[agent], priority0,
                slot_permutation=slot_permutation,
                candidate_permutation=candidate_permutation)
            transformed_outputs["slot"][agent] = torch.tensor(slot_restored)
            transformed_outputs["candidate"][agent] = torch.tensor(
                candidate_restored)
            transformed_outputs["slot_candidate"][agent] = torch.tensor(
                combined_restored)
            transformed_outputs["slot_agent_candidate"][agent] = torch.tensor(
                combined_restored)
            objective0 = objectives[0]
            if not (slot_result["objective"] == candidate_result["objective"] ==
                    combined_result["objective"] == objective0):
                raise AssertionError("four-level objective changed under relabeling")
            matrix_rows.append({
                **common,
                "tie_level": deterministic.tie_level,
                "r_invoked": not deterministic.unique,
                "unique_unchanged_all_replicas": unique_unchanged,
                "triple_preserved_all_replicas": triple_preserved,
                "extended_unique_all_replicas": True,
                "valid_injective_all_replicas": valid,
                "same_state_reproducible": same_state,
                "persistent_state_equal_across_rounds": persistent,
                "distinct_representatives": len(set(assignments)),
                "pairwise_assignment_hamming": float(np.mean(hamming)),
                "slot_equivariant": slot_restored == original,
                "candidate_equivariant": candidate_restored == original,
                "combined_equivariant": combined_restored == original,
                "d0_primary_optimal": d0["primary_optimal"],
                "primary_regret": d0["primary_regret"],
            })

        # Agent relabeling is an outer permutation: the per-agent score,
        # previous IDs, goals, mask, and R payload travel together.  Restore
        # that common relabeling before the whole-scene comparison.
        permuted_agents = transformed_outputs[
            "slot_agent_candidate"][agent_permutation]
        restored_agents = torch.empty_like(permuted_agents)
        restored_agents[agent_permutation] = permuted_agents
        transformed_outputs["slot_agent_candidate"] = restored_agents
        original_world = output_by_replica[0].t().contiguous()
        for transform, value in transformed_outputs.items():
            transformed_world = value.t().contiguous()
            scene_match = _match_world_sets(original_world, transformed_world)
            scene_rows.append({
                "seed": int(seed), "window": int(window_index),
                "round": int(round_index), "transform": transform,
                **scene_match,
                "pathwise_invariant": torch.equal(
                    output_by_replica[0], value),
                "support_invariant": torch.equal(
                    output_by_replica[0].sort(-1).values,
                    value.sort(-1).values),
            })
            for component in components:
                agents = torch.tensor(component, dtype=torch.long)
                component_rows.append({
                    "seed": int(seed), "window": int(window_index),
                    "round": int(round_index), "transform": transform,
                    "component_size_bin": component_size_bin(len(component)),
                    **_match_world_sets(
                        output_by_replica[0][agents].t().contiguous(),
                        value[agents].t().contiguous()),
                })
    return matrix_rows, scene_rows, component_rows


def summarize_semantic_phase(matrix_rows, scene_rows, component_rows):
    interacting = [row for row in matrix_rows
                   if row["tie_level"] != "DEGREE_ZERO_STRUCTURAL_IDENTITY"]
    tie_counts = Counter(row["tie_level"] for row in interacting)
    ambiguous = [row for row in interacting
                 if row["tie_level"] == "FULL_TUPLE_AMBIGUOUS"]
    by_field = {}
    for field in ("round", "degree_bin", "component_size_bin",
                  "agent_count_bin"):
        groups = defaultdict(list)
        for row in interacting:
            key = f"round{row[field]}" if field == "round" else row[field]
            groups[key].append(row)
        by_field[field] = {
            key: {
                "count": len(group),
                "tie_levels": dict(Counter(
                    row["tie_level"] for row in group)),
                "ambiguous_rate": float(np.mean([
                    row["tie_level"] == "FULL_TUPLE_AMBIGUOUS"
                    for row in group])),
                "distinct_representatives": _finite_summary([
                    row["distinct_representatives"] for row in group
                    if row["r_invoked"]]),
                "pairwise_assignment_hamming": _finite_summary([
                    row["pairwise_assignment_hamming"] for row in group
                    if row["r_invoked"]]),
            }
            for key, group in sorted(groups.items(), key=lambda item: str(item[0]))
        }
    permutation = {}
    for transform in ("slot", "candidate", "slot_candidate",
                      "slot_agent_candidate"):
        permutation[transform] = {}
        for round_index in (1, 2):
            rows = [row for row in scene_rows
                    if row["transform"] == transform and
                    row["round"] == round_index]
            components = [row for row in component_rows
                          if row["transform"] == transform and
                          row["round"] == round_index]
            permutation[transform][f"round{round_index}"] = {
                "count": len(rows),
                "pathwise_invariance_rate": float(np.mean([
                    row["pathwise_invariant"] for row in rows])),
                "support_invariance_rate": float(np.mean([
                    row["support_invariant"] for row in rows])),
                "scene_world_set_invariance_rate": float(np.mean([
                    row["exact_world_set_equal"] for row in rows])),
                "scene_matched_hamming": _finite_summary([
                    row["matched_mean_normalized_hamming"] for row in rows]),
                "component_world_set_invariance_rate": float(np.mean([
                    row["exact_world_set_equal"] for row in components])),
            }
    return {
        "interacting_matrix_count": len(interacting),
        "degree_zero_identity_count": len(matrix_rows) - len(interacting),
        "tie_level_counts": {level: tie_counts.get(level, 0)
                             for level in TIE_LEVELS},
        "by_stratum": by_field,
        "r_invoked_count": len(ambiguous),
        "historical_ambiguous_diversity": {
            "count": len(ambiguous),
            "distinct_representatives": _finite_summary([
                row["distinct_representatives"] for row in ambiguous]),
            "pairwise_assignment_hamming": _finite_summary([
                row["pairwise_assignment_hamming"] for row in ambiguous]),
            "fraction_more_than_one_representative": float(np.mean([
                row["distinct_representatives"] > 1 for row in ambiguous])),
        },
        "unique_cases_unchanged_rate": float(np.mean([
            row["unique_unchanged_all_replicas"] for row in interacting
            if row["tie_level"] != "FULL_TUPLE_AMBIGUOUS"])),
        "ambiguous_triple_preservation_rate": float(np.mean([
            row["triple_preserved_all_replicas"] for row in ambiguous])),
        "extended_uniqueness_rate": float(np.mean([
            row["extended_unique_all_replicas"] for row in interacting])),
        "valid_injective_coverage_rate": float(np.mean([
            row["valid_injective_all_replicas"] for row in matrix_rows])),
        "same_state_reproducibility_rate": float(np.mean([
            row["same_state_reproducible"] for row in interacting])),
        "persistent_state_rate": float(np.mean([
            row["persistent_state_equal_across_rounds"] for row in interacting])),
        "d0_exact_primary_optimal_rate": float(np.mean([
            row["d0_primary_optimal"] for row in interacting])),
        "d0_primary_regret": _finite_summary([
            row["primary_regret"] for row in interacting]),
        "permutation": permutation,
    }


def semantic_gate(summary):
    expected = {
        "PRIMARY_UNIQUE": 2023,
        "PRIMARY_TIED_RESOLVED_BY_STAY": 128,
        "STAY_TIED_RESOLVED_BY_GEOM": 145,
        "FULL_TUPLE_AMBIGUOUS": 614,
    }
    checks = {
        "matrix_count_2910": summary["interacting_matrix_count"] == 2910,
        "deterministic_tie_counts_reproduced": (
            summary["tie_level_counts"] == expected),
        "d0_exact_primary_optimal_100pct": (
            summary["d0_exact_primary_optimal_rate"] == 1.0),
        "unique_cases_unchanged_100pct": (
            summary["unique_cases_unchanged_rate"] == 1.0),
        "ambiguous_triple_preserved_100pct": (
            summary["ambiguous_triple_preservation_rate"] == 1.0),
        "extended_uniqueness_100pct": (
            summary["extended_uniqueness_rate"] == 1.0),
        "coverage_20_of_20": (
            summary["valid_injective_coverage_rate"] == 1.0),
        "same_state_reproducible_100pct": (
            summary["same_state_reproducibility_rate"] == 1.0),
        "persistent_state_100pct": summary["persistent_state_rate"] == 1.0,
    }
    for transform, rounds in summary["permutation"].items():
        for round_name, row in rounds.items():
            checks[f"{transform}_{round_name}_pathwise_100pct"] = (
                row["pathwise_invariance_rate"] == 1.0)
            checks[f"{transform}_{round_name}_scene_set_100pct"] = (
                row["scene_world_set_invariance_rate"] == 1.0)
    return {"passed": all(checks.values()), "checks": checks}


def _summarize_metric_rows(rows, field):
    groups = defaultdict(list)
    for row in rows:
        groups[row[field]].append(row)
    return {
        key: {
            "count": len(group),
            **{metric: _finite_summary([row[metric] for row in group])
               for metric in AUDIT_METRICS},
        }
        for key, group in sorted(groups.items())
    }


def accumulate_window_metric(metric_values, agent_count_values, edge_class,
                             count_bin, metric, values):
    """Accumulate native metric records without assuming agent cardinality.

    Joint metrics may return scene records while marginal metrics return agent
    records.  Degree-level marginal statistics are computed separately from
    trajectory tensors, so this function deliberately preserves each metric's
    native reduction.
    """
    metric_values["overall"][metric].extend(values)
    metric_values[edge_class][metric].extend(values)
    agent_count_values[count_bin][metric].extend(values)


@torch.no_grad()
def evaluate_policy(evaluator, policy, reference_states, tie_replica=0,
                    run_semantic=False):
    net = evaluator.net
    net.eval()
    controller = ResidualAuditController(
        net.jdv2_sampler, policy, reference_states, tie_replica=tie_replica,
        use_cuda=evaluator.device.type == "cuda")
    metric_runs = []
    marginal_rows = []
    candidate_rows, decomposition_rows, churn_rows = [], [], []
    assignment_rows, world_rows = [], []
    semantic_matrix_rows, semantic_scene_rows, semantic_component_rows = [], [], []
    certify_budget = [16]
    sampler_times, forward_times, runtime_per_seed = [], [], []
    if evaluator.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(evaluator.device)
    for seed in SEEDS:
        print(
            f"RESIDUAL policy={policy} replica={tie_replica} seed={seed}",
            flush=True)
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
                    edge_class = "E>0" if edge_index.shape[1] else "E=0"
                    controller.begin_window(seed, window_index, edge_index)
                    if evaluator.device.type == "cuda":
                        torch.cuda.synchronize(evaluator.device)
                    started = time.perf_counter()
                    with evaluator._autocast_context():
                        prediction, auxiliary = net.forward(inputs, if_test=True)
                    if evaluator.device.type == "cuda":
                        torch.cuda.synchronize(evaluator.device)
                    forward_times.append(time.perf_counter() - started)
                    controller.end_window()
                    num_agents = int(auxiliary["unary_score"].shape[0])
                    degree = _degrees(num_agents, edge_index).cpu()
                    component_sizes = _component_sizes(num_agents, edge_index)
                    count_bin = _agent_count_bin(num_agents)
                    for metric in AUDIT_METRICS:
                        values = net.compute_model_metrics(
                            metric_name=metric, predictions=prediction,
                            metric_mask=metric_mask,
                            all_aux_outputs=auxiliary, inputs=inputs,
                            obs_length=net.args.obs_length)
                        accumulate_window_metric(
                            metric_values, agent_count_values, edge_class,
                            count_bin, metric, values)
                    marginal_rows.extend(marginal_stratum_records(
                        net, inputs, prediction, auxiliary, metric_mask,
                        f"{policy}_replica{tie_replica}", seed,
                        window_index))
                    records = _agent_and_decomposition_records(
                        net, inputs, prediction, auxiliary, metric_mask,
                        controller.trace,
                        f"{policy}_replica{tie_replica}", seed,
                        window_index)
                    candidate_rows.extend(records[0])
                    decomposition_rows.extend(records[1])
                    churn_rows.extend(_churn_records(
                        controller.trace,
                        _metadata(num_agents, edge_index, policy, seed,
                                  window_index), metric_mask))
                    final_ids = controller.trace[-1][
                        "candidate_index"].detach().cpu()
                    world_rows.append({
                        "seed": int(seed), "window": int(window_index),
                        "edge_class": edge_class,
                        "ids": final_ids.t().contiguous(),
                    })
                    for round_index in range(1, len(controller.trace)):
                        resolution = controller.resolution_trace[round_index]
                        for agent in range(num_agents):
                            assignment_rows.append({
                                "seed": int(seed), "window": int(window_index),
                                "round": int(round_index), "agent": int(agent),
                                "degree_bin": _degree_bin(int(degree[agent])),
                                "component_size_bin": component_size_bin(
                                    component_sizes[agent]),
                                "tie_level": resolution[agent]["tie_level"],
                                "r_invoked": resolution[agent]["r_invoked"],
                                "ids": tuple(int(item) for item in
                                             controller.trace[round_index][
                                                 "candidate_index"][agent]
                                             .detach().cpu().tolist()),
                            })
                    if run_semantic and edge_index.shape[1]:
                        local = analyze_historical_scene(
                            controller.trace, controller.goal_candidates,
                            edge_index, seed, window_index, certify_budget)
                        semantic_matrix_rows.extend(local[0])
                        semantic_scene_rows.extend(local[1])
                        semantic_component_rows.extend(local[2])
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
                    for metric, items in values.items()
                }
                for name, values in agent_count_values.items()
            },
        })
    runtime = _runtime_summary(
        runtime_per_seed, sampler_times, forward_times, evaluator.device)
    runtime["exact_solver_seconds"] = _finite_summary(
        controller.solver_seconds)
    runtime["exact_solver_total_seconds"] = float(sum(
        controller.solver_seconds))
    summary = {
        "policy": policy,
        "tie_replica": int(tie_replica),
        "metric_runs": metric_runs,
        "metric_summary": _summarize_metric_runs(metric_runs),
        "edge_metric_summary": {
            edge_class: {
                metric: _finite_summary([
                    run[edge_class][metric] for run in metric_runs])
                for metric in AUDIT_METRICS
            }
            for edge_class in ("E=0", "E>0")
        },
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
        "round_candidate_summary": _summarize_round_records(candidate_rows),
        "error_decomposition": _summarize_decomposition(decomposition_rows),
        "slot_churn_and_two_cycle": _summarize_churn(churn_rows),
        "runtime": runtime,
        "paired_rng_checks": dict(controller.boundary_checks),
    }
    if run_semantic:
        summary["semantic_phase"] = summarize_semantic_phase(
            semantic_matrix_rows, semantic_scene_rows,
            semantic_component_rows)
    return summary, assignment_rows, world_rows


def summarize_tie_sensitivity(replica_outputs, d0_output):
    """Separate fixed-eval-seed tie variation from eval-seed variation."""
    metrics = list(AUDIT_METRICS)
    metric_by_seed = {}
    aggregate_by_replica = {}
    for replica, output in replica_outputs.items():
        aggregate_by_replica[str(replica)] = {
            metric: output["metric_summary"]["overall"][metric]["mean"]
            for metric in metrics
        }
    for seed in SEEDS:
        rows = {
            replica: next(row for row in output["metric_runs"]
                          if row["seed"] == seed)["overall"]
            for replica, output in replica_outputs.items()
        }
        d0_row = next(row for row in d0_output["metric_runs"]
                      if row["seed"] == seed)["overall"]
        metric_by_seed[str(seed)] = {
            metric: {
                "values_by_replica": {
                    str(replica): rows[replica][metric]
                    for replica in TIE_REPLICAS},
                "mean": float(np.mean([
                    rows[replica][metric] for replica in TIE_REPLICAS])),
                "std": float(np.std([
                    rows[replica][metric] for replica in TIE_REPLICAS])),
                "range": float(np.ptp([
                    rows[replica][metric] for replica in TIE_REPLICAS])),
                "max_absolute_deviation_from_D0": float(max(
                    abs(rows[replica][metric] - d0_row[metric])
                    for replica in TIE_REPLICAS)),
                "D0": d0_row[metric],
            }
            for metric in metrics
        }
    aggregate_spread = {
        metric: {
            "minimum": min(aggregate_by_replica[str(replica)][metric]
                           for replica in TIE_REPLICAS),
            "maximum": max(aggregate_by_replica[str(replica)][metric]
                           for replica in TIE_REPLICAS),
            "relative_spread_vs_D0": (
                (max(aggregate_by_replica[str(replica)][metric]
                     for replica in TIE_REPLICAS) -
                 min(aggregate_by_replica[str(replica)][metric]
                     for replica in TIE_REPLICAS)) /
                d0_output["metric_summary"]["overall"][metric]["mean"]),
            "D0_eval_seed_std": d0_output["metric_summary"]["overall"][
                metric]["std"],
            "mean_tie_only_std": float(np.mean([
                metric_by_seed[str(seed)][metric]["std"] for seed in SEEDS])),
        }
        for metric in metrics
    }
    return {
        "by_fixed_eval_seed": metric_by_seed,
        "aggregate_by_tie_replica": aggregate_by_replica,
        "aggregate_spread": aggregate_spread,
    }


def summarize_assignment_sensitivity(assignment_by_replica):
    grouped = defaultdict(dict)
    for replica, rows in assignment_by_replica.items():
        for row in rows:
            key = (row["seed"], row["window"], row["round"], row["agent"])
            grouped[key][replica] = row
    records = []
    for key, values in grouped.items():
        if len(values) != len(TIE_REPLICAS):
            raise RuntimeError("missing tie replica assignment record")
        assignments = [values[replica]["ids"] for replica in TIE_REPLICAS]
        pairs = list(itertools.combinations(assignments, 2))
        hamming = [np.mean([left[index] != right[index]
                            for index in range(len(left))])
                   for left, right in pairs]
        first = values[TIE_REPLICAS[0]]
        records.append({
            "round": first["round"],
            "degree_bin": first["degree_bin"],
            "component_size_bin": first["component_size_bin"],
            "any_r_invoked": any(row["r_invoked"] for row in values.values()),
            "distinct_representatives": len(set(assignments)),
            "pairwise_hamming": float(np.mean(hamming)),
        })
    result = {}
    for field in ("round", "degree_bin", "component_size_bin"):
        groups = defaultdict(list)
        for row in records:
            key = f"round{row[field]}" if field == "round" else row[field]
            groups[key].append(row)
        result[field] = {
            key: {
                "count": len(group),
                "r_invoked_rate": float(np.mean([
                    row["any_r_invoked"] for row in group])),
                "distinct_representatives": _finite_summary([
                    row["distinct_representatives"] for row in group]),
                "pairwise_hamming": _finite_summary([
                    row["pairwise_hamming"] for row in group]),
            }
            for key, group in sorted(groups.items(), key=lambda item: str(item[0]))
        }
    return result


def summarize_world_sensitivity(world_by_replica):
    grouped = defaultdict(dict)
    for replica, rows in world_by_replica.items():
        for row in rows:
            grouped[(row["seed"], row["window"])][replica] = row
    records = []
    for key, values in grouped.items():
        matches = []
        for left, right in itertools.combinations(TIE_REPLICAS, 2):
            matches.append(_match_world_sets(
                values[left]["ids"], values[right]["ids"]))
        records.append({
            "edge_class": values[0]["edge_class"],
            "matched_mean_hamming": float(np.mean([
                row["matched_mean_normalized_hamming"] for row in matches])),
            "matched_exact_world_fraction": float(np.mean([
                row["matched_exact_world_fraction"] for row in matches])),
            "exact_whole_set_all_pairs": all(
                row["exact_world_set_equal"] for row in matches),
        })
    result = {}
    for name, rows in (
            ("overall", records),
            ("E=0", [row for row in records if row["edge_class"] == "E=0"]),
            ("E>0", [row for row in records if row["edge_class"] == "E>0"])):
        result[name] = {
            "count": len(rows),
            "matched_mean_hamming": _finite_summary([
                row["matched_mean_hamming"] for row in rows]),
            "matched_exact_world_fraction": _finite_summary([
                row["matched_exact_world_fraction"] for row in rows]),
            "exact_whole_set_all_pairs_rate": float(np.mean([
                row["exact_whole_set_all_pairs"] for row in rows])),
        }
    return result


def phase0_gate(semantic, exact_reference):
    """Reproduce the frozen exact-matrix census before testing R."""
    expected_phase = exact_reference["phase_1"]
    expected_ties = {
        "PRIMARY_UNIQUE": 2023,
        "PRIMARY_TIED_RESOLVED_BY_STAY": 128,
        "STAY_TIED_RESOLVED_BY_GEOM": 145,
        "FULL_TUPLE_AMBIGUOUS": 614,
    }
    checks = {
        "matrix_count_reproduced": (
            semantic["interacting_matrix_count"] ==
            expected_phase["matrix_count"] == 2910),
        "degree_zero_count_reproduced": (
            semantic["degree_zero_identity_count"] ==
            expected_phase["degree_zero_identity_count"] == 300),
        "tie_counts_reproduced": semantic["tie_level_counts"] == expected_ties,
        "D0_primary_optimal_reproduced": (
            semantic["d0_exact_primary_optimal_rate"] == 1.0 and
            expected_phase["all_exact_primary_optimal"]),
        "D0_zero_primary_regret": (
            semantic["d0_primary_regret"]["mean"] == 0.0 and
            semantic["d0_primary_regret"]["std"] == 0.0),
    }
    return {"passed": all(checks.values()), "checks": checks}


def phase2_adoption_gate(replica_outputs, d0_output, gdts):
    """Apply the frozen ±2% stability and GDTS comparison rules."""
    metrics = [
        "minADE@K", "minFDE@K", "JADE", "JFDE",
        "Joint_Goal_Endpoint_Error", "Relative_Motion_Error",
    ]
    d0 = {
        metric: d0_output["metric_summary"]["overall"][metric]["mean"]
        for metric in metrics
    }
    replica = {
        str(index): {
            metric: output["metric_summary"]["overall"][metric]["mean"]
            for metric in metrics
        }
        for index, output in replica_outputs.items()
    }
    per_replica = {
        index: {
            metric: (values[metric] - d0[metric]) / d0[metric]
            for metric in metrics
        }
        for index, values in replica.items()
    }
    spread = {
        metric: (
            max(values[metric] for values in replica.values()) -
            min(values[metric] for values in replica.values())) / d0[metric]
        for metric in metrics
    }
    checks = {
        "every_replica_every_metric_within_2pct_of_D0": all(
            abs(value) <= 0.02
            for values in per_replica.values() for value in values.values()),
        "maximum_aggregate_replica_spread_at_most_2pct": all(
            value <= 0.02 for value in spread.values()),
        "minFDE_vs_GDTS_at_most_plus_2pct": all(
            (values["minFDE@K"] - gdts["minFDE@K"]["mean"]) /
            gdts["minFDE@K"]["mean"] <= 0.02
            for values in replica.values()),
        "joint_metrics_remain_better_than_GDTS": all(
            values[metric] < gdts[metric]["mean"]
            for values in replica.values()
            for metric in (
                "JADE", "JFDE", "Joint_Goal_Endpoint_Error",
                "Relative_Motion_Error")),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "relative_change_each_replica_vs_D0": per_replica,
        "relative_replica_spread_vs_D0": spread,
        "interpretation": (
            "eligible_for_separate_production_adoption_validation"
            if all(checks.values()) else
            "tie_representative_sensitivity_is_material"),
    }


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _git_head():
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, text=True,
        capture_output=True).stdout.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cpsr-reference", required=True)
    parser.add_argument("--tie-reference", required=True)
    parser.add_argument("--exact-reference", required=True)
    parser.add_argument("--gdts-reference", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    cli = parser.parse_args()

    output = Path(cli.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    evaluator, epoch = _active_evaluator(
        cli.config, cli.checkpoint, output.parent / "runtime", cli.device)
    if epoch != 13 or not evaluator.net.strict_no_z:
        raise RuntimeError("audit requires strict no-z epoch-13 checkpoint")
    checkpoint = torch.load(cli.checkpoint, map_location="cpu")
    architecture = checkpoint.get("architecture_config", {})
    if architecture.get("architecture_variant") != "strict_no_z":
        raise RuntimeError("checkpoint architecture is not strict_no_z")

    reference_paths = {
        "cpsr": Path(cli.cpsr_reference).resolve(),
        "tie": Path(cli.tie_reference).resolve(),
        "exact": Path(cli.exact_reference).resolve(),
        "gdts": Path(cli.gdts_reference).resolve(),
    }
    references = {
        name: json.loads(path.read_text())
        for name, path in reference_paths.items()
    }
    if references["cpsr"].get("status") != "VALIDATION_COMPLETE":
        raise RuntimeError("CPSR reference is incomplete")
    if references["tie"].get("status") != \
            "AUDIT_COMPLETE_NO_PRODUCTION_CHANGE":
        raise RuntimeError("tie-semantics reference is incomplete")
    if references["exact"].get("status") != \
            "PHASE_1_SEMANTIC_GATE_FAILED_STOPPED_BEFORE_PHASE_2":
        raise RuntimeError("exact semantic reference has unexpected status")
    gdts = references["gdts"]["interventions"]["gdts"]["summary"]

    result = {
        "status": "RUNNING_BASELINE_REPRODUCTION",
        "protocol": {
            "name": "RESIDUAL_EXACT_EQUIVALENCE_STOCHASTIC_AUDIT",
            "split": "ETH validation",
            "windows": 139,
            "evaluation_seeds": list(SEEDS),
            "tie_replicas": list(TIE_REPLICAS),
            "K_P_M_rank": [21, 20, 4, 8],
            "degree_zero_semantics": "structural identity",
            "interacting_semantics": (
                "maximize exact (J,C_stay,C_geom); only a residual full "
                "equivalence class receives persistent exchangeable R"),
            "R_definition": (
                "a local seeded permutation rho of valid assignment edges, "
                "R[e]=2^rho(e); not a posterior, Gibbs, MCMC, Gumbel, or "
                "uniform draw over assignments"),
            "R_key": "(evaluation_seed, window, tie_replica, agent)",
            "R_persistent_across_rounds": True,
            "rng_isolation": (
                "local CPU random.Random for R; categorical pre-initial, "
                "pre-diffusion, and window-end RNG states are paired"),
        },
        "provenance": {
            "source_commit": _git_head(),
            "checkpoint": str(Path(cli.checkpoint).resolve()),
            "checkpoint_epoch": int(epoch),
            "checkpoint_sha256": file_sha256(cli.checkpoint),
            "architecture_config": architecture,
            "cache_manifest_hash": checkpoint.get("cache_manifest_hash"),
            "source_checkpoint_hash": checkpoint.get("source_checkpoint_hash"),
            "references": {
                name: {"path": str(path), "sha256": file_sha256(path)}
                for name, path in reference_paths.items()
            },
        },
        "phase_0": {},
        "phase_1": {},
        "phase_2": {},
    }
    _write_json(output, result)

    # The unchanged categorical run supplies the paired downstream RNG states.
    from tools.cpsr_inference_validation import evaluate_policy as evaluate_cpsr
    reference_states = {}
    categorical = evaluate_cpsr(
        evaluator, "original_categorical", reference_states)
    categorical_gate = reproduction_gate(
        categorical,
        references["cpsr"]["policies"]["original_categorical"])
    result["phase_0"]["categorical_reproduction_gate"] = categorical_gate
    if not categorical_gate["passed"]:
        result["status"] = "STOPPED_CATEGORICAL_REPRODUCTION_FAILED"
        _write_json(output, result)
        raise RuntimeError("categorical reproduction failed")

    d0, _, _ = evaluate_policy(
        evaluator, "D0_reference", reference_states, run_semantic=True)
    d0_gate = reproduction_gate(
        d0, references["tie"]["policies"]["D0_degree0_identity"])
    result["phase_0"]["D0_reproduction_gate"] = d0_gate
    result["phase_0"]["D0"] = d0
    if not d0_gate["passed"]:
        result["status"] = "STOPPED_D0_REPRODUCTION_FAILED"
        _write_json(output, result)
        raise RuntimeError("D0 reproduction failed")

    result["phase_0"]["exact_matrix_reproduction_gate"] = phase0_gate(
        d0["semantic_phase"], references["exact"])
    result["phase_1"]["semantic_summary"] = d0["semantic_phase"]
    result["phase_1"]["gate"] = semantic_gate(d0["semantic_phase"])
    if not (result["phase_0"]["exact_matrix_reproduction_gate"]["passed"]
            and result["phase_1"]["gate"]["passed"]):
        result["status"] = "STOPPED_PHASE_1_SEMANTIC_GATE_FAILED"
        _write_json(output, result)
        raise RuntimeError("residual stochastic semantic gate failed")

    expected_windows = len(SEEDS) * 139
    expected_rounds = len(SEEDS) * 92 * 2
    d0_rng = d0["paired_rng_checks"]
    for boundary in (
            "sampler_global_rng_unchanged", "pre_diffusion", "window_end"):
        if d0_rng.get(boundary) != expected_windows:
            raise RuntimeError(f"incomplete D0 RNG assertion: {boundary}")
    if d0_rng.get("round_generator_unchanged") != expected_rounds:
        raise RuntimeError("incomplete D0 refinement RNG assertions")

    result["status"] = "PHASE_1_PASSED_RUNNING_PHASE_2"
    _write_json(output, result)
    replica_outputs, assignment_rows, world_rows = {}, {}, {}
    for replica in TIE_REPLICAS:
        evaluated, assignments, worlds = evaluate_policy(
            evaluator, "persistent_exchangeable_exact_tie",
            reference_states, tie_replica=replica)
        replica_outputs[replica] = evaluated
        assignment_rows[replica] = assignments
        world_rows[replica] = worlds
        checks = evaluated["paired_rng_checks"]
        for boundary in (
                "sampler_global_rng_unchanged", "pre_diffusion", "window_end"):
            if checks.get(boundary) != expected_windows:
                raise RuntimeError(
                    f"replica {replica} incomplete RNG: {boundary}")
        if checks.get("round_generator_unchanged") != expected_rounds:
            raise RuntimeError(
                f"replica {replica} incomplete round RNG checks")
        result["phase_2"].setdefault("replicas", {})[str(replica)] = evaluated
        _write_json(output, result)

    result["phase_2"]["tie_only_metric_sensitivity"] = \
        summarize_tie_sensitivity(replica_outputs, d0)
    result["phase_2"]["assignment_sensitivity"] = \
        summarize_assignment_sensitivity(assignment_rows)
    result["phase_2"]["joint_world_sensitivity"] = \
        summarize_world_sensitivity(world_rows)
    fresh = references["cpsr"]["policies"][
        "structured_gumbel_assignment"]["slot_churn_and_two_cycle"]
    result["phase_2"]["fresh_gumbel_churn_reference"] = fresh
    result["phase_2"]["persistent_R_churn_by_replica"] = {
        str(replica): output["slot_churn_and_two_cycle"]
        for replica, output in replica_outputs.items()
    }
    result["phase_2"]["adoption_gate"] = phase2_adoption_gate(
        replica_outputs, d0, gdts)
    result["status"] = "AUDIT_COMPLETE_NO_PRODUCTION_CHANGE"
    _write_json(output, result)
    print(f"WROTE {output}", flush=True)


if __name__ == "__main__":
    main()
