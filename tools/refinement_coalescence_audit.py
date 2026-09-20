#!/usr/bin/env python3
"""Read-only decomposition of JDV2 refinement-induced slot coalescence.

This audit never changes the production sampler source or checkpoint.  A
context manager intercepts the existing categorical primitive, applies the
requested diagnostic policy, traces each round, and restores the primitive
and sampler configuration on exit.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from src.jdv2_audit import AUDIT_METRICS
from src.metrics import compute_metric_mask
import src.models.joint_dependency_v2.joint_sampler as sampler_module
from src.utils import isolated_random_seed
from tools.audit_jdv2_stage_a import _active_evaluator
from tools.sampler_coverage_intervention import (
    SEEDS,
    _agent_count_bin,
    _degree_bin,
    _degrees,
    _finite_summary,
    _initial_generator,
    _rank_tensor,
    _summarize_metric_runs,
    _world_ground_truth,
    _world_predictions,
    capture_rng_state,
    file_sha256,
    initial_candidate_ids,
    restore_rng_state,
    rng_states_equal,
)


VARIANTS = ("R0", "R1", "R2", "MAP", "ASSIGN")
ALL_POLICIES = ("iid_control",) + VARIANTS
ROUND_COUNT = {
    "iid_control": 2,
    "R0": 0,
    "R1": 1,
    "R2": 2,
    "MAP": 2,
    "ASSIGN": 2,
}


def refinement_map(score, mask):
    """Independent deterministic per-slot MAP over valid candidates."""
    if score.ndim != 3 or mask.shape != score.shape:
        raise ValueError("score/mask must have shape [N,P,K]")
    valid_score = score.float().masked_fill(~mask, float("-inf"))
    if not torch.isfinite(valid_score[mask]).all():
        raise FloatingPointError("valid refinement scores contain NaN/Inf")
    if bool((mask.sum(-1) == 0).any()):
        raise ValueError("a slot has no valid refinement candidate")
    return valid_score.argmax(-1)


def coverage_assignment(score, mask):
    """Maximum-weight injective assignment for each independent agent."""
    if score.ndim != 3 or mask.shape != score.shape:
        raise ValueError("score/mask must have shape [N,P,K]")
    num_agents, num_slots, num_candidates = score.shape
    if num_slots > num_candidates:
        raise ValueError(
            f"injective assignment requires P<=K; got P={num_slots}, "
            f"K={num_candidates}")
    if not torch.isfinite(score[mask]).all():
        raise FloatingPointError("valid assignment scores contain NaN/Inf")
    result = torch.empty(
        (num_agents, num_slots), dtype=torch.long, device=score.device)
    score_cpu = score.detach().float().cpu().numpy()
    mask_cpu = mask.detach().cpu().numpy().astype(bool)
    for agent in range(num_agents):
        valid_per_row = mask_cpu[agent].sum(axis=1)
        if np.any(valid_per_row == 0):
            raise ValueError("an assignment row has no valid candidate")
        valid_columns = np.any(mask_cpu[agent], axis=0)
        if int(valid_columns.sum()) < num_slots:
            raise ValueError("fewer than P assignment candidates are valid")
        cost = -score_cpu[agent].astype(np.float64, copy=True)
        finite_scale = max(float(np.abs(cost[mask_cpu[agent]]).max()), 1.0)
        invalid_cost = finite_scale * 1e9
        cost[~mask_cpu[agent]] = invalid_cost
        rows, columns = linear_sum_assignment(cost)
        if rows.size != num_slots or not np.array_equal(
                np.sort(rows), np.arange(num_slots)):
            raise RuntimeError("Hungarian solver returned an incomplete match")
        ordered = np.empty(num_slots, dtype=np.int64)
        ordered[rows] = columns
        if not mask_cpu[agent, np.arange(num_slots), ordered].all():
            raise RuntimeError("Hungarian solver selected an invalid candidate")
        if np.unique(ordered).size != num_slots:
            raise RuntimeError("Hungarian assignment is not injective")
        result[agent] = torch.from_numpy(ordered).to(result.device)
    return result


class RefinementAuditController:
    """Trace/replace sampler calls while pairing every downstream RNG stage."""

    def __init__(self, sampler, variant, reference_states, use_cuda=True):
        if variant not in ALL_POLICIES:
            raise ValueError(f"unknown refinement variant: {variant}")
        self.sampler = sampler
        self.variant = variant
        self.reference_states = reference_states
        self.use_cuda = bool(use_cuda)
        self.original_categorical = sampler_module._categorical
        self.original_steps = int(sampler.num_refinement_steps)
        self.key = None
        self.call_index = 0
        self.expected_calls = None
        self.edge_count = None
        self.init_generator = None
        self.trace = []
        self.boundary_checks = defaultdict(int)

    def begin_window(self, seed, window_index, edge_count, device):
        self.key = (int(seed), int(window_index))
        self.call_index = 0
        self.edge_count = int(edge_count)
        active_rounds = ROUND_COUNT[self.variant] if self.edge_count else 0
        self.expected_calls = 1 + active_rounds
        self.init_generator = (
            _initial_generator(seed, window_index, device)
            if self.variant != "iid_control" else None)
        self.trace = []

    def _reference(self):
        reference = self.reference_states.get(self.key)
        if reference is None:
            raise RuntimeError(f"missing iid RNG reference for {self.key}")
        return reference

    def _restore_and_check(self, name):
        expected = self._reference()[name]
        restore_rng_state(expected)
        if not rng_states_equal(capture_rng_state(self.use_cuda), expected):
            raise RuntimeError(f"failed to restore RNG boundary {name}")
        self.boundary_checks[name] += 1

    def _record(self, score, mask, selected):
        self.trace.append({
            "score": score.detach().float().clone(),
            "mask": mask.detach().bool().clone(),
            "candidate_index": selected.detach().long().clone(),
        })

    def _select(self, score, mask, temperature, sampling_mode, generator):
        if self.call_index == 0:
            if self.variant == "iid_control":
                return self.original_categorical(
                    score, mask, temperature, sampling_mode, generator)
            return initial_candidate_ids(
                score, mask, temperature, sampling_mode,
                self.init_generator, "weighted_without_replacement",
                num_samples=score.shape[1])
        if self.variant in {"iid_control", "R1", "R2"}:
            return self.original_categorical(
                score, mask, temperature, sampling_mode, generator)
        if self.variant == "MAP":
            return refinement_map(score, mask)
        if self.variant == "ASSIGN":
            return coverage_assignment(score, mask)
        raise RuntimeError("R0 unexpectedly entered refinement")

    def __enter__(self):
        if sampler_module._categorical is not self.original_categorical:
            raise RuntimeError("categorical primitive is already patched")
        self.sampler.num_refinement_steps = ROUND_COUNT[self.variant]

        def categorical(score, mask, temperature, sampling_mode, generator):
            if self.key is None:
                raise RuntimeError("begin_window was not called")
            if self.call_index >= self.expected_calls:
                raise RuntimeError("too many categorical calls in one window")
            call = self.call_index
            if call == 0 and self.variant == "iid_control":
                self.reference_states[self.key] = {
                    "pre_initial": capture_rng_state(self.use_cuda)}
            elif call == 0:
                restore_rng_state(self._reference()["pre_initial"])

            selected = self._select(
                score, mask, temperature, sampling_mode, generator)
            self._record(score, mask, selected)

            if self.variant == "iid_control":
                if call == 0 and self.expected_calls > 1:
                    name = "pre_round1"
                elif call == 1 and self.expected_calls > 2:
                    name = "pre_round2"
                else:
                    name = "pre_diffusion"
                self.reference_states[self.key][name] = capture_rng_state(
                    self.use_cuda)
            else:
                if call == 0 and self.expected_calls > 1:
                    self._restore_and_check("pre_round1")
                elif call == 1 and self.expected_calls > 2:
                    self._restore_and_check("pre_round2")
                else:
                    self._restore_and_check("pre_diffusion")

            self.call_index += 1
            return selected

        sampler_module._categorical = categorical
        return self

    def end_window(self):
        if self.call_index != self.expected_calls:
            raise RuntimeError(
                f"categorical call mismatch: {self.call_index} != "
                f"{self.expected_calls}")
        state = capture_rng_state(self.use_cuda)
        if self.variant == "iid_control":
            self.reference_states[self.key]["window_end"] = state
        elif not rng_states_equal(
                state, self._reference()["window_end"]):
            raise RuntimeError("window-end RNG diverged from iid control")
        else:
            self.boundary_checks["window_end"] += 1
        self.key = None

    def __exit__(self, exc_type, exc, traceback):
        sampler_module._categorical = self.original_categorical
        self.sampler.num_refinement_steps = self.original_steps


def _pairwise_values(matrix):
    count = matrix.shape[0]
    if count < 2:
        return matrix.new_empty((0,))
    indices = torch.triu_indices(count, count, offset=1, device=matrix.device)
    return matrix[indices[0], indices[1]]


def _set_jaccard(top_index, num_candidates):
    indicator = F.one_hot(top_index, num_classes=num_candidates).any(1).float()
    intersection = indicator @ indicator.transpose(0, 1)
    size = indicator.sum(-1)
    union = size[:, None] + size[None, :] - intersection
    return _pairwise_values(intersection / union.clamp_min(1.0))


def _rank_correlation(reference_score, conditional_score, mask):
    """Spearman correlation over K for each [agent,slot]."""
    masked_reference = reference_score[:, None, :].expand_as(
        conditional_score).masked_fill(~mask, float("-inf"))
    masked_conditional = conditional_score.masked_fill(~mask, float("-inf"))
    ref_order = torch.argsort(
        masked_reference, dim=-1, descending=True, stable=True)
    cond_order = torch.argsort(
        masked_conditional, dim=-1, descending=True, stable=True)
    ref_rank = torch.empty_like(ref_order)
    cond_rank = torch.empty_like(cond_order)
    positions = torch.arange(
        conditional_score.shape[-1], device=conditional_score.device
    ).view(1, 1, -1).expand_as(ref_order)
    ref_rank.scatter_(-1, ref_order, positions)
    cond_rank.scatter_(-1, cond_order, positions)
    valid = mask.float()
    count = valid.sum(-1).clamp_min(2.0)
    ref = ref_rank.float()
    cond = cond_rank.float()
    ref_mean = (ref * valid).sum(-1) / count
    cond_mean = (cond * valid).sum(-1) / count
    ref_center = (ref - ref_mean[..., None]) * valid
    cond_center = (cond - cond_mean[..., None]) * valid
    numerator = (ref_center * cond_center).sum(-1)
    denominator = torch.sqrt(
        ref_center.square().sum(-1) * cond_center.square().sum(-1)
    ).clamp_min(1e-12)
    return numerator / denominator


def _score_statistics(trace, temperature):
    """Aggregate conditional score geometry without persisting dense scores."""
    if len(trace) <= 1:
        return {}
    unary = trace[0]["score"][:, 0]
    result = {}
    for round_index, item in enumerate(trace[1:], start=1):
        score = item["score"].float()
        mask = item["mask"]
        selected = item["candidate_index"]
        masked = score.masked_fill(~mask, float("-inf"))
        probability = F.softmax(masked / float(temperature), dim=-1)
        log_probability = torch.log(probability.clamp_min(1e-12))
        entropy = -(probability * log_probability).sum(-1)
        maximum_probability = probability.max(-1).values
        argmax = masked.argmax(-1)
        pair_term = unary[:, None, :] - score
        valid_pair_term = pair_term.masked_fill(~mask, 0.0)
        pair_variance = valid_pair_term.var(dim=1, unbiased=False).mean(-1)
        rank_correlation = _rank_correlation(unary, score, mask)
        selected_score = score.gather(-1, selected[..., None]).squeeze(-1)
        argmax_score = masked.max(-1).values
        unary_top1 = unary.masked_fill(~mask[:, 0], float("-inf")).argmax(-1)

        js_values = []
        top3_values = []
        top5_values = []
        for agent in range(score.shape[0]):
            local = probability[agent]
            midpoint = 0.5 * (local[:, None, :] + local[None, :, :])
            log_midpoint = torch.log(midpoint.clamp_min(1e-12))
            js = 0.5 * (
                (local[:, None, :] * (log_probability[agent][:, None, :] -
                                      log_midpoint)).sum(-1) +
                (local[None, :, :] * (log_probability[agent][None, :, :] -
                                      log_midpoint)).sum(-1))
            js_values.append(_pairwise_values(js))
            top3 = torch.topk(masked[agent], 3, dim=-1).indices
            top5 = torch.topk(masked[agent], 5, dim=-1).indices
            top3_values.append(_set_jaccard(top3, score.shape[-1]))
            top5_values.append(_set_jaccard(top5, score.shape[-1]))

        result[f"round{round_index}"] = {
            "entropy": entropy.detach().cpu().flatten().tolist(),
            "max_probability": maximum_probability.detach().cpu().flatten().tolist(),
            "pairwise_js": torch.cat(js_values).detach().cpu().tolist(),
            "top3_jaccard": torch.cat(top3_values).detach().cpu().tolist(),
            "top5_jaccard": torch.cat(top5_values).detach().cpu().tolist(),
            "unique_argmax": [
                float(torch.unique(row).numel()) for row in argmax],
            "unique_sampled": [
                float(torch.unique(row).numel()) for row in selected],
            "unary_top1_unique": [1.0] * score.shape[0],
            "pair_energy_slot_variance": pair_variance.detach().cpu().tolist(),
            "unary_conditional_spearman": rank_correlation.detach().cpu().flatten().tolist(),
            "conditional_top1_changed_from_unary": (
                argmax != unary_top1[:, None]).float().detach().cpu().flatten().tolist(),
            "assigned_conditional_score": selected_score.detach().cpu().flatten().tolist(),
            "independent_map_score": argmax_score.detach().cpu().flatten().tolist(),
            "assignment_score_regret": (
                argmax_score - selected_score).detach().cpu().flatten().tolist(),
        }
    return result


def _merge_score_statistics(target, source):
    for round_name, metrics in source.items():
        for name, values in metrics.items():
            target[round_name][name].extend(values)


def _summarize_score_statistics(values):
    def score_summary(items):
        summary = _finite_summary(items)
        if items:
            value = np.asarray(items, dtype=np.float64)
            summary["p95"] = float(np.quantile(value, 0.95))
        return summary

    return {
        round_name: {
            name: score_summary(items) for name, items in metrics.items()
        } for round_name, metrics in values.items()
    }


def _scene_groups(scene_index):
    groups = []
    for value in torch.unique(scene_index.long(), sorted=True):
        groups.append(torch.nonzero(
            scene_index.long() == value, as_tuple=False).flatten())
    return groups


def _world_records(round_ids, scene_index, variant, seed, window_index):
    records = []
    for round_index, ids in enumerate(round_ids):
        for local_scene, agent_ids in enumerate(_scene_groups(scene_index)):
            world = ids[agent_ids].transpose(0, 1).contiguous()  # [P,N]
            unique_worlds = int(torch.unique(world, dim=0).shape[0])
            difference = (world[:, None, :] != world[None, :, :]).sum(-1)
            hamming = _pairwise_values(difference.float())
            records.append({
                "variant": variant,
                "seed": int(seed),
                "window": int(window_index),
                "scene": int(local_scene),
                "round": int(round_index),
                "num_agents": int(agent_ids.numel()),
                "unique_joint_worlds": unique_worlds,
                "duplicate_joint_worlds": int(world.shape[0] - unique_worlds),
                "mean_pairwise_hamming": float(hamming.mean().cpu()),
                "minimum_pairwise_hamming": float(hamming.min().cpu()),
                "identical_world_pair_fraction": float(
                    hamming.eq(0).float().mean().cpu()),
            })
    return records


def _transition_values(source, target):
    multiplicity = []
    for candidate in torch.unique(target):
        incoming = torch.unique(source[target == candidate]).numel()
        multiplicity.append(float(incoming))
    value = torch.tensor(multiplicity, dtype=torch.float32)
    return {
        "maximum_convergence_multiplicity": float(value.max()),
        "mean_convergence_multiplicity": float(value.mean()),
        "fraction_targets_ge2": float(value.ge(2).float().mean()),
        "fraction_targets_ge3": float(value.ge(3).float().mean()),
        "fraction_targets_ge5": float(value.ge(5).float().mean()),
    }


def _transition_records(round_ids, metadata):
    records = []
    pairs = []
    if len(round_ids) >= 2:
        pairs.append((0, 1))
    if len(round_ids) >= 3:
        pairs.extend(((1, 2), (0, 2)))
    for source_round, target_round in pairs:
        for agent in range(round_ids[0].shape[0]):
            records.append({
                **metadata[agent],
                "transition": f"round{source_round}->round{target_round}",
                **_transition_values(
                    round_ids[source_round][agent],
                    round_ids[target_round][agent]),
            })
    return records


def _agent_and_decomposition_records(
        net, inputs, output, auxiliary, metric_mask, trace, variant, seed,
        window_index):
    candidate = auxiliary["goal_candidates_world"].float()
    bank_score = auxiliary["candidate_log_prior"].float()
    unary_score = auxiliary["unary_score"].float()
    round_ids = [item["candidate_index"].long() for item in trace]
    num_agents, num_candidates = bank_score.shape
    gt_goal = _world_ground_truth(net, inputs)[-1]
    candidate_error = torch.linalg.vector_norm(
        candidate - gt_goal[:, None], dim=-1)
    bank_rank = _rank_tensor(bank_score)
    unary_rank = _rank_tensor(unary_score)
    bank_oracle, oracle_id = candidate_error.min(-1)
    edge_index = auxiliary["edge_index"].long()
    degree = _degrees(num_agents, edge_index)
    edge_class = "E>0" if edge_index.shape[1] else "E=0"

    prediction_world = _world_predictions(net, output, inputs)
    trajectory_error = torch.linalg.vector_norm(
        prediction_world[:, -1] - gt_goal[None], dim=-1).transpose(0, 1)
    trajectory_min = trajectory_error.min(-1).values

    agent_rows = []
    decomposition_rows = []
    metadata = []
    for agent in range(num_agents):
        metadata.append({
            "variant": variant,
            "seed": int(seed),
            "window": int(window_index),
            "agent": int(agent),
            "edge_class": edge_class,
            "agent_count_bin": _agent_count_bin(num_agents),
            "degree_bin": _degree_bin(int(degree[agent])),
        })
    for agent in torch.nonzero(metric_mask, as_tuple=False).flatten().tolist():
        local_oracles = []
        for round_index, ids in enumerate(round_ids):
            selected_error = candidate_error[agent].gather(0, ids[agent])
            selected_bank_rank = bank_rank[agent].gather(0, ids[agent])
            selected_unary_rank = unary_rank[agent].gather(0, ids[agent])
            unique_count = int(torch.unique(ids[agent]).numel())
            local_oracles.append(float(selected_error.min().cpu()))
            row = {
                **metadata[agent],
                "round": int(round_index),
                "unique_count": unique_count,
                "duplicate_count": int(ids[agent].numel() - unique_count),
                "unique_ratio": unique_count / float(ids[agent].numel()),
                "frozen_top1_coverage": bool((selected_bank_rank <= 1).any()),
                "frozen_top3_coverage": bool((selected_bank_rank <= 3).any()),
                "frozen_top5_coverage": bool((selected_bank_rank <= 5).any()),
                "unary_top1_coverage": bool((selected_unary_rank <= 1).any()),
                "unary_top3_coverage": bool((selected_unary_rank <= 3).any()),
                "unary_top5_coverage": bool((selected_unary_rank <= 5).any()),
                "goal_oracle_error": local_oracles[-1],
                "mean_frozen_rank": float(
                    selected_bank_rank.float().mean().cpu()),
                "mean_unary_rank": float(
                    selected_unary_rank.float().mean().cpu()),
                "gt_oracle_present": bool((ids[agent] == oracle_id[agent]).any()),
            }
            agent_rows.append(row)
        decomposition = {
            **metadata[agent],
            "bank_oracle": float(bank_oracle[agent].cpu()),
            "round0_oracle": local_oracles[0],
            "round1_oracle": local_oracles[1] if len(local_oracles) > 1 else None,
            "round2_oracle": local_oracles[2] if len(local_oracles) > 2 else None,
            "final_goal_oracle": local_oracles[-1],
            "trajectory_minFDE": float(trajectory_min[agent].cpu()),
            "round1_delta": (
                local_oracles[1] - local_oracles[0]
                if len(local_oracles) > 1 else None),
            "round2_delta": (
                local_oracles[2] - local_oracles[1]
                if len(local_oracles) > 2 else None),
            "diffusion_gap": float(trajectory_min[agent].cpu()) - local_oracles[-1],
        }
        decomposition_rows.append(decomposition)
    transition_rows = _transition_records(round_ids, metadata)
    world_rows = _world_records(
        round_ids, inputs["scene_index"], variant, seed, window_index)
    return agent_rows, decomposition_rows, transition_rows, world_rows


def _group_summary(rows, scalar_names, boolean_names=()):
    if not rows:
        return {"count": 0}
    result = {
        "count": len(rows),
        "scalars": {
            name: _finite_summary([
                row[name] for row in rows if row.get(name) is not None])
            for name in scalar_names
        },
    }
    for name in boolean_names:
        result[name] = float(np.mean([row[name] for row in rows]))
    return result


def _stratified(rows, summarizer):
    result = {"overall": summarizer(rows), "strata": {}}
    for field in ("edge_class", "degree_bin", "agent_count_bin"):
        groups = defaultdict(list)
        for row in rows:
            groups[row[field]].append(row)
        result["strata"][field] = {
            key: summarizer(value) for key, value in sorted(groups.items())}
    return result


def _summarize_round_records(rows):
    scalar = (
        "unique_count", "duplicate_count", "unique_ratio",
        "goal_oracle_error", "mean_frozen_rank", "mean_unary_rank")
    boolean = (
        "frozen_top1_coverage", "frozen_top3_coverage",
        "frozen_top5_coverage", "unary_top1_coverage",
        "unary_top3_coverage", "unary_top5_coverage", "gt_oracle_present")
    result = {}
    for round_index in sorted({row["round"] for row in rows}):
        local = [row for row in rows if row["round"] == round_index]
        result[f"round{round_index}"] = _stratified(
            local, lambda group: _group_summary(group, scalar, boolean))
    return result


def _summarize_decomposition(rows):
    scalar = (
        "bank_oracle", "round0_oracle", "round1_oracle", "round2_oracle",
        "final_goal_oracle", "trajectory_minFDE", "round1_delta",
        "round2_delta", "diffusion_gap")
    return _stratified(rows, lambda group: _group_summary(group, scalar))


def _summarize_transitions(rows):
    scalar = (
        "maximum_convergence_multiplicity", "mean_convergence_multiplicity",
        "fraction_targets_ge2", "fraction_targets_ge3",
        "fraction_targets_ge5")
    result = {}
    for name in sorted({row["transition"] for row in rows}):
        local = [row for row in rows if row["transition"] == name]
        result[name] = _stratified(
            local, lambda group: _group_summary(group, scalar))
    return result


def _summarize_worlds(rows):
    scalar = (
        "unique_joint_worlds", "duplicate_joint_worlds",
        "mean_pairwise_hamming", "minimum_pairwise_hamming",
        "identical_world_pair_fraction")
    result = {}
    for round_index in sorted({row["round"] for row in rows}):
        local = [row for row in rows if row["round"] == round_index]
        result[f"round{round_index}"] = _group_summary(local, scalar)
    return result


@torch.no_grad()
def evaluate_variant(evaluator, variant, reference_states):
    net = evaluator.net
    net.eval()
    metric_runs = []
    agent_rows = []
    decomposition_rows = []
    transition_rows = []
    world_rows = []
    score_values = defaultdict(lambda: defaultdict(list))
    window_counts = defaultdict(int)
    controller = RefinementAuditController(
        net.jdv2_sampler, variant, reference_states,
        use_cuda=evaluator.device.type == "cuda")
    for seed in SEEDS:
        print(f"VARIANT {variant} seed={seed}", flush=True)
        metric_values = defaultdict(lambda: defaultdict(list))
        with isolated_random_seed(
                seed, use_cuda=evaluator.device.type == "cuda"):
            with controller:
                for window_index, (batch_data, batch_id) in enumerate(
                        evaluator.data_loaders["valid"]):
                    inputs, seq_list = net.prepare_inputs(batch_data, batch_id)
                    metric_mask = compute_metric_mask(seq_list)
                    edge_count = int(
                        inputs["jdv2_cache"]["edge_index"].shape[1])
                    controller.begin_window(
                        seed, window_index, edge_count, evaluator.device)
                    with evaluator._autocast_context():
                        prediction, auxiliary = net.forward(inputs, if_test=True)
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
                    rows = _agent_and_decomposition_records(
                        net, inputs, prediction, auxiliary, metric_mask,
                        controller.trace, variant, seed, window_index)
                    agent_rows.extend(rows[0])
                    decomposition_rows.extend(rows[1])
                    transition_rows.extend(rows[2])
                    world_rows.extend(rows[3])
                    _merge_score_statistics(
                        score_values,
                        _score_statistics(
                            controller.trace, net.jdv2_sampler.temperature))
                    del inputs, prediction, auxiliary, seq_list
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
        "world_diversity_summary": _summarize_worlds(world_rows),
        "window_counts": {
            str(seed): {
                name: window_counts[(seed, name)] for name in ("E=0", "E>0")}
            for seed in SEEDS
        },
        "paired_rng_checks": dict(controller.boundary_checks),
    }


def _reference_weighted_values(reference):
    policy = reference["policies"]["weighted_without_replacement"]
    overall = policy["candidate_summary"]["overall"]
    return {
        "metric_runs": policy["metric_runs"],
        "initial_unique": overall["scalars"]["initial_unique_count"]["mean"],
        "final_unique": overall["scalars"]["final_unique_count"]["mean"],
        "initial_oracle": overall["scalars"]["initial_slot_oracle_error"]["mean"],
        "final_oracle": overall["scalars"]["final_selected_goal_oracle_error"]["mean"],
        "trajectory_minFDE": overall["scalars"]["trajectory_minFDE"]["mean"],
    }


def baseline_reproduction_gate(r2, reference, tolerance=1e-10):
    expected = _reference_weighted_values(reference)
    checks = {}
    expected_runs = {int(row["seed"]): row for row in expected["metric_runs"]}
    for row in r2["metric_runs"]:
        seed = int(row["seed"])
        for metric in AUDIT_METRICS:
            actual = row["overall"][metric]
            target = expected_runs[seed]["overall"][metric]
            checks[f"seed_{seed}_{metric}"] = {
                "actual": actual,
                "expected": target,
                "absolute_error": abs(actual - target),
            }
    round0 = r2["round_candidate_summary"]["round0"]["overall"]["scalars"]
    round2 = r2["round_candidate_summary"]["round2"]["overall"]["scalars"]
    decomposition = r2["error_decomposition"]["overall"]["scalars"]
    actual_values = {
        "initial_unique": round0["unique_count"]["mean"],
        "final_unique": round2["unique_count"]["mean"],
        "initial_oracle": round0["goal_oracle_error"]["mean"],
        "final_oracle": round2["goal_oracle_error"]["mean"],
        "trajectory_minFDE": decomposition["trajectory_minFDE"]["mean"],
    }
    for name, actual in actual_values.items():
        target = expected[name]
        checks[name] = {
            "actual": actual,
            "expected": target,
            "absolute_error": abs(actual - target),
        }
    return {
        "passed": all(item["absolute_error"] <= tolerance
                      for item in checks.values()),
        "tolerance": tolerance,
        "checks": checks,
    }


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sampler-reference", required=True)
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
    reference_path = Path(cli.sampler_reference).resolve()
    sampler_reference = json.loads(reference_path.read_text())
    if sampler_reference.get("status") != "AUDIT_COMPLETE":
        raise RuntimeError("sampler intervention reference is incomplete")

    result = {
        "status": "RUNNING_IID_RNG_CONTROL",
        "protocol": {
            "split": "valid",
            "windows": 139,
            "seeds": list(SEEDS),
            "candidates": 21,
            "samples": 20,
            "canonical_initialization": "weighted_without_replacement",
            "variants": list(VARIANTS),
            "assignment_status": "diagnostic intervention only",
            "rng_pairing": (
                "iid control captures Python/NumPy/CPU Torch/all-CUDA states "
                "at pre-initial, pre-round1, pre-round2, pre-diffusion and "
                "window-end boundaries; every diagnostic restores the exact "
                "available control boundary and uses the same diffusion state."),
        },
        "provenance": {
            "checkpoint": str(Path(cli.checkpoint).resolve()),
            "checkpoint_epoch": int(epoch),
            "checkpoint_sha256": file_sha256(cli.checkpoint),
            "architecture_config": architecture,
            "cache_manifest_hash": checkpoint.get("cache_manifest_hash"),
            "source_checkpoint_hash": checkpoint.get("source_checkpoint_hash"),
            "sampler_reference": str(reference_path),
            "sampler_reference_sha256": file_sha256(reference_path),
        },
        "policies": {},
    }
    _write_json(output, result)
    reference_states = {}
    result["policies"]["iid_control"] = evaluate_variant(
        evaluator, "iid_control", reference_states)
    result["status"] = "RUNNING_R2_BASELINE_GATE"
    _write_json(output, result)

    r2 = evaluate_variant(evaluator, "R2", reference_states)
    result["policies"]["R2"] = r2
    gate = baseline_reproduction_gate(r2, sampler_reference)
    result["baseline_reproduction_gate"] = gate
    if not gate["passed"]:
        result["status"] = "STOPPED_BASELINE_REPRODUCTION_FAILED"
        _write_json(output, result)
        raise RuntimeError("R2 failed weighted two-round reproduction gate")
    result["status"] = "R2_BASELINE_GATE_PASSED"
    _write_json(output, result)

    for variant in ("R0", "R1", "MAP", "ASSIGN"):
        result["policies"][variant] = evaluate_variant(
            evaluator, variant, reference_states)
        result["status"] = f"COMPLETED_{variant}"
        _write_json(output, result)

    expected_windows = len(SEEDS) * 139
    for variant in VARIANTS:
        checks = result["policies"][variant]["paired_rng_checks"]
        if checks.get("pre_diffusion") != expected_windows or \
                checks.get("window_end") != expected_windows:
            raise RuntimeError(f"incomplete diffusion RNG checks for {variant}")
        if ROUND_COUNT[variant] > 0:
            expected_refinement = len(SEEDS) * 92
            if checks.get("pre_round1") != expected_refinement:
                raise RuntimeError(f"incomplete round1 RNG checks for {variant}")
        if ROUND_COUNT[variant] > 1:
            expected_refinement = len(SEEDS) * 92
            if checks.get("pre_round2") != expected_refinement:
                raise RuntimeError(f"incomplete round2 RNG checks for {variant}")

    result["status"] = "AUDIT_COMPLETE"
    _write_json(output, result)
    print(f"WROTE {output}", flush=True)


if __name__ == "__main__":
    main()
