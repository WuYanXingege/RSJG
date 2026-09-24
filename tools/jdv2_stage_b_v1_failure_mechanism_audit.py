#!/usr/bin/env python3
"""Read-only causal audit of the JDV2 Stage-B V1 dependency corrector.

This module deliberately lives outside ``src``.  It replays the production
diffusion tree from an explicit noise tape and permits only diagnostic
transformations of the learned residual and relation input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import torch

from src.jdv2_audit import AUDIT_METRICS
from src.metrics import compute_metric_mask
from src.models.model import jdv2_select_scene_oracle_branch
from src.utils import isolated_random_seed
from tools.audit_jdv2_stage_a import _active_evaluator


MODES = (
    "stage_a", "full_v1", "common_only", "component_centered",
    "scene_centered", "oracle_only", "zero_relation", "mean_relation",
    "branch_shuffled_relation",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def distribution(values: Iterable[float]) -> dict:
    array = np.asarray(list(values), dtype=np.float64)
    if not array.size:
        return {"count": 0}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "p10": float(np.quantile(array, 0.10)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "max": float(array.max()),
    }


def rng_snapshot(use_cuda: bool) -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": (torch.cuda.get_rng_state_all()
                 if use_cuda and torch.cuda.is_available() else None),
    }


def rng_restore(state: Mapping) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


@dataclass
class NoiseTape:
    """Exact random tensors consumed by one production ``ts_sample`` call."""

    x_T: torch.Tensor
    branch_noise: tuple[tuple[torch.Tensor, ...], ...]


def make_noise_tape(net, context: torch.Tensor) -> NoiseTape:
    """Consume RNG exactly in production order and retain compact noise."""
    batch_size = context.size(0)
    # Production creates x_T on CPU and only then transfers it to the device.
    x_T = torch.randn([batch_size, net.args.pred_length, 2]).to(context.device)
    schedule = np.linspace(
        net.var_sched.num_steps, 0, net.args.ddim_step + 1)
    per_branch = []
    for _ in range(net.args.num_samples):
        noises = []
        for _step in range(
                int(net.args.branch_stage_step + 1),
                int(net.args.ddim_step + 1)):
            noises.append(torch.randn_like(x_T))
        per_branch.append(tuple(noises))
    expected = int(net.args.ddim_step - net.args.branch_stage_step)
    assert len(per_branch) == net.args.num_samples
    assert all(len(row) == expected for row in per_branch)
    return NoiseTape(x_T=x_T, branch_noise=tuple(per_branch))


def connected_components(num_agents: int, edge_index: torch.Tensor):
    """Return interacting components and degree; singleton degree-zero omitted."""
    degree = torch.zeros(num_agents, dtype=torch.long,
                         device=edge_index.device)
    adjacency = [[] for _ in range(num_agents)]
    if edge_index.numel():
        src, dst = edge_index.long()
        ones = torch.ones_like(src)
        degree.index_add_(0, src, ones)
        degree.index_add_(0, dst, ones)
        for left, right in zip(src.detach().cpu().tolist(),
                               dst.detach().cpu().tolist()):
            adjacency[left].append(right)
            adjacency[right].append(left)
    seen = set()
    components = []
    for root in range(num_agents):
        if root in seen or not adjacency[root]:
            continue
        stack = [root]
        seen.add(root)
        current = []
        while stack:
            node = stack.pop()
            current.append(node)
            for neighbor in adjacency[node]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        components.append(torch.as_tensor(
            sorted(current), dtype=torch.long, device=edge_index.device))
    return components, degree


def decompose_component_residual(
        delta: torch.Tensor, edge_index: torch.Tensor) -> dict[str, torch.Tensor]:
    """Orthogonally split residual into component-common and centered parts."""
    components, degree = connected_components(delta.shape[0], edge_index)
    common = torch.zeros_like(delta)
    for component in components:
        common[component] = delta[component].float().mean(
            dim=0, keepdim=True).to(delta.dtype)
    centered = torch.where(
        (degree > 0)[:, None, None], delta - common, torch.zeros_like(delta))
    return {"common": common, "centered": centered, "degree": degree,
            "components": components}


def scene_center_residual(
        delta: torch.Tensor, edge_index: torch.Tensor,
        scene_index: torch.Tensor) -> torch.Tensor:
    _, degree = connected_components(delta.shape[0], edge_index)
    output = torch.zeros_like(delta)
    for scene in torch.unique(scene_index, sorted=True):
        active = scene_index.eq(scene) & degree.gt(0)
        if active.any():
            mean = delta[active].float().mean(dim=0, keepdim=True)
            output[active] = delta[active] - mean.to(delta.dtype)
    return output


def edge_relative(delta: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    if not edge_index.shape[1]:
        return delta.new_empty((0,) + tuple(delta.shape[1:]))
    src, dst = edge_index.long()
    return delta[dst] - delta[src]


def relation_for_mode(relation: torch.Tensor, mode: str,
                      branch: int) -> torch.Tensor:
    """Return one [E,16] relation slice for a diagnostic relation mode."""
    if mode == "zero_relation":
        return torch.zeros_like(relation[:, branch])
    if mode == "mean_relation":
        return relation.float().mean(dim=1).to(relation.dtype)
    if mode == "branch_shuffled_relation":
        return relation[:, (branch + 1) % relation.shape[1]]
    return relation[:, branch]


def transform_delta(
        delta: torch.Tensor, mode: str, edge_index: torch.Tensor,
        scene_index: torch.Tensor, branch: int,
        oracle_agent_branch: torch.Tensor | None = None) -> torch.Tensor:
    """Apply one of the approved residual-only counterfactuals."""
    if mode == "stage_a":
        return torch.zeros_like(delta)
    if mode in {"full_v1", "zero_relation", "mean_relation",
                "branch_shuffled_relation"}:
        return delta
    parts = decompose_component_residual(delta, edge_index)
    if mode == "common_only":
        return parts["common"]
    if mode == "component_centered":
        return parts["centered"]
    if mode == "scene_centered":
        return scene_center_residual(delta, edge_index, scene_index)
    if mode == "oracle_only":
        if oracle_agent_branch is None:
            raise ValueError("oracle_only requires oracle_agent_branch")
        return torch.where(
            oracle_agent_branch.eq(int(branch))[:, None, None], delta,
            torch.zeros_like(delta))
    raise ValueError(f"Unknown audit mode: {mode}")


def _component_size_bin(size: int) -> str:
    if size == 2:
        return "2"
    if size <= 4:
        return "3-4"
    return ">=5"


class ResidualAccumulator:
    """Online scalar residual decomposition; never persists dense activations."""

    def __init__(self):
        self.values = defaultdict(list)
        self.identity_max = 0.0
        self.center_sum_max = 0.0
        self.edge_relative_max = 0.0

    def add(self, delta, edge_index, timestep: int, seed: int):
        parts = decompose_component_residual(delta, edge_index)
        common, centered = parts["common"], parts["centered"]
        reconstruction = (common + centered - delta).float().abs()
        self.identity_max = max(
            self.identity_max,
            float(reconstruction.max().cpu()) if reconstruction.numel() else 0)
        relative_difference = (
            edge_relative(centered, edge_index).float() -
            edge_relative(delta, edge_index).float()).abs()
        self.edge_relative_max = max(
            self.edge_relative_max,
            float(relative_difference.max().cpu())
            if relative_difference.numel() else 0)
        for component in parts["components"]:
            local = delta[component].float()
            local_common = common[component].float()
            local_centered = centered[component].float()
            centered_sum = local_centered.sum(dim=0).abs()
            self.center_sum_max = max(
                self.center_sum_max, float(centered_sum.max().cpu()))
            total = float(local.square().sum().cpu())
            common_energy = float(local_common.square().sum().cpu())
            centered_energy = float(local_centered.square().sum().cpu())
            denominator = max(total, 1e-20)
            row = {
                "total_energy": total,
                "common_energy": common_energy,
                "centered_energy": centered_energy,
                "common_fraction": common_energy / denominator,
                "centered_fraction": centered_energy / denominator,
            }
            strata = (
                "overall", f"timestep={timestep}",
                f"component_size={_component_size_bin(component.numel())}",
                f"scene_agents={delta.shape[0]}", f"seed={seed}")
            for stratum in strata:
                for key, value in row.items():
                    self.values[(stratum, key)].append(value)

    def summary(self):
        strata = sorted({key[0] for key in self.values})
        return {
            "invariants": {
                "reconstruction_max_abs": self.identity_max,
                "centered_component_sum_max_abs": self.center_sum_max,
                "edge_relative_difference_max_abs": self.edge_relative_max,
            },
            "strata": {
                stratum: {
                    name: distribution(self.values[(stratum, name)])
                    for name in (
                        "total_energy", "common_energy", "centered_energy",
                        "common_fraction", "centered_fraction")
                }
                for stratum in strata
            },
        }


@torch.no_grad()
def trunk_from_tape(net, all_context, tape: NoiseTape) -> torch.Tensor:
    context = all_context[-1]
    x_t = tape.x_T
    for timestep in range(
            net.var_sched.num_steps, net.args.trunk_stage_step, -1):
        alpha = net.var_sched.alphas[timestep]
        alpha_bar = net.var_sched.alpha_bars[timestep]
        c0 = 1.0 / torch.sqrt(alpha)
        c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)
        beta = net.var_sched.betas[[timestep] * context.size(0)]
        epsilon = net.diffnet(x_t, beta=beta, context=context)
        x_t = c0 * (x_t - c1 * epsilon)
    return x_t


@torch.no_grad()
def replay_branches(
        net, all_context, dependency_state, tape: NoiseTape, mode: str,
        scene_index: torch.Tensor, oracle_agent_branch: torch.Tensor | None,
        seed: int, residual_accumulator: ResidualAccumulator | None = None,
        middle_result: torch.Tensor | None = None,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Exact local replay of production branch diffusion from ``NoiseTape``."""
    if mode not in MODES:
        raise ValueError(mode)
    if middle_result is None:
        middle_result = trunk_from_tape(net, all_context, tape)
    simple_var = net.args.dataset in ["eth5", "ind"]
    eta = 1 if simple_var else 0
    schedule = np.linspace(
        net.var_sched.num_steps, 0, net.args.ddim_step + 1)
    outputs = []
    residuals = []
    edge_index = dependency_state["edge_index"]
    relation = dependency_state["relation_embedding"]
    for branch in range(net.args.num_samples):
        context = all_context[branch]
        x_t = middle_result
        for noise_index, step in enumerate(range(
                int(net.args.branch_stage_step + 1),
                int(net.args.ddim_step + 1))):
            cur_t = int(schedule[step - 1])
            prev_t = int(schedule[step])
            ab_cur = net.var_sched.alpha_bars[cur_t]
            ab_prev = (net.var_sched.alpha_bars[prev_t]
                       if prev_t >= 0 else 1)
            beta = net.var_sched.betas[[cur_t] * context.size(0)]
            epsilon = net.diffnet(x_t, beta=beta, context=context)
            if mode != "stage_a":
                last_map = dependency_state["last_position_map"]
                position_map = last_map[:, None] + torch.cumsum(x_t, dim=1)
                position_world = dependency_state[
                    "scene"].make_world_coord_torch(
                        position_map * float(net.args.down_factor))
                anchor = dependency_state["last_position_world"][:, None]
                noisy_world_velocity = torch.diff(
                    torch.cat((anchor.float(), position_world.float()), dim=1),
                    dim=1) / float(net.args.trajectory_dt)
                relation_slice = relation_for_mode(relation, mode, branch)
                raw_delta = net.jdv2_corrector(
                    noisy_world_velocity,
                    dependency_state["last_position_world"], edge_index,
                    relation_slice, torch.tensor(cur_t, device=x_t.device),
                    edge_weight=dependency_state["edge_weight"])
                if residual_accumulator is not None and mode == "full_v1":
                    residual_accumulator.add(
                        raw_delta, edge_index, cur_t, seed)
                delta = transform_delta(
                    raw_delta, mode, edge_index, scene_index, branch,
                    oracle_agent_branch)
                epsilon = epsilon + delta
                residuals.append(delta.detach())
            variance = eta * (1 - ab_prev) / (1 - ab_cur) * \
                (1 - ab_cur / ab_prev)
            first = (ab_prev / ab_cur) ** 0.5 * x_t
            second = ((1 - ab_prev - variance) ** 0.5 -
                      (ab_prev * (1 - ab_cur) / ab_cur) ** 0.5) * epsilon
            if simple_var:
                third = (1 - ab_cur / ab_prev) ** 0.5 * \
                    tape.branch_noise[branch][noise_index]
            else:
                third = variance ** 0.5 * tape.branch_noise[branch][noise_index]
            x_t = first + second + third
        outputs.append(x_t)
    return torch.stack(outputs), residuals


def velocities_to_predictions(net, inputs, velocity: torch.Tensor):
    x = inputs["x_augmented"].detach()
    num_agents = x.shape[1]
    prediction = torch.zeros(
        [net.args.num_samples, net.args.seq_length, num_agents, 2],
        device=velocity.device, dtype=velocity.dtype)
    prediction[:, :net.args.obs_length] = x[
        :net.args.obs_length, :, 6:8].repeat(net.args.num_samples, 1, 1, 1)
    for step in range(net.args.obs_length, net.args.seq_length):
        prediction[:, step] = prediction[:, step - 1] + \
            velocity[:, :, step - net.args.obs_length]
    return prediction


def _metric_append(store, net, prediction, auxiliary, inputs, mask, stratum):
    for metric in AUDIT_METRICS:
        values = net.compute_model_metrics(
            metric_name=metric, predictions=prediction, metric_mask=mask,
            all_aux_outputs=auxiliary, inputs=inputs,
            obs_length=net.args.obs_length)
        store[stratum][metric].extend(float(value) for value in values)
        store["overall"][metric].extend(float(value) for value in values)


def _metric_finalize(store):
    return {
        stratum: {metric: float(np.mean(values))
                  for metric, values in metrics.items()}
        for stratum, metrics in store.items()
    }


def _spearman(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.size < 2 or np.all(left == left[0]) or np.all(right == right[0]):
        return None
    # Average-rank ties through scipy when available; fallback has no ties in
    # the principal rank audit and is kept for minimal environments.
    try:
        from scipy.stats import spearmanr
        return float(spearmanr(left, right).statistic)
    except ImportError:
        return float(np.corrcoef(
            np.argsort(np.argsort(left)), np.argsort(np.argsort(right)))[0, 1])


def _branch_rows(net, inputs, stage_a, full, auxiliary):
    """Return compact per-scene/per-branch distribution-shift records."""
    scene_index = inputs["scene_index"].long()
    goals = auxiliary["joint_goal_points_world"][:, :net.args.num_samples]
    target_goal = inputs["world_coord"][-1]
    stage_a_world = []
    full_world = []
    for branch in range(net.args.num_samples):
        stage_a_world.append(inputs["scene"].make_world_coord_torch(
            stage_a[branch] * net.args.down_factor))
        full_world.append(inputs["scene"].make_world_coord_torch(
            full[branch] * net.args.down_factor))
    stage_a_world = torch.stack(stage_a_world)[:, net.args.obs_length:]
    full_world = torch.stack(full_world)[:, net.args.obs_length:]
    target = inputs["world_coord"][net.args.obs_length:].permute(1, 0, 2)
    edge_index = auxiliary["edge_index"].long()
    rows = []
    for scene in torch.unique(scene_index, sorted=True):
        agents = torch.nonzero(scene_index.eq(scene), as_tuple=False).flatten()
        goal_error = (goals[agents] - target_goal[agents, None]).square().sum(
            dim=-1).mean(dim=0)
        ade_a = torch.linalg.vector_norm(
            stage_a_world[:, :, agents] -
            target[agents].permute(1, 0, 2)[None], dim=-1
        ).mean(dim=(1, 2))
        fde_a = torch.linalg.vector_norm(
            stage_a_world[:, -1, agents] - target[agents, -1][None], dim=-1
        ).mean(dim=1)
        ade_b = torch.linalg.vector_norm(
            full_world[:, :, agents] -
            target[agents].permute(1, 0, 2)[None], dim=-1
        ).mean(dim=(1, 2))
        fde_b = torch.linalg.vector_norm(
            full_world[:, -1, agents] - target[agents, -1][None], dim=-1
        ).mean(dim=1)
        goal_order = torch.argsort(goal_error)
        goal_rank = torch.empty_like(goal_order)
        goal_rank[goal_order] = torch.arange(20, device=goal_order.device) + 1
        trajectory_order = torch.argsort(ade_a)
        trajectory_rank = torch.empty_like(trajectory_order)
        trajectory_rank[trajectory_order] = \
            torch.arange(20, device=trajectory_order.device) + 1
        scene_edges = (scene_index[edge_index[0]].eq(scene)
                       if edge_index.shape[1] else torch.zeros(
                           0, dtype=torch.bool, device=scene_index.device))
        for branch in range(20):
            relative_a = relative_b = 0.0
            if scene_edges.any():
                src, dst = edge_index[:, scene_edges]
                target_relative = target[dst] - target[src]
                relative_a = float(torch.linalg.vector_norm(
                    (stage_a_world[branch, :, dst] - stage_a_world[branch, :, src]) -
                    target_relative.permute(1, 0, 2), dim=-1).mean().cpu())
                relative_b = float(torch.linalg.vector_norm(
                    (full_world[branch, :, dst] - full_world[branch, :, src]) -
                    target_relative.permute(1, 0, 2), dim=-1).mean().cpu())
            rows.append({
                "goal_error": float(goal_error[branch].cpu()),
                "goal_rank": int(goal_rank[branch].cpu()),
                "trajectory_rank": int(trajectory_rank[branch].cpu()),
                "ade_stage_a": float(ade_a[branch].cpu()),
                "fde_stage_a": float(fde_a[branch].cpu()),
                "delta_ade": float((ade_b[branch] - ade_a[branch]).cpu()),
                "delta_fde": float((fde_b[branch] - fde_a[branch]).cpu()),
                "delta_relative": relative_b - relative_a,
                "goal_oracle": branch == int(goal_error.argmin().cpu()),
                "trajectory_oracle": branch == int(ade_a.argmin().cpu()),
            })
    return rows


def summarize_branch_rows(rows):
    def rank_group(rank):
        if rank == 1:
            return "1"
        if rank <= 5:
            return "2-5"
        if rank <= 10:
            return "6-10"
        return "11-20"

    output = {}
    for rank_name in ("goal_rank", "trajectory_rank"):
        output[rank_name] = {}
        for group in ("1", "2-5", "6-10", "11-20"):
            selected = [row for row in rows
                        if rank_group(row[rank_name]) == group]
            output[rank_name][group] = {
                key: distribution(row[key] for row in selected)
                for key in ("delta_ade", "delta_fde", "delta_relative")}
    goal_oracle = [row for row in rows if row["goal_oracle"]]
    return {
        "grouped": output,
        "scene_count": len(goal_oracle),
        "goal_oracle_equals_trajectory_oracle_rate": float(np.mean([
            row["trajectory_oracle"] for row in goal_oracle])),
        "trajectory_oracle_goal_rank": distribution(
            row["goal_rank"] for row in rows if row["trajectory_oracle"]),
        "spearman": {
            "goal_error_vs_delta_ade": _spearman(
                [row["goal_error"] for row in rows],
                [row["delta_ade"] for row in rows]),
            "goal_error_vs_delta_fde": _spearman(
                [row["goal_error"] for row in rows],
                [row["delta_fde"] for row in rows]),
            "goal_rank_vs_delta_ade": _spearman(
                [row["goal_rank"] for row in rows],
                [row["delta_ade"] for row in rows]),
            "goal_rank_vs_delta_fde": _spearman(
                [row["goal_rank"] for row in rows],
                [row["delta_fde"] for row in rows]),
        },
    }


def _residual_difference(reference, alternate):
    if len(reference) != len(alternate):
        raise AssertionError("residual trace lengths differ")
    if not reference:
        return {"max_abs": 0.0, "mean_abs": 0.0,
                "rms_difference": 0.0}
    maxima, sums, squares, counts = [], 0.0, 0.0, 0
    for left, right in zip(reference, alternate):
        difference = (left.float() - right.float()).abs()
        maxima.append(float(difference.max().cpu()))
        sums += float(difference.sum().cpu())
        squares += float(difference.square().sum().cpu())
        counts += difference.numel()
    return {"max_abs": max(maxima), "mean_abs": sums / max(counts, 1),
            "rms_difference": math.sqrt(squares / max(counts, 1))}


@torch.no_grad()
def run_counterfactual_audit(evaluator, seeds: Sequence[int], parity_seed: int):
    net = evaluator.net
    net.eval()
    metrics = {mode: [] for mode in MODES}
    residual = ResidualAccumulator()
    parity = {
        "seed": parity_seed, "windows": 0, "full_max_abs": 0.0,
        "none_max_abs": 0.0, "candidate_ids_equal": True,
        "goals_equal": True,
    }
    coverage = []
    branch_rows = []
    relation_sensitivity = defaultdict(list)
    start = time.perf_counter()
    for seed in seeds:
        print(f"FAILURE_AUDIT inference seed={seed}", flush=True)
        per_mode = {
            mode: defaultdict(lambda: defaultdict(list)) for mode in MODES}
        with isolated_random_seed(
                int(seed), use_cuda=evaluator.device.type == "cuda"):
            for window_index, (batch_data, batch_id) in enumerate(
                    evaluator.data_loaders["valid"]):
                inputs, sequence = net.prepare_inputs(batch_data, batch_id)
                mask = compute_metric_mask(sequence)
                net.jdv2_sampler.set_sampling_context(seed, window_index)
                with evaluator._autocast_context():
                    contexts, auxiliary = net.encode(inputs, if_test=True)
                dependency = auxiliary["dependency_state"]
                ids = auxiliary["joint_candidate_index"][:, :20]
                goals = auxiliary["joint_goal_points_world"][:, :20]
                coverage.extend(torch.unique(row).numel() for row in ids)
                oracle = jdv2_select_scene_oracle_branch(
                    goals, inputs["world_coord"][-1], inputs["scene_index"],
                    dependency["edge_index"], dependency["relation_embedding"])

                pre_noise = rng_snapshot(evaluator.device.type == "cuda")
                tape = make_noise_tape(net, contexts[-1])
                post_noise = rng_snapshot(evaluator.device.type == "cuda")
                with evaluator._autocast_context():
                    middle = trunk_from_tape(net, contexts, tape)
                    velocities = {}
                    traces = {}
                    for mode in MODES:
                        velocities[mode], traces[mode] = replay_branches(
                            net, contexts, dependency, tape, mode,
                            inputs["scene_index"], oracle["agent_branch"], seed,
                            residual_accumulator=(residual if mode == "full_v1"
                                                  else None),
                            middle_result=middle)
                rng_restore(post_noise)

                predictions = {
                    mode: velocities_to_predictions(net, inputs, value)
                    for mode, value in velocities.items()}
                edge_class = ("E>0" if dependency["edge_index"].shape[1]
                              else "E=0")
                for mode in MODES:
                    _metric_append(per_mode[mode], net, predictions[mode],
                                   auxiliary, inputs, mask, edge_class)
                    assert torch.equal(ids, auxiliary[
                        "joint_candidate_index"][:, :20])
                    assert torch.equal(goals, auxiliary[
                        "joint_goal_points_world"][:, :20])
                    if edge_class == "E=0":
                        if not torch.equal(predictions[mode],
                                           predictions["stage_a"]):
                            raise AssertionError(
                                f"E=0 {mode} differs from Stage A")

                branch_rows.extend(_branch_rows(
                    net, inputs, predictions["stage_a"],
                    predictions["full_v1"], auxiliary))
                for relation_mode in (
                        "zero_relation", "mean_relation",
                        "branch_shuffled_relation"):
                    difference = (predictions[relation_mode].float() -
                                  predictions["full_v1"].float()).abs()
                    relation_sensitivity[
                        relation_mode + ".trajectory_max_abs"].append(
                            float(difference.max().cpu()))
                    relation_sensitivity[
                        relation_mode + ".trajectory_mean_abs"].append(
                            float(difference.mean().cpu()))
                    for key, value in _residual_difference(
                            traces["full_v1"], traces[relation_mode]).items():
                        relation_sensitivity[
                            relation_mode + ".residual_" + key].append(value)

                if seed == parity_seed:
                    rng_restore(pre_noise)
                    with evaluator._autocast_context():
                        production_full = net.ts_sample(
                            contexts, dependency_state=dependency)
                    post_production = rng_snapshot(
                        evaluator.device.type == "cuda")
                    parity["full_max_abs"] = max(
                        parity["full_max_abs"], float((
                            production_full.float() -
                            velocities["full_v1"].float()).abs().max().cpu()))
                    old_flag = net.args.use_dependency_corrector
                    try:
                        net.args.use_dependency_corrector = False
                        rng_restore(pre_noise)
                        with evaluator._autocast_context():
                            production_none = net.ts_sample(
                                contexts, dependency_state=dependency)
                    finally:
                        net.args.use_dependency_corrector = old_flag
                    parity["none_max_abs"] = max(
                        parity["none_max_abs"], float((
                            production_none.float() -
                            velocities["stage_a"].float()).abs().max().cpu()))
                    parity["windows"] += 1
                    rng_restore(post_production)
                    if parity["full_max_abs"] != 0.0 or \
                            parity["none_max_abs"] != 0.0:
                        raise RuntimeError("AUDIT_IMPLEMENTATION_BLOCKED: "
                                           "ts_sample parity failed")
                del inputs, sequence, contexts, auxiliary, predictions
                del velocities, traces, tape, middle, mask
        for mode in MODES:
            metrics[mode].append({"seed": int(seed),
                                  **_metric_finalize(per_mode[mode])})
    return {
        "parity": parity,
        "coverage": {
            "count": len(coverage), "mean_unique": float(np.mean(coverage)),
            "minimum_unique": int(min(coverage)),
            "maximum_unique": int(max(coverage)),
            "coverage_20_of_20_rate": float(np.mean(
                np.asarray(coverage) == 20)),
        },
        "metric_runs": metrics,
        "metric_summary": {
            mode: {
                stratum: {
                    metric: distribution(
                        run[stratum][metric] for run in metrics[mode]
                        if stratum in run)
                    for metric in AUDIT_METRICS}
                for stratum in ("overall", "E=0", "E>0")}
            for mode in MODES
        },
        "residual_decomposition": residual.summary(),
        "branch_distribution": summarize_branch_rows(branch_rows),
        "relation_sensitivity": {
            key: distribution(values)
            for key, values in relation_sensitivity.items()},
        "runtime_seconds": float(time.perf_counter() - start),
    }


def _flatten_gradients(gradients, parameters):
    chunks = []
    for gradient, parameter in zip(gradients, parameters):
        chunks.append((torch.zeros_like(parameter) if gradient is None
                       else gradient).detach().float().reshape(-1).cpu())
    return torch.cat(chunks) if chunks else torch.empty(0)


def gradient_comparison(diff, relative, weight: float = 0.05):
    weighted = relative * float(weight)
    diff_norm = float(torch.linalg.vector_norm(diff))
    rel_norm = float(torch.linalg.vector_norm(relative))
    weighted_norm = float(torch.linalg.vector_norm(weighted))
    denominator = max(diff_norm * rel_norm, 1e-30)
    cosine = float(torch.dot(diff, relative) / denominator)
    return {
        "diff_norm": diff_norm, "relative_norm": rel_norm,
        "weighted_relative_norm": weighted_norm,
        "weighted_to_diff_ratio": weighted_norm / max(diff_norm, 1e-30),
        "cosine_raw": cosine, "cosine_weighted": cosine,
        "dot_raw": float(torch.dot(diff, relative)),
        "conflict": bool(cosine < 0),
    }


def _gradient_summary(rows):
    names = [key for key in rows[0] if key not in {"conflict"}]
    result = {name: distribution(row[name] for row in rows) for name in names}
    result["conflict_rate"] = float(np.mean([row["conflict"] for row in rows]))
    return result


def run_gradient_audit(evaluator, state_name: str, window_limit: int = 64):
    """Measure separate loss gradients without optimizer construction/steps."""
    net = evaluator.net
    net.train(True)
    parameters = [(name, parameter) for name, parameter in net.named_parameters()
                  if name.startswith("jdv2_corrector.")]
    tensors = [parameter for _, parameter in parameters]
    before = {name: parameter.detach().cpu().clone()
              for name, parameter in parameters}
    rows = []
    module_rows = defaultdict(list)
    with isolated_random_seed(7102035, use_cuda=evaluator.device.type == "cuda"):
        for batch_index, (batch_data, batch_id) in enumerate(
                evaluator.data_loaders["train"]):
            inputs, sequence = net.prepare_inputs(batch_data, batch_id)
            if not inputs["jdv2_cache"]["edge_index"].shape[1]:
                continue
            net.set_jdv2_training_sampling_context(
                2035, 1, batch_index, len(evaluator.data_loaders["train"]))
            with evaluator._autocast_context():
                losses = net._jdv2_dependency_losses(inputs)
            grad_diff = torch.autograd.grad(
                losses["jdv2_diffusion_loss"], tensors,
                retain_graph=True, allow_unused=True)
            grad_relative = torch.autograd.grad(
                losses["jdv2_relative_loss"], tensors,
                allow_unused=True)
            flat_diff = _flatten_gradients(grad_diff, tensors)
            flat_relative = _flatten_gradients(grad_relative, tensors)
            rows.append(gradient_comparison(flat_diff, flat_relative))
            for prefix in (
                    "state_encoder", "timestep_network", "gate", "value",
                    "output"):
                indices = [index for index, (name, _) in enumerate(parameters)
                           if name.startswith("jdv2_corrector." + prefix)]
                local_diff = _flatten_gradients(
                    [grad_diff[index] for index in indices],
                    [tensors[index] for index in indices])
                local_relative = _flatten_gradients(
                    [grad_relative[index] for index in indices],
                    [tensors[index] for index in indices])
                module_rows[prefix].append(
                    gradient_comparison(local_diff, local_relative))
            del inputs, sequence, losses, grad_diff, grad_relative
            if len(rows) >= window_limit:
                break
    if len(rows) < window_limit:
        raise RuntimeError(
            f"Only {len(rows)} E>0 training windows available; need {window_limit}")
    for name, parameter in parameters:
        if not torch.equal(parameter.detach().cpu(), before[name]):
            raise AssertionError("gradient audit changed corrector parameter")
    net.eval()
    return {
        "state": state_name, "e_gt0_windows": len(rows),
        "global": _gradient_summary(rows),
        "per_module": {name: _gradient_summary(values)
                       for name, values in module_rows.items()},
        "no_parameter_update": True,
        "parameter_names_all_corrector": all(
            name.startswith("jdv2_corrector.") for name, _ in parameters),
    }


def checkpoint_contract(stage_a: Path, stage_b: Path):
    return {
        "stage_a_path": str(stage_a.resolve()),
        "stage_a_sha256": sha256(stage_a),
        "stage_b_path": str(stage_b.resolve()),
        "stage_b_sha256": sha256(stage_b),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage-a-checkpoint", required=True)
    parser.add_argument("--stage-b-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[2035, 2036, 2037, 2038, 2039])
    parser.add_argument("--gradient-windows", type=int, default=64)
    return parser.parse_args()


def main():
    args = parse_args()
    stage_a = Path(args.stage_a_checkpoint)
    stage_b = Path(args.stage_b_checkpoint)
    output = Path(args.output)
    expected_a = "699336d49aaecbccfaac8b44f00c0df5a73b521548fc29d3da37d071148fd5bb"
    expected_b = "e24a1cbfc5760db57bcb5cacd64c66a4afd1aed0f11bb1efefa8549933d21c1a"
    contract = checkpoint_contract(stage_a, stage_b)
    if contract["stage_a_sha256"] != expected_a or \
            contract["stage_b_sha256"] != expected_b:
        raise RuntimeError("immutable checkpoint hash mismatch")
    evaluator, epoch = _active_evaluator(
        args.config, str(stage_b), output.parent / "runtime_best", args.device)
    result = {
        "status": "RUNNING",
        "protocol": {
            "split": "ETH validation", "windows": 139,
            "seeds": args.seeds, "paired_noise_tape": True,
            "checkpoint_epoch": int(epoch), "read_only": True,
            "counterfactual_modes": list(MODES),
        },
        "provenance": {
            **contract,
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True).strip(),
        },
    }

    def persist():
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    persist()
    result["paired_counterfactual"] = run_counterfactual_audit(
        evaluator, args.seeds, parity_seed=args.seeds[0])
    result["parity_gate_passed"] = True
    persist()
    result["gradient_audit"] = {
        "BEST": run_gradient_audit(
            evaluator, "BEST", window_limit=args.gradient_windows)}
    persist()
    init_evaluator, init_epoch = _active_evaluator(
        args.config, str(stage_a), output.parent / "runtime_init", args.device)
    result["gradient_audit"]["INIT"] = run_gradient_audit(
        init_evaluator, "INIT", window_limit=args.gradient_windows)
    result["gradient_audit"]["INIT"]["checkpoint_epoch"] = int(init_epoch)
    result["status"] = "COMPLETE"
    persist()
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
