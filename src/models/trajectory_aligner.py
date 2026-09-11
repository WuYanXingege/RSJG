"""Relation-aware, marginal-preserving alignment of trajectory samples.

The base predictor produces ``K`` trajectories for every agent.  Joint
metrics, however, require sample column ``k`` to describe one coherent scene.
This module learns only how to *permute* each agent's existing ``K``
trajectories.  Consequently the unordered per-agent sample set is unchanged
at inference and minADE/minFDE are preserved exactly.

Training uses Sinkhorn matrices as a differentiable relaxation of a
permutation.  Evaluation projects them to hard bijections and includes an
identity/keep path for uncertain components.  No Cartesian ``K ** N`` table is
ever materialized.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from src.models.interaction_graph import (
    EDGE_FEATURE_DIM,
    REVERSE_SIGN_FEATURE_INDICES,
)
from src.models.joint_goal import canonicalize_scene_index


def _identity_assignments(
        num_agents: int, num_samples: int, reference: torch.Tensor,
) -> List[torch.Tensor]:
    identity = torch.eye(
        num_samples, dtype=reference.dtype, device=reference.device)
    return [identity for _ in range(num_agents)]


def _connected_components(
        agents: Sequence[int], edge_index: torch.Tensor,
) -> List[List[int]]:
    """Return deterministic components of an already scene-local graph."""
    agent_set = set(int(index) for index in agents)
    adjacency = {index: [] for index in agent_set}
    for source, target in edge_index.detach().t().cpu().tolist():
        source, target = int(source), int(target)
        if source in agent_set and target in agent_set:
            adjacency[source].append(target)
            adjacency[target].append(source)

    components: List[List[int]] = []
    unseen = set(agent_set)
    while unseen:
        root = min(unseen)
        stack = [root]
        unseen.remove(root)
        component = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbour in sorted(adjacency[current], reverse=True):
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    stack.append(neighbour)
        components.append(sorted(component))
    return components


def _greedy_bijection(probability: torch.Tensor) -> torch.Tensor:
    """Project a square soft assignment to a deterministic hard bijection."""
    if probability.ndim != 2 or probability.shape[0] != probability.shape[1]:
        raise ValueError("probability must be a square matrix")
    size = probability.shape[0]
    remaining_rows = list(range(size))
    remaining_columns = list(range(size))
    permutation = torch.empty(
        size, dtype=torch.long, device=probability.device)
    detached = probability.detach()
    while remaining_rows:
        rows = torch.as_tensor(
            remaining_rows, dtype=torch.long, device=probability.device)
        columns = torch.as_tensor(
            remaining_columns, dtype=torch.long, device=probability.device)
        submatrix = detached.index_select(0, rows).index_select(1, columns)
        flat_index = int(submatrix.reshape(-1).argmax().item())
        local_row = flat_index // len(remaining_columns)
        local_column = flat_index % len(remaining_columns)
        row = remaining_rows.pop(local_row)
        column = remaining_columns.pop(local_column)
        permutation[row] = column
    return permutation


class RelationAwareTrajectoryAligner(nn.Module):
    """Assemble coherent scene samples without changing marginal sample sets.

    ``trajectories`` use ``[N,K,T,2]`` world-coordinate layout.  Pair scores
    combine complete predicted paths, observed-agent embeddings, geometric
    graph features and inferred relation probabilities.  The adaptive edge
    gate is soft while training; optional top-k pruning is used only for hard
    inference, avoiding the detached-gate optimization gap of the earlier
    graph implementation.
    """

    def __init__(
            self,
            agent_dim: int,
            edge_dim: int = EDGE_FEATURE_DIM,
            relation_dim: int = 0,
            hidden_dim: int = 128,
            sinkhorn_iterations: int = 8,
            alignment_iterations: int = 2,
            temperature: float = 0.15,
            keep_threshold: float = 0.60,
            top_k_edges: int = 4,
    ) -> None:
        super().__init__()
        if min(agent_dim, hidden_dim) <= 0 or edge_dim < 0 or relation_dim < 0:
            raise ValueError("Invalid trajectory-aligner feature dimensions")
        if sinkhorn_iterations <= 0 or alignment_iterations <= 0:
            raise ValueError("Sinkhorn/alignment iterations must be positive")
        if temperature <= 0:
            raise ValueError("Alignment temperature must be positive")
        if not 0 <= keep_threshold <= 1:
            raise ValueError("keep_threshold must lie in [0,1]")
        if top_k_edges < 0:
            raise ValueError("top_k_edges cannot be negative")

        self.agent_dim = int(agent_dim)
        self.edge_dim = int(edge_dim)
        self.relation_dim = int(relation_dim)
        self.hidden_dim = int(hidden_dim)
        self.sinkhorn_iterations = int(sinkhorn_iterations)
        self.alignment_iterations = int(alignment_iterations)
        self.temperature = float(temperature)
        self.keep_threshold = float(keep_threshold)
        self.top_k_edges = int(top_k_edges)

        self.step_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.trajectory_encoder = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.agent_encoder = nn.Sequential(
            nn.Linear(agent_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )

        edge_context_dim = 2 * hidden_dim + edge_dim + relation_dim
        self.edge_gate = nn.Sequential(
            nn.Linear(edge_context_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Pair geometry: final/minimum distance, absolute formation change
        # (x,y), and cosine similarity of net displacements.
        pair_input_dim = 2 * hidden_dim + edge_dim + relation_dim + 5
        self.pair_scorer = nn.Sequential(
            nn.Linear(pair_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Decide whether a proposed reassignment is trustworthy.  A negative
        # initialization makes a fresh checkpoint choose exact identity.
        self.keep_head = nn.Sequential(
            nn.Linear(5, hidden_dim // 2 if hidden_dim >= 2 else 1),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2 if hidden_dim >= 2 else 1, 1),
        )
        nn.init.normal_(self.pair_scorer[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.pair_scorer[-1].bias)
        nn.init.zeros_(self.keep_head[-1].weight)
        nn.init.constant_(self.keep_head[-1].bias, -2.2)

    def _validate(
            self,
            trajectories: torch.Tensor,
            last_pos: torch.Tensor,
            agent_feat: torch.Tensor,
            edge_index: torch.Tensor,
            edge_feat: torch.Tensor,
            edge_weight: torch.Tensor,
            relation_prob: Optional[torch.Tensor],
            scene_index: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if trajectories.ndim != 4 or trajectories.shape[-1] != 2:
            raise ValueError("trajectories must have shape [N,K,T,2]")
        num_agents, num_samples, num_steps, _ = trajectories.shape
        if num_agents <= 0 or num_samples <= 0 or num_steps <= 0:
            raise ValueError("trajectories must be non-empty")
        if last_pos.shape != (num_agents, 2):
            raise ValueError("last_pos must have shape [N,2]")
        if agent_feat.shape != (num_agents, self.agent_dim):
            raise ValueError("agent_feat must have shape [N,agent_dim]")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2,E]")
        edge_count = edge_index.shape[1]
        if edge_feat.shape != (edge_count, self.edge_dim):
            raise ValueError("edge_feat must have shape [E,edge_dim]")
        if edge_weight.shape != (edge_count,):
            raise ValueError("edge_weight must have shape [E]")
        if edge_count:
            source, target = edge_index
            if ((edge_index < 0).any() or (edge_index >= num_agents).any() or
                    not bool((source < target).all())):
                raise ValueError("edge_index must contain canonical valid pairs")
        expected_relation = (edge_count, self.relation_dim)
        if self.relation_dim:
            if relation_prob is None or relation_prob.shape != expected_relation:
                raise ValueError(
                    "relation_prob must have shape [E,relation_dim]")
        elif relation_prob is not None and relation_prob.shape != (edge_count, 0):
            raise ValueError("relation_prob must be None or [E,0]")
        tensors = [trajectories, last_pos, agent_feat, edge_feat, edge_weight]
        if relation_prob is not None:
            tensors.append(relation_prob)
        if any(not torch.isfinite(tensor).all() for tensor in tensors):
            raise ValueError("Trajectory-aligner inputs must be finite")
        if (edge_weight < 0).any():
            raise ValueError("edge_weight must be non-negative")
        _, compact_scene = canonicalize_scene_index(
            scene_index, num_agents, trajectories.device)
        if edge_count:
            source, target = edge_index
            if (compact_scene[source] != compact_scene[target]).any():
                raise ValueError("Edges cannot cross scene boundaries")
        return compact_scene

    @staticmethod
    def _symmetric_edge_features(edge_feat: torch.Tensor) -> torch.Tensor:
        symmetric = edge_feat.clone()
        symmetric[:, REVERSE_SIGN_FEATURE_INDICES] = symmetric[
            :, REVERSE_SIGN_FEATURE_INDICES].abs()
        return symmetric

    def _encode_trajectories(
            self, trajectories: torch.Tensor, last_pos: torch.Tensor,
            agent_feat: torch.Tensor,
    ) -> torch.Tensor:
        previous = torch.cat((
            last_pos[:, None, None, :].expand(-1, trajectories.shape[1], 1, -1),
            trajectories[:, :, :-1]), dim=2)
        velocity = trajectories - previous
        relative = trajectories - last_pos[:, None, None, :]
        step_hidden = self.step_encoder(torch.cat((relative, velocity), dim=-1))
        trajectory_hidden = self.trajectory_encoder(torch.cat((
            step_hidden.mean(dim=2), step_hidden[:, :, -1]), dim=-1))
        return trajectory_hidden + self.agent_encoder(agent_feat)[:, None, :]

    def pair_scores(
            self,
            trajectories: torch.Tensor,
            last_pos: torch.Tensor,
            agent_feat: torch.Tensor,
            edge_index: torch.Tensor,
            edge_feat: torch.Tensor,
            edge_weight: torch.Tensor,
            relation_prob: Optional[torch.Tensor] = None,
            scene_index: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``[E,K,K]`` compatibility, edge gates and scene indices."""
        compact_scene = self._validate(
            trajectories, last_pos, agent_feat, edge_index, edge_feat,
            edge_weight, relation_prob, scene_index)
        edge_count = edge_index.shape[1]
        num_samples = trajectories.shape[1]
        if relation_prob is None:
            relation_prob = trajectories.new_empty((edge_count, 0))
        if edge_count == 0:
            return (
                trajectories.new_empty((0, num_samples, num_samples)),
                trajectories.new_empty((0,)), compact_scene)

        hidden = self._encode_trajectories(
            trajectories, last_pos, agent_feat)
        source, target = edge_index
        symmetric_edge = self._symmetric_edge_features(edge_feat)
        ensemble_source = hidden[source].mean(dim=1)
        ensemble_target = hidden[target].mean(dim=1)
        edge_context = torch.cat((
            ensemble_source + ensemble_target,
            (ensemble_source - ensemble_target).abs(),
            symmetric_edge,
            relation_prob,
        ), dim=-1)
        learned_gate = torch.sigmoid(self.edge_gate(edge_context).squeeze(-1))
        gate = learned_gate * edge_weight.clamp(min=0)

        source_hidden = hidden[source][:, :, None, :].expand(
            -1, -1, num_samples, -1)
        target_hidden = hidden[target][:, None, :, :].expand(
            -1, num_samples, -1, -1)
        source_path = trajectories[source][:, :, None, :, :]
        target_path = trajectories[target][:, None, :, :, :]
        relative_path = target_path - source_path
        distance = torch.linalg.vector_norm(relative_path, dim=-1)
        final_distance = distance[:, :, :, -1:]
        minimum_distance = distance.min(dim=-1, keepdim=True).values
        observed_relative = (
            last_pos[target] - last_pos[source])[:, None, None, :]
        formation_change = (
            relative_path[:, :, :, -1] - observed_relative).abs()
        source_motion = trajectories[source][:, :, -1] - last_pos[source][:, None]
        target_motion = trajectories[target][:, :, -1] - last_pos[target][:, None]
        motion_dot = torch.einsum('eki,eli->ekl', source_motion, target_motion)
        motion_norm = (
            torch.linalg.vector_norm(source_motion, dim=-1)[:, :, None] *
            torch.linalg.vector_norm(target_motion, dim=-1)[:, None, :])
        motion_cosine = (motion_dot / motion_norm.clamp_min(1e-6)).clamp(-1, 1)

        expanded_edge = symmetric_edge[:, None, None, :].expand(
            -1, num_samples, num_samples, -1)
        expanded_relation = relation_prob[:, None, None, :].expand(
            -1, num_samples, num_samples, -1)
        pair_input = torch.cat((
            source_hidden + target_hidden,
            (source_hidden - target_hidden).abs(),
            expanded_edge,
            expanded_relation,
            final_distance,
            minimum_distance,
            formation_change,
            motion_cosine.unsqueeze(-1),
        ), dim=-1)
        score = self.pair_scorer(pair_input).squeeze(-1)
        if not torch.isfinite(score).all():
            raise FloatingPointError("Trajectory compatibility is non-finite")
        return score, gate, compact_scene

    def _active_edge_mask(
            self, edge_index: torch.Tensor, gate: torch.Tensor,
            num_agents: int,
    ) -> torch.Tensor:
        edge_count = edge_index.shape[1]
        if self.training or self.top_k_edges <= 0 or edge_count == 0:
            return torch.ones(
                edge_count, dtype=torch.bool, device=edge_index.device)
        source, target = edge_index
        keep = torch.zeros(
            edge_count, dtype=torch.bool, device=edge_index.device)
        for agent in range(num_agents):
            incident = ((source == agent) | (target == agent)).nonzero(
                as_tuple=False).flatten()
            if incident.numel() <= self.top_k_edges:
                keep[incident] = True
            else:
                local = torch.topk(
                    gate[incident], self.top_k_edges, sorted=False).indices
                keep[incident[local]] = True
        return keep

    def _sinkhorn(self, score: torch.Tensor) -> torch.Tensor:
        log_assignment = score / self.temperature
        log_assignment = log_assignment - log_assignment.max().detach()
        for _ in range(self.sinkhorn_iterations):
            log_assignment = log_assignment - torch.logsumexp(
                log_assignment, dim=1, keepdim=True)
            log_assignment = log_assignment - torch.logsumexp(
                log_assignment, dim=0, keepdim=True)
        return log_assignment.exp()

    @staticmethod
    def _reference_agent(
            component: Sequence[int], edge_index: torch.Tensor,
            gate: torch.Tensor, agent_feat: torch.Tensor,
    ) -> int:
        # Weighted degree is permutation-equivariant; feature norm resolves
        # ordinary ties without relying on input order.
        score = agent_feat.new_zeros((agent_feat.shape[0],))
        if edge_index.shape[1]:
            source, target = edge_index
            score.index_add_(0, source, gate)
            score.index_add_(0, target, gate)
        component_tensor = torch.as_tensor(
            component, dtype=torch.long, device=agent_feat.device)
        tie_break = 1e-6 * torch.linalg.vector_norm(
            agent_feat[component_tensor], dim=-1)
        return int(component_tensor[
            (score[component_tensor] + tie_break).argmax()].item())

    def soft_assignments(
            self,
            pair_score: torch.Tensor,
            gate: torch.Tensor,
            edge_index: torch.Tensor,
            compact_scene: torch.Tensor,
            agent_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute doubly-stochastic slot-to-local-sample assignments."""
        num_agents = agent_feat.shape[0]
        num_samples = pair_score.shape[1] if pair_score.shape[0] else None
        if num_samples is None:
            raise ValueError("num_samples is ambiguous for an empty pair tensor")
        assignments = _identity_assignments(
            num_agents, num_samples, agent_feat)
        confidence = agent_feat.new_zeros((num_agents,))

        active_mask = self._active_edge_mask(
            edge_index, gate, num_agents)
        active_index = edge_index[:, active_mask]
        active_pair_score = pair_score[active_mask]
        active_gate = gate[active_mask]
        edge_lookup: Dict[Tuple[int, int], int] = {}
        for edge_id, pair in enumerate(active_index.detach().t().cpu().tolist()):
            edge_lookup[(int(pair[0]), int(pair[1]))] = edge_id

        num_scenes = int(compact_scene.max().item()) + 1
        identity = torch.eye(
            num_samples, dtype=agent_feat.dtype, device=agent_feat.device)
        for scene_id in range(num_scenes):
            scene_agents = (compact_scene == scene_id).nonzero(
                as_tuple=False).flatten().detach().cpu().tolist()
            for component in _connected_components(scene_agents, active_index):
                if len(component) <= 1:
                    continue
                reference = self._reference_agent(
                    component, active_index, active_gate, agent_feat)
                for _ in range(self.alignment_iterations):
                    updated = list(assignments)
                    for agent in component:
                        if agent == reference:
                            continue
                        slot_score = agent_feat.new_zeros(
                            (num_samples, num_samples))
                        total_gate = agent_feat.new_zeros(())
                        for neighbour in component:
                            key = (min(agent, neighbour), max(agent, neighbour))
                            if key not in edge_lookup:
                                continue
                            edge_id = edge_lookup[key]
                            local_pair = active_pair_score[edge_id]
                            local_gate = active_gate[edge_id]
                            if agent < neighbour:
                                contribution = assignments[neighbour] @ \
                                    local_pair.transpose(0, 1)
                            else:
                                contribution = assignments[neighbour] @ local_pair
                            slot_score = slot_score + local_gate * contribution
                            total_gate = total_gate + local_gate
                        if float(total_gate.detach()) <= 0:
                            continue
                        slot_score = slot_score / total_gate.clamp_min(1e-6)
                        proposal = self._sinkhorn(slot_score)
                        proposal_value = (proposal * slot_score).sum() / num_samples
                        identity_value = (identity * slot_score).sum() / num_samples
                        evidence = torch.stack((
                            proposal_value,
                            identity_value,
                            proposal_value - identity_value,
                            slot_score.std(unbiased=False),
                            total_gate / max(len(component) - 1, 1),
                        ))
                        keep_probability = torch.sigmoid(
                            self.keep_head(evidence).squeeze())
                        updated[agent] = (
                            (1.0 - keep_probability) * identity +
                            keep_probability * proposal)
                        confidence[agent] = keep_probability
                    assignments = updated
        return torch.stack(assignments), confidence

    def align(
            self,
            trajectories: torch.Tensor,
            last_pos: torch.Tensor,
            agent_feat: torch.Tensor,
            edge_index: torch.Tensor,
            edge_feat: torch.Tensor,
            edge_weight: torch.Tensor,
            relation_prob: Optional[torch.Tensor] = None,
            scene_index: Optional[torch.Tensor] = None,
            hard: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Align trajectories and return assignments plus diagnostics."""
        num_agents, num_samples = trajectories.shape[:2]
        pair_score, gate, compact_scene = self.pair_scores(
            trajectories, last_pos, agent_feat, edge_index, edge_feat,
            edge_weight, relation_prob, scene_index)
        if edge_index.shape[1] == 0:
            soft = torch.eye(
                num_samples, dtype=trajectories.dtype,
                device=trajectories.device)[None].expand(
                    num_agents, -1, -1).clone()
            confidence = trajectories.new_zeros((num_agents,))
        else:
            soft, confidence = self.soft_assignments(
                pair_score, gate, edge_index, compact_scene, agent_feat)

        if hard:
            identity_permutation = torch.arange(
                num_samples, dtype=torch.long, device=trajectories.device)
            permutations = []
            for agent in range(num_agents):
                if float(confidence[agent].detach()) < self.keep_threshold:
                    permutations.append(identity_permutation)
                else:
                    permutations.append(_greedy_bijection(soft[agent]))
            permutation = torch.stack(permutations)
            aligned = trajectories.gather(
                1, permutation[:, :, None, None].expand(
                    -1, -1, trajectories.shape[2], 2))
            assignment = F.one_hot(
                permutation, num_classes=num_samples).to(trajectories.dtype)
        else:
            permutation = soft.argmax(dim=-1)
            assignment = soft
            aligned = torch.einsum('nkl,nltd->nktd', soft, trajectories)

        changed = permutation.ne(torch.arange(
            num_samples, device=trajectories.device)[None]).float()
        active_mask = self._active_edge_mask(edge_index, gate, num_agents)
        return {
            'aligned_trajectories': aligned,
            'assignment': assignment,
            'soft_assignment': soft,
            'permutation': permutation,
            'keep_confidence': confidence,
            'pair_score': pair_score,
            'edge_gate': gate,
            'active_edge_mask': active_mask,
            'changed_fraction': changed.mean(),
        }

    @staticmethod
    def _agent_errors(
            trajectories: torch.Tensor,
            ground_truth: torch.Tensor,
            future_mask: torch.Tensor,
            fde_weight: float,
    ) -> torch.Tensor:
        """Return marginal ADE+weighted-FDE costs with shape ``[N,K]``."""
        distance = torch.linalg.vector_norm(
            trajectories - ground_truth[:, None], dim=-1)
        valid = future_mask[:, None].to(distance.dtype)
        ade = (distance * valid).sum(dim=-1) / valid.sum(dim=-1).clamp_min(1)
        # Cached social windows normally contain complete tracks.  The general
        # path still uses each agent's last valid future timestep.
        last_index = future_mask.long().sum(dim=-1).clamp_min(1) - 1
        gather = last_index[:, None, None].expand(-1, trajectories.shape[1], 1)
        fde = distance.gather(2, gather).squeeze(-1)
        return ade + float(fde_weight) * fde

    @staticmethod
    def _scene_softmin(
            trajectories: torch.Tensor,
            ground_truth: torch.Tensor,
            future_mask: torch.Tensor,
            compact_scene: torch.Tensor,
            fde_weight: float,
            temperature: float,
    ) -> torch.Tensor:
        distance = torch.linalg.vector_norm(
            trajectories - ground_truth[:, None], dim=-1)
        valid = future_mask[:, None].to(distance.dtype)
        per_agent_ade = (
            (distance * valid).sum(dim=-1) /
            valid.sum(dim=-1).clamp_min(1))
        last_index = future_mask.long().sum(dim=-1).clamp_min(1) - 1
        gather = last_index[:, None, None].expand(-1, trajectories.shape[1], 1)
        per_agent_fde = distance.gather(2, gather).squeeze(-1)
        per_agent = per_agent_ade + float(fde_weight) * per_agent_fde
        num_scenes = int(compact_scene.max().item()) + 1
        scene_cost = trajectories.new_zeros((num_scenes, trajectories.shape[1]))
        scene_cost.index_add_(0, compact_scene, per_agent)
        counts = torch.bincount(compact_scene, minlength=num_scenes).to(
            device=trajectories.device, dtype=trajectories.dtype)
        scene_cost = scene_cost / counts.clamp_min(1)[:, None]
        normalized_softmin = -float(temperature) * (
            torch.logsumexp(-scene_cost / float(temperature), dim=-1) -
            math.log(trajectories.shape[1]))
        return normalized_softmin.mean()

    def loss(
            self,
            trajectories: torch.Tensor,
            ground_truth: torch.Tensor,
            future_mask: torch.Tensor,
            last_pos: torch.Tensor,
            agent_feat: torch.Tensor,
            edge_index: torch.Tensor,
            edge_feat: torch.Tensor,
            edge_weight: torch.Tensor,
            relation_prob: Optional[torch.Tensor] = None,
            scene_index: Optional[torch.Tensor] = None,
            target_temperature: float = 0.5,
            fde_weight: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """Train against actual K predicted paths with a JMM-shaped objective."""
        if ground_truth.shape != (
                trajectories.shape[0], trajectories.shape[2], 2):
            raise ValueError("ground_truth must have shape [N,T,2]")
        if future_mask.shape != (
                trajectories.shape[0], trajectories.shape[2]):
            raise ValueError("future_mask must have shape [N,T]")
        if target_temperature <= 0 or fde_weight < 0:
            raise ValueError("Invalid alignment-loss temperature/FDE weight")
        result = self.align(
            trajectories, last_pos, agent_feat, edge_index, edge_feat,
            edge_weight, relation_prob, scene_index, hard=False)
        _, compact_scene = canonicalize_scene_index(
            scene_index, trajectories.shape[0], trajectories.device)
        aligned_joint = self._scene_softmin(
            result['aligned_trajectories'], ground_truth, future_mask,
            compact_scene, fde_weight, target_temperature)
        raw_joint = self._scene_softmin(
            trajectories, ground_truth, future_mask, compact_scene,
            fde_weight, target_temperature)

        agent_cost = self._agent_errors(
            trajectories, ground_truth, future_mask, fde_weight).detach()
        if edge_index.shape[1]:
            source, target = edge_index
            oracle_cost = (
                agent_cost[source, :, None] + agent_cost[target, None, :]) / 2
            target_prob = torch.softmax(
                -oracle_cost.reshape(oracle_cost.shape[0], -1) /
                float(target_temperature), dim=-1)
            predicted_log_prob = torch.log_softmax(
                result['pair_score'].reshape(oracle_cost.shape[0], -1) /
                float(target_temperature), dim=-1)
            pair_loss_per_edge = -(target_prob * predicted_log_prob).sum(dim=-1)
            pair_weight = result['edge_gate'].detach().clamp_min(1e-4)
            pair_loss = (
                pair_loss_per_edge * pair_weight).sum() / pair_weight.sum()
            pair_loss = pair_loss / max(
                math.log(trajectories.shape[1] ** 2), 1.0)
        else:
            pair_loss = sum(
                parameter.sum() * 0.0 for parameter in self.parameters())

        soft = result['soft_assignment'].clamp_min(1e-8)
        entropy = -(soft * soft.log()).sum(dim=-1).mean()
        entropy = entropy / max(math.log(trajectories.shape[1]), 1.0)
        no_harm = F.relu(aligned_joint - raw_joint)
        return {
            'trajectory_alignment_loss': aligned_joint,
            'trajectory_pair_loss': pair_loss,
            'alignment_no_harm_loss': no_harm,
            'alignment_entropy_loss': entropy,
            'raw_joint_surrogate': raw_joint.detach(),
            'aligned_joint_surrogate': aligned_joint.detach(),
            'mean_keep_confidence': result['keep_confidence'].mean().detach(),
            'alignment_changed_fraction': result['changed_fraction'].detach(),
            'alignment_mean_edge_gate': (
                result['edge_gate'].mean().detach()
                if result['edge_gate'].numel() else trajectories.new_zeros(())),
        }


__all__ = ["RelationAwareTrajectoryAligner"]
