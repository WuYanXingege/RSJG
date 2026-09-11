"""Relation-conditioned low-rank compatibility for trajectory pairs.

For every supplied sparse edge ``e=(i,j)`` this module factorizes each latent
relation compatibility as

``compatibility[e,m,a,b] = <A[e,m,a], B[e,m,b]> / sqrt(rank)``.

Candidate-conditioned relation probabilities are mixed in *compatibility
space* with ``logsumexp(log q + compatibility)``.  A differentiable gate may
then act per candidate pair, per edge, or be disabled.  No branch detaches the
gate and no training-time top-k pruning is performed here; hard inference
pruning belongs to the downstream synchronizer wrapper.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
from torch import nn

from src.models.trajectory_pair_relation import (
    TRAJECTORY_PAIR_GEOMETRY_DIM,
    _reverse_edge_context,
    _validate_sparse_edges,
    reverse_trajectory_pair_geometry,
)


def trajectory_relation_energy_from_factors(
        left_factor: torch.Tensor,
        right_factor: torch.Tensor,
) -> torch.Tensor:
    """Return relation energies ``[E,K,K,M]`` from low-rank factors.

    Factors use shapes ``[E,M,K,R]``.  Energy is negative compatibility, so a
    larger factor inner product denotes a more desirable trajectory pairing.
    """
    if left_factor.ndim != 4 or right_factor.ndim != 4:
        raise ValueError("factors must have shape [E,M,K,rank]")
    if left_factor.shape != right_factor.shape:
        raise ValueError("left and right factors must have identical shapes")
    if left_factor.shape[-1] <= 0:
        raise ValueError("factor rank must be positive")
    if not torch.isfinite(left_factor).all() or not torch.isfinite(
            right_factor).all():
        raise ValueError("low-rank factors must be finite")
    compatibility = torch.einsum(
        "emkr,emlr->eklm", left_factor, right_factor)
    return -compatibility / math.sqrt(float(left_factor.shape[-1]))


def trajectory_relation_mixture_compatibility(
        relation_energy: torch.Tensor,
        relation_prob: torch.Tensor,
        eps: float = 1e-8,
) -> torch.Tensor:
    """Mix modes in compatibility space and return ``[E,K,K]`` scores.

    ``relation_energy`` and ``relation_prob`` have shapes ``[E,K,K,M]``.
    Zero-probability rows receive a uniform fallback; negative or non-finite
    relation weights are rejected.
    """
    if relation_energy.ndim != 4:
        raise ValueError("relation_energy must have shape [E,K,K,M]")
    if relation_prob.shape != relation_energy.shape:
        raise ValueError("relation_prob must match relation_energy")
    if eps <= 0:
        raise ValueError("eps must be positive")
    if not torch.isfinite(relation_energy).all() or not torch.isfinite(
            relation_prob).all():
        raise ValueError("relation energy/probability must be finite")
    if bool((relation_prob < 0).any()):
        raise ValueError("relation_prob must be non-negative")
    relation_sum = relation_prob.sum(dim=-1, keepdim=True)
    num_modes = relation_prob.shape[-1]
    if num_modes <= 0:
        raise ValueError("at least one relation mode is required")
    uniform = torch.full_like(relation_prob, 1.0 / num_modes)
    normalized = torch.where(
        relation_sum > eps,
        relation_prob / relation_sum.clamp_min(eps), uniform)
    # -E is relation-specific compatibility.  This is intentionally not
    # ``sum(q * E)``: a latent mixture is a log mixture of compatibilities.
    return torch.logsumexp(
        normalized.clamp_min(eps).log() - relation_energy, dim=-1)


def _normalize_gate_type(gate_type: str) -> str:
    normalized = str(gate_type).lower().replace("-", "_")
    aliases = {
        "pair": "pair",
        "pair_specific": "pair",
        "pairwise": "pair",
        "edge": "edge",
        "edge_level": "edge",
        "none": "none",
        "off": "none",
    }
    if normalized not in aliases:
        raise ValueError("gate_type must be pair, edge, or none")
    return aliases[normalized]


def _normalize_energy_type(energy_type: str) -> str:
    normalized = str(energy_type).lower().replace("-", "_")
    aliases = {
        "lowrank": "lowrank",
        "low_rank": "lowrank",
        "mlp": "mlp",
        "direct_mlp": "mlp",
    }
    if normalized not in aliases:
        raise ValueError("energy_type must be lowrank or mlp")
    return aliases[normalized]


class RelationConditionedTrajectoryPairEnergy(nn.Module):
    """Generate sparse low-rank trajectory compatibility and continuous gates.

    Parameters
    ----------
    trajectory_dim:
        Dimension of complete-trajectory embeddings ``h_i^a``.
    edge_dim:
        Dimension of observation graph edge features.
    num_relation_modes:
        Number of latent relation modes ``M``.
    rank:
        Per-mode low-rank factor size.
    gate_type:
        ``"pair"`` for ``g_ij^{ab}``, ``"edge"`` for one shared ``g_ij``,
        or ``"none"``.  All choices remain differentiable; this class never
        applies hard top-k pruning.
    energy_type:
        ``"lowrank"`` for relation-specific rank factors (the V4 default),
        or ``"mlp"`` for a direct pair scorer with the same mixture/gating
        contract.  The latter is the required trajectory-energy ablation.
    """

    def __init__(
            self,
            trajectory_dim: int,
            edge_dim: int,
            num_relation_modes: int = 4,
            rank: int = 8,
            geometry_dim: int = TRAJECTORY_PAIR_GEOMETRY_DIM,
            hidden_dim: int = 128,
            gate_type: str = "pair",
            pair_specific_gate: Optional[bool] = None,
            energy_type: str = "lowrank",
            eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if trajectory_dim <= 0 or edge_dim < 0 or geometry_dim <= 0:
            raise ValueError("invalid trajectory/edge/geometry dimension")
        if num_relation_modes <= 0 or rank <= 0 or hidden_dim <= 0:
            raise ValueError("relation modes, rank and hidden_dim must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")
        if pair_specific_gate is not None:
            gate_type = "pair" if pair_specific_gate else "edge"
        self.trajectory_dim = int(trajectory_dim)
        self.edge_dim = int(edge_dim)
        self.num_relation_modes = int(num_relation_modes)
        self.rank = int(rank)
        self.geometry_dim = int(geometry_dim)
        self.hidden_dim = int(hidden_dim)
        self.gate_type = _normalize_gate_type(gate_type)
        self.energy_type = _normalize_energy_type(energy_type)
        self.eps = float(eps)

        self.candidate_encoder = nn.Sequential(
            nn.Linear(self.trajectory_dim, self.hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(self.hidden_dim),
        )
        self.edge_context_encoder = nn.Sequential(
            nn.Linear(2 * self.trajectory_dim + self.edge_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )
        self.relation_embedding = nn.Parameter(torch.empty(
            self.num_relation_modes, self.hidden_dim))
        nn.init.normal_(self.relation_embedding, mean=0.0,
                        std=self.hidden_dim ** -0.5)
        self.factor_head = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.rank),
        )
        direct_energy_input = (
            2 * self.trajectory_dim + self.edge_dim + self.geometry_dim)
        self.direct_compatibility_head = nn.Sequential(
            nn.Linear(direct_energy_input, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.num_relation_modes),
        )

        pair_gate_input = (
            2 * self.trajectory_dim + self.edge_dim + self.geometry_dim +
            self.num_relation_modes)
        self.pair_gate_head = nn.Sequential(
            nn.Linear(pair_gate_input, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, max(self.hidden_dim // 2, 1)),
            nn.SiLU(),
            nn.Linear(max(self.hidden_dim // 2, 1), 1),
        )
        edge_gate_input = (
            2 * self.trajectory_dim + self.edge_dim + self.num_relation_modes)
        self.edge_gate_head = nn.Sequential(
            nn.Linear(edge_gate_input, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 1),
        )
        # Neutral 0.5 gates are safer than initially suppressing all graph
        # evidence, while small weights break exact ties for task gradients.
        nn.init.normal_(self.pair_gate_head[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.pair_gate_head[-1].bias)
        nn.init.normal_(self.edge_gate_head[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.edge_gate_head[-1].bias)
        nn.init.normal_(self.direct_compatibility_head[-1].weight,
                        mean=0.0, std=1e-3)
        nn.init.zeros_(self.direct_compatibility_head[-1].bias)

    def _validate(
            self,
            trajectory_feat: torch.Tensor,
            edge_index: torch.Tensor,
            edge_feat: torch.Tensor,
            relation_prob: torch.Tensor,
            pair_geometry: torch.Tensor,
            edge_weight: Optional[torch.Tensor],
            scene_index: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if trajectory_feat.ndim != 3 or trajectory_feat.shape[-1] != self.trajectory_dim:
            raise ValueError(
                "trajectory_feat must have shape [N,K,trajectory_dim]")
        num_agents, num_candidates, _ = trajectory_feat.shape
        if num_agents <= 0 or num_candidates <= 0:
            raise ValueError("trajectory_feat cannot be empty in N or K")
        if not torch.is_floating_point(trajectory_feat) or not torch.isfinite(
                trajectory_feat).all():
            raise ValueError("trajectory_feat must be finite floating point")
        _validate_sparse_edges(
            edge_index, num_agents, trajectory_feat.device, scene_index,
            require_canonical=True)
        edge_count = edge_index.shape[1]
        if edge_feat.shape != (edge_count, self.edge_dim):
            raise ValueError("edge_feat must have shape [E,edge_dim]")
        if pair_geometry.shape != (
                edge_count, num_candidates, num_candidates, self.geometry_dim):
            raise ValueError("pair_geometry must have shape [E,K,K,geometry_dim]")
        if relation_prob.ndim == 2:
            if relation_prob.shape != (edge_count, self.num_relation_modes):
                raise ValueError("edge relation_prob must have shape [E,M]")
            relation_prob = relation_prob[:, None, None, :].expand(
                -1, num_candidates, num_candidates, -1)
        elif relation_prob.shape != (
                edge_count, num_candidates, num_candidates,
                self.num_relation_modes):
            raise ValueError("relation_prob must have shape [E,K,K,M] or [E,M]")
        floating_inputs = (edge_feat, pair_geometry, relation_prob)
        if any(tensor.device != trajectory_feat.device for tensor in floating_inputs):
            raise ValueError("all energy inputs must share a device")
        if any(tensor.dtype != trajectory_feat.dtype for tensor in floating_inputs):
            raise ValueError("all floating energy inputs must share a dtype")
        if any(not torch.isfinite(tensor).all() for tensor in floating_inputs):
            raise ValueError("all energy inputs must be finite")
        if bool((relation_prob < 0).any()):
            raise ValueError("relation_prob must be non-negative")
        relation_sum = relation_prob.sum(dim=-1, keepdim=True)
        uniform = torch.full_like(
            relation_prob, 1.0 / self.num_relation_modes)
        relation_prob = torch.where(
            relation_sum > self.eps,
            relation_prob / relation_sum.clamp_min(self.eps), uniform)

        if edge_weight is None:
            weight = trajectory_feat.new_ones((edge_count,))
        else:
            if edge_weight.shape not in ((edge_count,), (edge_count, 1)):
                raise ValueError("edge_weight must have shape [E] or [E,1]")
            weight = edge_weight.reshape(edge_count)
            if weight.device != trajectory_feat.device or \
                    weight.dtype != trajectory_feat.dtype:
                raise ValueError("edge_weight must share input device and dtype")
            if not torch.isfinite(weight).all() or bool((weight < 0).any()):
                raise ValueError("edge_weight must be finite and non-negative")
        return relation_prob, weight

    def _low_rank_factors(
            self,
            trajectory_feat: torch.Tensor,
            edge_index: torch.Tensor,
            edge_feat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source, destination = edge_index
        source_mean = trajectory_feat[source].mean(dim=1)
        destination_mean = trajectory_feat[destination].mean(dim=1)
        source_context = self.edge_context_encoder(torch.cat((
            source_mean, destination_mean, edge_feat), dim=-1))
        destination_context = self.edge_context_encoder(torch.cat((
            destination_mean, source_mean, _reverse_edge_context(edge_feat)),
            dim=-1))
        encoded_candidate = self.candidate_encoder(trajectory_feat)
        source_candidate = encoded_candidate[source]
        destination_candidate = encoded_candidate[destination]
        relation = self.relation_embedding[None, :, None, :]
        left_hidden = (
            source_context[:, None, None, :] +
            source_candidate[:, None, :, :] + relation)
        right_hidden = (
            destination_context[:, None, None, :] +
            destination_candidate[:, None, :, :] + relation)
        return self.factor_head(left_hidden), self.factor_head(right_hidden)

    def _pair_gate(
            self,
            trajectory_feat: torch.Tensor,
            edge_index: torch.Tensor,
            edge_feat: torch.Tensor,
            relation_prob: torch.Tensor,
            pair_geometry: torch.Tensor,
    ) -> torch.Tensor:
        edge_count = edge_index.shape[1]
        num_candidates = trajectory_feat.shape[1]
        if self.gate_type == "none":
            return trajectory_feat.new_ones((
                edge_count, num_candidates, num_candidates))

        source, destination = edge_index
        reverse_edge = _reverse_edge_context(edge_feat)
        if self.gate_type == "edge":
            source_mean = trajectory_feat[source].mean(dim=1)
            destination_mean = trajectory_feat[destination].mean(dim=1)
            mean_relation = relation_prob.mean(dim=(1, 2))
            forward_input = torch.cat((
                source_mean, destination_mean, edge_feat, mean_relation), dim=-1)
            reverse_input = torch.cat((
                destination_mean, source_mean, reverse_edge, mean_relation), dim=-1)
            edge_logit = 0.5 * (
                self.edge_gate_head(forward_input) +
                self.edge_gate_head(reverse_input))
            edge_gate = torch.sigmoid(edge_logit.squeeze(-1))
            return edge_gate[:, None, None].expand(
                -1, num_candidates, num_candidates)

        source_hidden = trajectory_feat[source][:, :, None, :].expand(
            -1, -1, num_candidates, -1)
        destination_hidden = trajectory_feat[destination][:, None, :, :].expand(
            -1, num_candidates, -1, -1)
        expanded_edge = edge_feat[:, None, None, :].expand(
            -1, num_candidates, num_candidates, -1)
        reverse_expanded_edge = reverse_edge[:, None, None, :].expand(
            -1, num_candidates, num_candidates, -1)
        forward_input = torch.cat((
            source_hidden, destination_hidden, expanded_edge,
            pair_geometry, relation_prob), dim=-1)
        reverse_geometry = (
            reverse_trajectory_pair_geometry(pair_geometry)
            if self.geometry_dim == TRAJECTORY_PAIR_GEOMETRY_DIM
            else pair_geometry)
        reverse_input = torch.cat((
            destination_hidden, source_hidden, reverse_expanded_edge,
            reverse_geometry, relation_prob), dim=-1)
        gate_logit = 0.5 * (
            self.pair_gate_head(forward_input) +
            self.pair_gate_head(reverse_input))
        return torch.sigmoid(gate_logit.squeeze(-1))

    def _direct_relation_energy(
            self,
            trajectory_feat: torch.Tensor,
            edge_index: torch.Tensor,
            edge_feat: torch.Tensor,
            pair_geometry: torch.Tensor,
    ) -> torch.Tensor:
        """Return symmetric direct-MLP energies ``[E,K,K,M]``."""
        num_candidates = trajectory_feat.shape[1]
        source, destination = edge_index
        source_hidden = trajectory_feat[source][:, :, None, :].expand(
            -1, -1, num_candidates, -1)
        destination_hidden = trajectory_feat[destination][:, None, :, :].expand(
            -1, num_candidates, -1, -1)
        expanded_edge = edge_feat[:, None, None, :].expand(
            -1, num_candidates, num_candidates, -1)
        reverse_edge = _reverse_edge_context(edge_feat)
        expanded_reverse_edge = reverse_edge[:, None, None, :].expand(
            -1, num_candidates, num_candidates, -1)
        reverse_geometry = (
            reverse_trajectory_pair_geometry(pair_geometry)
            if self.geometry_dim == TRAJECTORY_PAIR_GEOMETRY_DIM
            else pair_geometry)
        forward_input = torch.cat((
            source_hidden, destination_hidden, expanded_edge,
            pair_geometry), dim=-1)
        reverse_input = torch.cat((
            destination_hidden, source_hidden, expanded_reverse_edge,
            reverse_geometry), dim=-1)
        compatibility = 0.5 * (
            self.direct_compatibility_head(forward_input) +
            self.direct_compatibility_head(reverse_input))
        return -compatibility

    def forward(
            self,
            trajectory_feat: torch.Tensor,
            edge_index: torch.Tensor,
            edge_feat: torch.Tensor,
            relation_prob: torch.Tensor,
            pair_geometry: torch.Tensor,
            edge_weight: Optional[torch.Tensor] = None,
            scene_index: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Return factors, relation energies, gates and ``pair_score[E,K,K]``."""
        relation_prob, weight = self._validate(
            trajectory_feat, edge_index, edge_feat, relation_prob,
            pair_geometry, edge_weight, scene_index)
        if self.energy_type == "lowrank":
            left_factor, right_factor = self._low_rank_factors(
                trajectory_feat, edge_index, edge_feat)
            relation_energy = trajectory_relation_energy_from_factors(
                left_factor, right_factor)
        else:
            edge_count = edge_index.shape[1]
            num_candidates = trajectory_feat.shape[1]
            left_factor = trajectory_feat.new_zeros((
                edge_count, self.num_relation_modes, num_candidates,
                self.rank))
            right_factor = torch.zeros_like(left_factor)
            relation_energy = self._direct_relation_energy(
                trajectory_feat, edge_index, edge_feat, pair_geometry)
        compatibility = trajectory_relation_mixture_compatibility(
            relation_energy, relation_prob, eps=self.eps)
        pair_gate = self._pair_gate(
            trajectory_feat, edge_index, edge_feat, relation_prob,
            pair_geometry)
        pair_score = compatibility * pair_gate * weight[:, None, None]
        tensors = (left_factor, right_factor, relation_energy, compatibility,
                   pair_gate, pair_score)
        if any(not torch.isfinite(tensor).all() for tensor in tensors):
            raise FloatingPointError("trajectory-pair energy is non-finite")
        return {
            "edge_index": edge_index,
            "left_factor": left_factor,
            "right_factor": right_factor,
            "relation_energy": relation_energy,
            "effective_energy": -compatibility,
            "pair_compatibility": compatibility,
            "pair_gate": pair_gate,
            "edge_gate": pair_gate.mean(dim=(1, 2)),
            "edge_weight": weight,
            "pair_score": pair_score,
            "relation_prob": relation_prob,
        }


LowRankTrajectoryPairEnergy = RelationConditionedTrajectoryPairEnergy
TrajectoryPairEnergy = RelationConditionedTrajectoryPairEnergy


__all__ = [
    "LowRankTrajectoryPairEnergy",
    "RelationConditionedTrajectoryPairEnergy",
    "TrajectoryPairEnergy",
    "trajectory_relation_energy_from_factors",
    "trajectory_relation_mixture_compatibility",
]
