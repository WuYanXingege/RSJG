"""Sparse relation-conditioned low-rank goal compatibility energies.

Only supplied interaction edges are evaluated.  For edge ``e=(i,j)`` and
relation ``m`` the learned potential is

    E_e^m(k,l) = -w_e <A_e^m(k), B_e^m(l)> / sqrt(rank).

The effective energy is the energy of a mixture of compatibilities,

    E_eff = -logsumexp_m(log q_e(m) - E_e^m),

not the expectation ``sum_m q_e(m) E_e^m``.  Sampling uses selected-neighbor
operations with shape ``[E,K]`` and therefore never materializes a dense
``[N,N,K,K]`` tensor.
"""

import math
from typing import Dict, Optional

import torch
from torch import nn

from src.models.joint_goal import canonicalize_scene_index
from src.models.interaction_graph import EDGE_FEATURE_DIM, reverse_edge_features


def normalize_relation_prob(
        relation_prob: torch.Tensor,
        eps: float = 1e-8,
) -> torch.Tensor:
    """Normalize non-negative relation weights with a uniform zero-row fallback."""
    if relation_prob.ndim != 2:
        raise ValueError("relation_prob must have shape [E, M].")
    if relation_prob.shape[1] <= 0:
        raise ValueError("At least one relation mode is required.")
    if not torch.isfinite(relation_prob).all():
        raise ValueError("relation_prob must be finite.")
    if (relation_prob < 0).any():
        raise ValueError("relation_prob cannot contain negative values.")
    if relation_prob.shape[0] == 0:
        return relation_prob

    row_sum = relation_prob.sum(dim=-1, keepdim=True)
    uniform = torch.full_like(relation_prob, 1.0 / relation_prob.shape[-1])
    normalized = relation_prob / row_sum.clamp_min(eps)
    return torch.where(row_sum > eps, normalized, uniform)


def relation_mixture_effective_energy(
        relation_energy: torch.Tensor,
        relation_prob: torch.Tensor,
        eps: float = 1e-8,
) -> torch.Tensor:
    """Mix relation-specific energies in compatibility space using logsumexp.

    Parameters
    ----------
    relation_energy:
        Tensor shaped ``[E, M, ...]``.  The trailing dimensions can represent
        all goal pairs ``[K,K]`` or candidates against selected neighbors
        ``[K]``.
    relation_prob:
        Soft relation probabilities ``[E,M]``.

    Returns
    -------
    torch.Tensor
        Effective energy with shape ``[E,...]``.
    """
    if relation_energy.ndim < 2:
        raise ValueError("relation_energy must have shape [E, M, ...].")
    if relation_energy.shape[:2] != relation_prob.shape:
        raise ValueError(
            "relation_energy [E,M,...] and relation_prob [E,M] disagree.")
    if not torch.isfinite(relation_energy).all():
        raise ValueError("relation_energy must be finite.")

    relation_prob = normalize_relation_prob(relation_prob, eps=eps)
    log_relation_prob = relation_prob.clamp_min(eps).log()
    while log_relation_prob.ndim < relation_energy.ndim:
        log_relation_prob = log_relation_prob.unsqueeze(-1)
    return -torch.logsumexp(log_relation_prob - relation_energy, dim=1)


def relation_energy_from_factors(
        left_factor: torch.Tensor,
        right_factor: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Build sparse relation energies ``[E,M,K,K]`` from rank factors.

    This diagnostic/training operation is quadratic only in candidates for
    supplied sparse edges; there is no dense pair of agent dimensions.
    Refinement sampling should use :func:`selected_effective_energy` instead.
    """
    if left_factor.ndim != 4 or right_factor.ndim != 4:
        raise ValueError("Factors must have shape [E, M, K, rank].")
    if left_factor.shape[:2] != right_factor.shape[:2] or \
            left_factor.shape[-1] != right_factor.shape[-1]:
        raise ValueError("Left and right factor shapes are incompatible.")
    edge_count = left_factor.shape[0]
    rank = left_factor.shape[-1]
    if rank <= 0:
        raise ValueError("Factor rank must be positive.")

    compatibility = torch.einsum(
        "emkr,emlr->emkl", left_factor, right_factor)
    relation_energy = -compatibility / math.sqrt(float(rank))
    weight = _coerce_edge_weight(
        edge_weight, edge_count, left_factor.device, left_factor.dtype)
    return relation_energy * weight[:, None, None, None]


def _coerce_edge_weight(
        edge_weight: Optional[torch.Tensor],
        edge_count: int,
        device: torch.device,
        dtype: torch.dtype,
) -> torch.Tensor:
    """Validate edge strengths and return a non-negative ``[E]`` tensor."""
    if edge_weight is None:
        return torch.ones(edge_count, device=device, dtype=dtype)
    if not torch.is_tensor(edge_weight):
        edge_weight = torch.as_tensor(edge_weight, device=device, dtype=dtype)
    edge_weight = edge_weight.to(device=device, dtype=dtype)
    if edge_weight.ndim == 2 and edge_weight.shape[-1] == 1:
        edge_weight = edge_weight.squeeze(-1)
    if edge_weight.shape != (edge_count,):
        raise ValueError("edge_weight must have shape [E] or [E,1].")
    if not torch.isfinite(edge_weight).all() or (edge_weight < 0).any():
        raise ValueError("edge_weight must be finite and non-negative.")
    return edge_weight


def _validate_selected_index(
        selected_index: torch.Tensor,
        edge_count: int,
        num_candidates: int,
        device: torch.device,
) -> torch.Tensor:
    """Validate one selected candidate index for each sparse edge."""
    if not torch.is_tensor(selected_index):
        selected_index = torch.as_tensor(selected_index, device=device)
    selected_index = selected_index.to(device=device, dtype=torch.long)
    if selected_index.shape != (edge_count,):
        raise ValueError("selected_index must have shape [E].")
    if edge_count and ((selected_index < 0).any() or
                       (selected_index >= num_candidates).any()):
        raise IndexError("selected_index contains an invalid candidate id.")
    return selected_index


def selected_relation_energy(
        left_factor: torch.Tensor,
        right_factor: torch.Tensor,
        selected_index: torch.Tensor,
        conditioned_side: str = "source",
        edge_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Evaluate ``[E,M,K]`` energies against one selected neighbor per edge.

    ``conditioned_side='source'`` varies the source candidate ``k`` and fixes
    destination candidate ``l``. ``'destination'`` fixes ``k`` and varies
    ``l``.  The latter deliberately uses ``A_src(k_fixed)`` against every
    ``B_dst(l)`` so asymmetric factors retain their correct orientation.
    """
    if left_factor.ndim != 4 or right_factor.ndim != 4:
        raise ValueError("Factors must have shape [E, M, K, rank].")
    if left_factor.shape != right_factor.shape:
        raise ValueError(
            "Selected-neighbor evaluation currently requires equal K and rank.")
    edge_count, num_relations, num_candidates, rank = left_factor.shape
    selected_index = _validate_selected_index(
        selected_index, edge_count, num_candidates, left_factor.device)

    if edge_count == 0:
        return left_factor.new_empty((0, num_relations, num_candidates))

    gather_index = selected_index[:, None, None, None].expand(
        -1, num_relations, 1, rank)
    if conditioned_side in ("source", "src", "left"):
        selected_right = right_factor.gather(2, gather_index).squeeze(2)
        compatibility = (
            left_factor * selected_right.unsqueeze(2)).sum(dim=-1)
    elif conditioned_side in ("destination", "dst", "right"):
        selected_left = left_factor.gather(2, gather_index).squeeze(2)
        compatibility = (
            right_factor * selected_left.unsqueeze(2)).sum(dim=-1)
    else:
        raise ValueError(
            "conditioned_side must be 'source' or 'destination'.")

    weight = _coerce_edge_weight(
        edge_weight, edge_count, left_factor.device, left_factor.dtype)
    return (-compatibility / math.sqrt(float(rank))) * weight[:, None, None]


def selected_effective_energy(
        left_factor: torch.Tensor,
        right_factor: torch.Tensor,
        relation_prob: torch.Tensor,
        selected_index: torch.Tensor,
        conditioned_side: str = "source",
        edge_weight: Optional[torch.Tensor] = None,
        eps: float = 1e-8,
) -> torch.Tensor:
    """Return effective candidate energies ``[E,K]`` for blocked refinement."""
    relation_energy = selected_relation_energy(
        left_factor=left_factor,
        right_factor=right_factor,
        selected_index=selected_index,
        conditioned_side=conditioned_side,
        edge_weight=edge_weight,
    )
    return relation_mixture_effective_energy(
        relation_energy, relation_prob, eps=eps)


def group_relative_displacement_feature(
        goal_candidates: torch.Tensor,
        last_pos: torch.Tensor,
        edge_index: torch.Tensor,
        source_choice: Optional[torch.Tensor] = None,
        destination_choice: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute formation-change features for candidate pairs on sparse edges.

    The returned vector is

    ``(g_i - g_j) - (x_i_last - x_j_last)``

    which is equivalently the difference between the two agents' candidate
    displacements.  A co-moving pair that preserves its observed formation has
    a vector near zero.  Relation-specific factors may learn to reward it,
    penalize it, or ignore it; no universal repulsive rule is imposed.

    With no choices the shape is ``[E,K,K,2]``.  Supplying only the destination
    or source choices yields ``[E,K,2]``; supplying both yields ``[E,2]``.
    No ``[N,N,K,K]`` object is ever created.
    """
    if goal_candidates.ndim != 3 or goal_candidates.shape[-1] != 2:
        raise ValueError("goal_candidates must have shape [N,K,2].")
    num_agents, num_candidates, _ = goal_candidates.shape
    if last_pos.shape != (num_agents, 2):
        raise ValueError("last_pos must have shape [N,2].")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2,E].")
    edge_index = edge_index.to(
        device=goal_candidates.device, dtype=torch.long)
    edge_count = edge_index.shape[1]
    if edge_count and ((edge_index < 0).any() or
                       (edge_index >= num_agents).any()):
        raise IndexError("edge_index contains an invalid agent id.")

    src, dst = edge_index
    src_displacement = goal_candidates[src] - last_pos[src, None, :]
    dst_displacement = goal_candidates[dst] - last_pos[dst, None, :]

    src_choice = None
    dst_choice = None
    if source_choice is not None:
        source_choice = _validate_selected_index(
            source_choice, edge_count, num_candidates, goal_candidates.device)
        src_choice = src_displacement[
            torch.arange(edge_count, device=goal_candidates.device),
            source_choice]
    if destination_choice is not None:
        destination_choice = _validate_selected_index(
            destination_choice, edge_count, num_candidates,
            goal_candidates.device)
        dst_choice = dst_displacement[
            torch.arange(edge_count, device=goal_candidates.device),
            destination_choice]

    if src_choice is not None and dst_choice is not None:
        return src_choice - dst_choice                         # [E,2]
    if dst_choice is not None:
        return src_displacement - dst_choice[:, None, :]       # [E,K,2]
    if src_choice is not None:
        return src_choice[:, None, :] - dst_displacement       # [E,K,2]
    return (src_displacement[:, :, None, :] -
            dst_displacement[:, None, :, :])                  # [E,K,K,2]


class LowRankGoalEnergy(nn.Module):
    """Generate relation-specific rank factors on a sparse interaction graph."""

    def __init__(
            self,
            agent_dim: int,
            edge_dim: int,
            num_relation_modes: int = 4,
            rank: int = 8,
            hidden_dim: int = 128,
            use_group_relative_feature: bool = True,
    ) -> None:
        super().__init__()
        if agent_dim <= 0 or edge_dim < 0 or hidden_dim <= 0:
            raise ValueError("Invalid feature dimension.")
        if num_relation_modes <= 0 or rank <= 0:
            raise ValueError("Relation modes and rank must be positive.")

        self.agent_dim = int(agent_dim)
        self.edge_dim = int(edge_dim)
        self.num_relation_modes = int(num_relation_modes)
        self.rank = int(rank)
        self.hidden_dim = int(hidden_dim)
        self.use_group_relative_feature = bool(use_group_relative_feature)

        # Observed relative displacement is explicit in the edge context.
        pair_input_dim = 2 * self.agent_dim + self.edge_dim + 2
        self.pair_encoder = nn.Sequential(
            nn.Linear(pair_input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )
        # Each side sees its displacement from its own last position and its
        # endpoint relative to the other pedestrian's last position. Their
        # factor product can therefore model formation preservation explicitly.
        # The same endpoint encoder/factor head is used at both ends. Together
        # with evaluating the pair context in both orientations, this makes a
        # physical pair potential independent of arbitrary agent ordering;
        # directional following/yielding cues remain available in the oriented
        # edge features seen by each endpoint.
        self.goal_encoder = nn.Sequential(
            nn.Linear(4, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.relation_embedding = nn.Parameter(
            torch.empty(self.num_relation_modes, self.hidden_dim))
        nn.init.normal_(self.relation_embedding, mean=0.0,
                        std=self.hidden_dim ** -0.5)
        self.factor_head = nn.Sequential(
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.rank),
        )

    def forward(
            self,
            agent_feat: torch.Tensor,
            goal_candidates: torch.Tensor,
            last_pos: torch.Tensor,
            edge_index: torch.Tensor,
            relation_prob: Optional[torch.Tensor] = None,
            edge_feat: Optional[torch.Tensor] = None,
            edge_weight: Optional[torch.Tensor] = None,
            scene_index: Optional[torch.Tensor] = None,
            return_full_matrix: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Encode low-rank factors for candidate goals on sparse edges.

        Shapes are ``agent_feat [N,D]``, ``goal_candidates [N,K,2]``,
        ``last_pos [N,2]``, ``edge_index [2,E]``, ``edge_feat [E,D_edge]``,
        and ``relation_prob [E,M]``.  The main outputs ``left_factor`` and
        ``right_factor`` are ``[E,M,K,rank]``.
        """
        if agent_feat.ndim != 2 or agent_feat.shape[1] != self.agent_dim:
            raise ValueError("agent_feat must have shape [N, agent_dim].")
        if goal_candidates.ndim != 3 or goal_candidates.shape[-1] != 2:
            raise ValueError("goal_candidates must have shape [N,K,2].")
        num_agents, num_candidates, _ = goal_candidates.shape
        if agent_feat.shape[0] != num_agents:
            raise ValueError("agent_feat and goal_candidates disagree on N.")
        if last_pos.shape != (num_agents, 2):
            raise ValueError("last_pos must have shape [N,2].")
        if not (agent_feat.device == goal_candidates.device == last_pos.device):
            raise ValueError("All node and goal tensors must share a device.")
        if not torch.isfinite(agent_feat).all() or \
                not torch.isfinite(goal_candidates).all() or \
                not torch.isfinite(last_pos).all():
            raise ValueError("Node and goal tensors must be finite.")

        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2,E].")
        edge_index = edge_index.to(
            device=goal_candidates.device, dtype=torch.long)
        edge_count = edge_index.shape[1]
        if edge_count and ((edge_index < 0).any() or
                           (edge_index >= num_agents).any()):
            raise IndexError("edge_index contains an invalid agent id.")
        src, dst = edge_index
        if edge_count and not bool((src < dst).all()):
            raise ValueError(
                "Each canonical undirected edge must satisfy source < destination.")

        if scene_index is not None:
            _, compact_scene = canonicalize_scene_index(
                scene_index, num_agents, goal_candidates.device)
            if edge_count and (compact_scene[src] != compact_scene[dst]).any():
                raise ValueError("Interaction edges cannot cross scene windows.")

        if edge_feat is None:
            if self.edge_dim and edge_count:
                raise ValueError(
                    "edge_feat is required when edge_dim is non-zero.")
            edge_feat = goal_candidates.new_zeros((edge_count, self.edge_dim))
        else:
            edge_feat = edge_feat.to(
                device=goal_candidates.device, dtype=goal_candidates.dtype)
            if edge_feat.shape != (edge_count, self.edge_dim):
                raise ValueError("edge_feat must have shape [E, edge_dim].")
            if not torch.isfinite(edge_feat).all():
                raise ValueError("edge_feat must be finite.")

        if relation_prob is None:
            relation_prob = goal_candidates.new_full(
                (edge_count, self.num_relation_modes),
                1.0 / self.num_relation_modes)
        else:
            relation_prob = relation_prob.to(
                device=goal_candidates.device, dtype=goal_candidates.dtype)
            if relation_prob.shape != (
                    edge_count, self.num_relation_modes):
                raise ValueError("relation_prob must have shape [E,M].")
            relation_prob = normalize_relation_prob(relation_prob)

        weight = _coerce_edge_weight(
            edge_weight, edge_count, goal_candidates.device,
            goal_candidates.dtype)

        observed_relative = last_pos[src] - last_pos[dst]       # [E,2]
        source_pair_input = torch.cat(
            (agent_feat[src], agent_feat[dst], edge_feat,
             -observed_relative), dim=-1)
        reverse_feat = (reverse_edge_features(edge_feat)
                        if self.edge_dim == EDGE_FEATURE_DIM else edge_feat)
        destination_pair_input = torch.cat(
            (agent_feat[dst], agent_feat[src],
             reverse_feat,
             observed_relative), dim=-1)
        source_pair_hidden = self.pair_encoder(
            source_pair_input)                                 # [E,H]
        destination_pair_hidden = self.pair_encoder(
            destination_pair_input)                            # [E,H]

        src_displacement = (
            goal_candidates[src] - last_pos[src, None, :])      # [E,K,2]
        dst_displacement = (
            goal_candidates[dst] - last_pos[dst, None, :])      # [E,K,2]
        if self.use_group_relative_feature:
            src_to_other_last = (
                goal_candidates[src] - last_pos[dst, None, :])  # [E,K,2]
            dst_to_other_last = (
                goal_candidates[dst] - last_pos[src, None, :])  # [E,K,2]
        else:
            # Keep layer/checkpoint shapes identical for a clean ablation while
            # hiding the explicit endpoint-to-other-last formation channels.
            src_to_other_last = torch.zeros_like(src_displacement)
            dst_to_other_last = torch.zeros_like(dst_displacement)

        left_goal_hidden = self.goal_encoder(torch.cat(
            (src_displacement, src_to_other_last), dim=-1))     # [E,K,H]
        right_goal_hidden = self.goal_encoder(torch.cat(
            (dst_displacement, dst_to_other_last), dim=-1))     # [E,K,H]

        relation_hidden = self.relation_embedding[
            None, :, None, :]                                   # [1,M,1,H]
        left_hidden = (
            source_pair_hidden[:, None, None, :] + relation_hidden +
            left_goal_hidden[:, None, :, :])                    # [E,M,K,H]
        right_hidden = (
            destination_pair_hidden[:, None, None, :] + relation_hidden +
            right_goal_hidden[:, None, :, :])                   # [E,M,K,H]
        left_factor = self.factor_head(left_hidden)              # [E,M,K,r]
        right_factor = self.factor_head(right_hidden)            # [E,M,K,r]

        output = {
            "edge_index": edge_index,
            "relation_prob": relation_prob,
            "edge_weight": weight,
            "left_factor": left_factor,
            "right_factor": right_factor,
            "observed_relative_displacement": observed_relative,
            "source_goal_displacement": src_displacement,
            "destination_goal_displacement": dst_displacement,
        }
        if return_full_matrix:
            relation_energy = relation_energy_from_factors(
                left_factor, right_factor, weight)
            output["relation_energy"] = relation_energy         # [E,M,K,K]
            output["effective_energy"] = \
                relation_mixture_effective_energy(
                    relation_energy, relation_prob)              # [E,K,K]
        return output


# Public names used in different integration styles.
GoalEnergy = LowRankGoalEnergy
RelationConditionedGoalEnergy = LowRankGoalEnergy
group_relative_displacement_features = group_relative_displacement_feature


__all__ = [
    "LowRankGoalEnergy",
    "GoalEnergy",
    "RelationConditionedGoalEnergy",
    "normalize_relation_prob",
    "relation_energy_from_factors",
    "relation_mixture_effective_energy",
    "selected_relation_energy",
    "selected_effective_energy",
    "group_relative_displacement_feature",
    "group_relative_displacement_features",
]
