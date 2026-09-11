"""Bounded continuous refinement for discrete joint goal proposals."""

from typing import Optional

import torch
from torch import nn

from src.models.interaction_graph import (
    EDGE_FEATURE_DIM,
    reverse_edge_features,
)


class ContinuousJointGoalRefiner(nn.Module):
    """Apply small permutation-equivariant endpoint residuals in world metres.

    Discrete TTST candidates retain responsibility for multi-modal coverage.
    This module only corrects quantization/localization error after a coherent
    joint proposal has been selected.  The final layer is initialized to zero,
    making a newly introduced refiner an exact identity before it is trained.
    """

    def __init__(
        self,
        agent_dim: int,
        edge_dim: int = EDGE_FEATURE_DIM,
        relation_dim: int = 0,
        hidden_dim: int = 128,
        num_steps: int = 1,
        max_delta: float = 0.5,
    ) -> None:
        super().__init__()
        if agent_dim <= 0 or edge_dim < 0 or relation_dim < 0:
            raise ValueError("Invalid continuous-refiner feature dimension.")
        if hidden_dim <= 0 or num_steps <= 0 or max_delta <= 0:
            raise ValueError(
                "hidden_dim, num_steps and max_delta must be positive.")
        self.agent_dim = int(agent_dim)
        self.edge_dim = int(edge_dim)
        self.relation_dim = int(relation_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_steps = int(num_steps)
        self.max_delta = float(max_delta)

        # Agent motion context plus its current endpoint displacement.
        self.node_encoder = nn.Sequential(
            nn.Linear(self.agent_dim + 2, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )
        # Receiver/neighbor context, observed edge geometry, latent relation,
        # and proposed relative endpoint displacement.
        message_input_dim = (
            2 * self.hidden_dim + self.edge_dim + self.relation_dim + 2)
        self.message_mlp = nn.Sequential(
            nn.Linear(message_input_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(self.hidden_dim),
        )
        self.delta_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 2),
        )
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)

    def _validate(
        self,
        joint_goal: torch.Tensor,
        last_pos: torch.Tensor,
        agent_feat: torch.Tensor,
        edge_index: torch.Tensor,
        edge_feat: torch.Tensor,
        edge_weight: torch.Tensor,
        relation_prob: Optional[torch.Tensor],
    ) -> None:
        if joint_goal.ndim != 3 or joint_goal.shape[-1] != 2:
            raise ValueError("joint_goal must have shape [N,P,2].")
        num_agents = joint_goal.shape[0]
        if last_pos.shape != (num_agents, 2):
            raise ValueError("last_pos must have shape [N,2].")
        if agent_feat.shape != (num_agents, self.agent_dim):
            raise ValueError(
                "agent_feat must have shape [N, agent_dim].")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2,E].")
        edge_count = edge_index.shape[1]
        if edge_feat.shape != (edge_count, self.edge_dim):
            raise ValueError("edge_feat must have shape [E, edge_dim].")
        if edge_weight.shape != (edge_count,):
            raise ValueError("edge_weight must have shape [E].")
        if edge_count:
            src, dst = edge_index
            if ((edge_index < 0).any() or
                    (edge_index >= num_agents).any() or
                    not bool((src < dst).all())):
                raise ValueError("edge_index must contain canonical valid pairs.")
        expected_relation_shape = (edge_count, self.relation_dim)
        if self.relation_dim:
            if relation_prob is None or \
                    relation_prob.shape != expected_relation_shape:
                raise ValueError(
                    "relation_prob must have shape [E, relation_dim].")
        elif relation_prob is not None and relation_prob.shape != (edge_count, 0):
            raise ValueError(
                "relation_prob must be None or [E,0] when relation_dim=0.")
        tensors = [joint_goal, last_pos, agent_feat, edge_feat, edge_weight]
        if relation_prob is not None:
            tensors.append(relation_prob)
        if any(not torch.isfinite(tensor).all() for tensor in tensors):
            raise ValueError("Continuous-refiner inputs must be finite.")
        if (edge_weight < 0).any():
            raise ValueError("edge_weight must be non-negative.")

    def forward(
        self,
        joint_goal: torch.Tensor,
        last_pos: torch.Tensor,
        agent_feat: torch.Tensor,
        edge_index: torch.Tensor,
        edge_feat: torch.Tensor,
        edge_weight: torch.Tensor,
        relation_prob: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return refined joint goals with the same ``[N,P,2]`` shape."""
        self._validate(
            joint_goal, last_pos, agent_feat, edge_index, edge_feat,
            edge_weight, relation_prob)
        num_agents, num_proposals, _ = joint_goal.shape
        edge_count = edge_index.shape[1]
        if num_agents == 0:
            return joint_goal

        if relation_prob is None:
            relation_prob = joint_goal.new_empty((edge_count, 0))
        else:
            relation_prob = relation_prob.to(
                device=joint_goal.device, dtype=joint_goal.dtype)
        edge_index = edge_index.to(
            device=joint_goal.device, dtype=torch.long)
        edge_feat = edge_feat.to(
            device=joint_goal.device, dtype=joint_goal.dtype)
        edge_weight = edge_weight.to(
            device=joint_goal.device, dtype=joint_goal.dtype)

        refined = joint_goal
        step_bound = self.max_delta / self.num_steps
        for _ in range(self.num_steps):
            displacement = refined - last_pos[:, None, :]       # [N,P,2]
            expanded_agent = agent_feat[:, None, :].expand(
                -1, num_proposals, -1)
            node_hidden = self.node_encoder(torch.cat(
                (expanded_agent, displacement), dim=-1))        # [N,P,H]

            aggregated = refined.new_zeros(
                (num_agents, num_proposals, self.hidden_dim))
            normalizer = refined.new_zeros((num_agents,))
            if edge_count:
                src, dst = edge_index
                relative_goal = refined[dst] - refined[src]     # [E,P,2]
                forward_edge = edge_feat[:, None, :].expand(
                    -1, num_proposals, -1)
                reverse_edge = reverse_edge_features(edge_feat)[
                    :, None, :].expand(-1, num_proposals, -1)
                relation = relation_prob[:, None, :].expand(
                    -1, num_proposals, -1)
                message_to_source = self.message_mlp(torch.cat((
                    node_hidden[src], node_hidden[dst], forward_edge,
                    relation, relative_goal), dim=-1))
                message_to_destination = self.message_mlp(torch.cat((
                    node_hidden[dst], node_hidden[src], reverse_edge,
                    relation, -relative_goal), dim=-1))
                weighted_source = message_to_source * edge_weight[:, None, None]
                weighted_destination = (
                    message_to_destination * edge_weight[:, None, None])
                aggregated.index_add_(0, src, weighted_source)
                aggregated.index_add_(0, dst, weighted_destination)
                normalizer.index_add_(0, src, edge_weight)
                normalizer.index_add_(0, dst, edge_weight)
                aggregated = aggregated / normalizer.clamp_min(1.0)[
                    :, None, None]

            updated = self.update_mlp(torch.cat(
                (node_hidden, aggregated), dim=-1))
            raw_delta = torch.tanh(self.delta_head(updated))
            # Bound the Euclidean correction (not each coordinate) so the
            # total displacement after all recurrent steps cannot exceed the
            # configured maximum by construction.
            raw_norm = torch.linalg.vector_norm(
                raw_delta, dim=-1, keepdim=True)
            delta = step_bound * raw_delta / raw_norm.clamp_min(1.0)
            refined = refined + delta
        if not torch.isfinite(refined).all():
            raise FloatingPointError(
                "Continuous goal refinement produced NaN or Inf.")
        return refined


__all__ = ["ContinuousJointGoalRefiner"]
