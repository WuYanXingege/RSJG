"""Latent relation inference over sparse canonical interaction edges."""

from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from src.models.interaction_graph import EDGE_FEATURE_DIM, reverse_edge_features


class RelationInference(nn.Module):
    """Infer an unlabeled soft relation distribution for every graph edge.

    A canonical pair is evaluated in both orientations with a shared network,
    and the two logits are averaged.  Consequently, changing agent order (and
    therefore flipping a canonical edge) does not change the pair's relation
    distribution.  Relation semantics remain latent; no mode is hard-coded as
    attraction, repulsion, following or yielding.

    Parameters
    ----------
    agent_dim:
        Input agent feature dimension ``D``.
    num_relation_modes:
        Number ``M`` of latent relation modes.
    edge_dim:
        Edge feature dimension ``D_edge``.
    hidden_dim:
        Width of the relation MLP.
    temperature:
        Default softmax temperature.
    hard:
        If true, use hard one-hot probabilities in the forward pass with a
        straight-through softmax gradient.
    """

    def __init__(
        self,
        agent_dim: int,
        num_relation_modes: int = 4,
        edge_dim: int = EDGE_FEATURE_DIM,
        hidden_dim: int = 128,
        temperature: float = 1.0,
        hard: bool = False,
    ) -> None:
        super().__init__()
        if agent_dim <= 0:
            raise ValueError("agent_dim must be positive")
        if num_relation_modes <= 0:
            raise ValueError("num_relation_modes must be positive")
        if edge_dim < 0:
            raise ValueError("edge_dim must be non-negative")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        self.agent_dim = agent_dim
        self.num_relation_modes = num_relation_modes
        self.edge_dim = edge_dim
        self.temperature = temperature
        self.hard = hard
        self.relation_mlp = nn.Sequential(
            nn.Linear(2 * agent_dim + edge_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_relation_modes),
        )

    def _validate_inputs(
        self,
        agent_feat: torch.Tensor,
        edge_index: torch.Tensor,
        edge_feat: torch.Tensor,
    ) -> None:
        if agent_feat.ndim != 2 or agent_feat.shape[1] != self.agent_dim:
            raise ValueError(
                f"agent_feat must have shape [N, {self.agent_dim}], "
                f"got {tuple(agent_feat.shape)}"
            )
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(
                f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}"
            )
        if edge_index.dtype != torch.long:
            raise TypeError("edge_index must use torch.long indices")
        num_edges = edge_index.shape[1]
        if edge_feat.shape != (num_edges, self.edge_dim):
            raise ValueError(
                f"edge_feat must have shape [E, {self.edge_dim}], "
                f"got {tuple(edge_feat.shape)}"
            )
        if not torch.isfinite(agent_feat).all() or not torch.isfinite(edge_feat).all():
            raise ValueError("agent_feat and edge_feat must be finite")
        if num_edges:
            if edge_index.min() < 0 or edge_index.max() >= agent_feat.shape[0]:
                raise IndexError("edge_index contains an out-of-range agent index")
            if not edge_index[0].lt(edge_index[1]).all():
                raise ValueError("each canonical edge must satisfy source < target")

    def forward(
        self,
        agent_feat: torch.Tensor,
        edge_index: torch.Tensor,
        edge_feat: torch.Tensor,
        temperature: Optional[float] = None,
        hard: Optional[bool] = None,
        return_logits: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Return relation probabilities with shape ``[E, M]``.

        Set ``hard=True`` for straight-through one-hot sampling.  The default
        is a soft mixture, as preferred for stable early training.  If
        ``return_logits`` is true, return ``(relation_prob, relation_logits)``.
        ``E=0`` is handled without a special case at the call site.
        """
        self._validate_inputs(agent_feat, edge_index, edge_feat)
        num_edges = edge_index.shape[1]
        if num_edges == 0:
            logits = agent_feat.new_empty((0, self.num_relation_modes))
            relation_prob = logits.clone()
            return (relation_prob, logits) if return_logits else relation_prob

        source, target = edge_index
        forward_input = torch.cat(
            (agent_feat[source], agent_feat[target], edge_feat), dim=-1
        )
        reverse_input = torch.cat(
            (
                agent_feat[target],
                agent_feat[source],
                reverse_edge_features(edge_feat),
            ),
            dim=-1,
        )
        forward_logits = self.relation_mlp(forward_input)
        reverse_logits = self.relation_mlp(reverse_input)
        logits = 0.5 * (forward_logits + reverse_logits)  # [E, M]

        current_temperature = self.temperature if temperature is None else temperature
        if current_temperature <= 0:
            raise ValueError("temperature must be positive")
        soft_prob = torch.softmax(logits / current_temperature, dim=-1)

        use_hard = self.hard if hard is None else hard
        if use_hard:
            hard_prob = F.one_hot(
                soft_prob.argmax(dim=-1), num_classes=self.num_relation_modes
            ).to(dtype=soft_prob.dtype)
            relation_prob = hard_prob - soft_prob.detach() + soft_prob
        else:
            relation_prob = soft_prob

        return (relation_prob, logits) if return_logits else relation_prob


LatentRelationInference = RelationInference


__all__ = ["LatentRelationInference", "RelationInference"]
