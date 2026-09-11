"""Lightweight permutation-equivariant social motion encoder."""

from typing import Optional

import torch
from torch import nn

from src.models.interaction_graph import (
    DEFAULT_OBSERVATION_DT,
    EDGE_FEATURE_DIM,
    reverse_edge_features,
)


class _SocialMessageLayer(nn.Module):
    """One directed message-passing layer implemented with ``index_add_``."""

    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        agent_feat: torch.Tensor,
        edge_index: torch.Tensor,
        edge_feat: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Update ``[N,D]`` features from canonical undirected edges."""
        num_agents = agent_feat.shape[0]
        if edge_index.shape[1] == 0:
            aggregated = torch.zeros_like(agent_feat)
        else:
            source, target = edge_index
            # edge_feat stores target-source geometry.  The shared MLP always
            # receives [receiver, neighbour, neighbour-receiver geometry].
            message_to_source = self.message_mlp(
                torch.cat(
                    (agent_feat[source], agent_feat[target], edge_feat), dim=-1
                )
            )
            message_to_target = self.message_mlp(
                torch.cat(
                    (
                        agent_feat[target],
                        agent_feat[source],
                        reverse_edge_features(edge_feat),
                    ),
                    dim=-1,
                )
            )
            message_to_source = message_to_source * edge_weight[:, None]
            message_to_target = message_to_target * edge_weight[:, None]

            aggregated = torch.zeros_like(agent_feat)
            aggregated.index_add_(0, source, message_to_source)
            aggregated.index_add_(0, target, message_to_target)
            normalizer = agent_feat.new_zeros((num_agents,))
            normalizer.index_add_(0, source, edge_weight)
            normalizer.index_add_(0, target, edge_weight)
            aggregated = aggregated / normalizer.clamp_min(1.0)[:, None]

        update = self.update_mlp(torch.cat((agent_feat, aggregated), dim=-1))
        return self.norm(agent_feat + self.dropout(update))


class SocialMotionEncoder(nn.Module):
    """Encode observed tracks and exchange sparse social messages.

    The temporal encoder and every graph operation share parameters across
    agents.  Aggregation is a symmetric sum/mean, so reordering agents only
    reorders the output features.

    Parameters
    ----------
    hidden_dim:
        Internal temporal and graph feature dimension.
    output_dim:
        Dimension ``D`` of the returned agent features.  Defaults to
        ``hidden_dim``.
    edge_dim:
        Number of graph edge features.
    num_message_layers:
        Number of lightweight graph message-passing layers.
    dropout:
        Dropout applied to graph updates.
    dt:
        Observation interval used to compute input velocities.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        output_dim: Optional[int] = None,
        edge_dim: int = EDGE_FEATURE_DIM,
        num_message_layers: int = 2,
        dropout: float = 0.0,
        dt: float = DEFAULT_OBSERVATION_DT,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if output_dim is not None and output_dim <= 0:
            raise ValueError("output_dim must be positive")
        if edge_dim < 0:
            raise ValueError("edge_dim must be non-negative")
        if num_message_layers < 0:
            raise ValueError("num_message_layers must be non-negative")
        if dt <= 0:
            raise ValueError("dt must be positive")

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim or hidden_dim
        self.edge_dim = edge_dim
        self.dt = dt

        # At each timestep: position relative to the last observation (2) and
        # finite-difference velocity (2), yielding [N,T_obs,4].
        self.input_projection = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.temporal_encoder = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.message_layers = nn.ModuleList(
            _SocialMessageLayer(hidden_dim, edge_dim, dropout)
            for _ in range(num_message_layers)
        )
        self.output_projection = (
            nn.Identity()
            if self.output_dim == hidden_dim
            else nn.Linear(hidden_dim, self.output_dim)
        )

    @staticmethod
    def _validate_observations(obs_traj: torch.Tensor) -> None:
        if obs_traj.ndim != 3 or obs_traj.shape[-1] != 2:
            raise ValueError(
                "obs_traj must have shape [N, T_obs, 2], "
                f"got {tuple(obs_traj.shape)}"
            )
        if obs_traj.shape[1] < 1:
            raise ValueError("obs_traj must contain at least one timestep")
        if not torch.is_floating_point(obs_traj):
            raise TypeError("obs_traj must be a floating-point tensor")
        if not torch.isfinite(obs_traj).all():
            raise ValueError("obs_traj contains NaN or Inf")

    def _validate_graph(
        self,
        num_agents: int,
        edge_index: torch.Tensor,
        edge_feat: torch.Tensor,
        edge_weight: torch.Tensor,
        scene_index: Optional[torch.Tensor],
    ) -> None:
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
        if edge_weight.shape != (num_edges,):
            raise ValueError(
                f"edge_weight must have shape [E], got {tuple(edge_weight.shape)}"
            )
        if not torch.isfinite(edge_feat).all() or not torch.isfinite(edge_weight).all():
            raise ValueError("edge features and weights must be finite")
        if (edge_weight < 0).any():
            raise ValueError("edge_weight must be non-negative")
        if num_edges:
            if edge_index.min() < 0 or edge_index.max() >= num_agents:
                raise IndexError("edge_index contains an out-of-range agent index")
            if not edge_index[0].lt(edge_index[1]).all():
                raise ValueError("each canonical edge must satisfy source < target")
            if scene_index is not None:
                source, target = edge_index
                if not scene_index[source].eq(scene_index[target]).all():
                    raise ValueError("edge_index contains an edge crossing scene boundaries")

    def forward(
        self,
        obs_traj: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        edge_feat: Optional[torch.Tensor] = None,
        edge_weight: Optional[torch.Tensor] = None,
        scene_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return social agent features with shape ``[N, output_dim]``.

        ``obs_traj`` is ``[N,T_obs,2]`` in world coordinates.  Sparse graph
        inputs use shapes ``edge_index=[2,E]``, ``edge_feat=[E,D_edge]`` and
        ``edge_weight=[E]``.  ``N=1`` and ``E=0`` are valid inputs.
        """
        self._validate_observations(obs_traj)
        num_agents, num_steps, _ = obs_traj.shape
        device = obs_traj.device

        if scene_index is not None:
            if scene_index.ndim != 1 or scene_index.shape[0] != num_agents:
                raise ValueError("scene_index must have shape [N]")
            scene_index = scene_index.to(device=device, dtype=torch.long)

        if edge_index is None:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        else:
            edge_index = edge_index.to(device=device)
        num_edges = edge_index.shape[1]
        if edge_feat is None:
            edge_feat = obs_traj.new_zeros((num_edges, self.edge_dim))
        else:
            edge_feat = edge_feat.to(device=device, dtype=obs_traj.dtype)
        if edge_weight is None:
            edge_weight = obs_traj.new_ones((num_edges,))
        else:
            edge_weight = edge_weight.to(device=device, dtype=obs_traj.dtype)
        self._validate_graph(
            num_agents, edge_index, edge_feat, edge_weight, scene_index
        )

        if num_agents == 0:
            return obs_traj.new_empty((0, self.output_dim))

        relative_position = obs_traj - obs_traj[:, -1:, :]
        velocity = torch.zeros_like(obs_traj)
        if num_steps > 1:
            velocity[:, 1:] = (obs_traj[:, 1:] - obs_traj[:, :-1]) / self.dt
            velocity[:, 0] = velocity[:, 1]
        temporal_input = torch.cat((relative_position, velocity), dim=-1)
        temporal_input = self.input_projection(temporal_input)
        _, final_hidden = self.temporal_encoder(temporal_input)
        agent_feat = final_hidden[-1]  # [N, hidden_dim]

        for message_layer in self.message_layers:
            agent_feat = message_layer(
                agent_feat, edge_index, edge_feat, edge_weight
            )
        return self.output_projection(agent_feat)  # [N, output_dim]


# Short alias matching the conceptual module name used by the model config.
SocialEncoder = SocialMotionEncoder


__all__ = ["SocialEncoder", "SocialMotionEncoder"]
