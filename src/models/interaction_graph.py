"""Sparse, scene-aware interaction graph construction.

The graph only states that two agents may interact.  It deliberately does not
assign a repulsive or attractive meaning to an edge; that interpretation is
left to the latent relation and goal-energy modules.
"""

from typing import Dict, Optional, Tuple

import torch
from torch import nn


EDGE_FEATURE_NAMES = (
    "relative_position_x",
    "relative_position_y",
    "relative_velocity_x",
    "relative_velocity_y",
    "distance",
    "heading_similarity",
    "speed_difference",
    "time_to_closest_approach",
    "valid_time_to_closest_approach",
    "closest_approach_distance",
    "minimum_historical_distance",
    "historical_distance_variance",
    "relative_formation_change_x",
    "relative_formation_change_y",
)
EDGE_FEATURE_DIM = len(EDGE_FEATURE_NAMES)
REVERSE_SIGN_FEATURE_INDICES = (0, 1, 2, 3, 6, 12, 13)
# ETH/UCY, SDD and inD preprocessing all yield observations at 2.5 Hz.
DEFAULT_OBSERVATION_DT = 0.4


def reverse_edge_features(edge_feat: torch.Tensor) -> torch.Tensor:
    """Express canonical ``target-source`` features in the reverse direction.

    Vector differences, signed speed difference and formation change negate;
    scalar distances, heading similarity and TTC features remain unchanged.
    """
    if edge_feat.ndim != 2 or edge_feat.shape[-1] != EDGE_FEATURE_DIM:
        raise ValueError(
            f"edge_feat must have shape [E, {EDGE_FEATURE_DIM}], "
            f"got {tuple(edge_feat.shape)}"
        )
    reversed_feat = edge_feat.clone()
    reversed_feat[:, REVERSE_SIGN_FEATURE_INDICES] *= -1
    return reversed_feat


def _canonical_graph_type(graph_type: str) -> str:
    """Map supported spelling variants to one internal graph type."""
    normalized = graph_type.lower().replace("_", "-")
    aliases = {
        "full": "full",
        "radius": "radius-only",
        "radius-only": "radius-only",
        "radius+ttc": "radius+ttc",
        "radius-ttc": "radius+ttc",
    }
    if normalized not in aliases:
        choices = "full, radius-only, radius+TTC"
        raise ValueError(f"Unsupported graph_type={graph_type!r}; choose {choices}")
    return aliases[normalized]


def _validate_inputs(
    obs_traj: torch.Tensor,
    scene_index: Optional[torch.Tensor],
    radius: float,
    ttc_threshold: float,
    dt: float,
) -> torch.Tensor:
    """Validate graph inputs and return a device-local integer scene index."""
    if obs_traj.ndim != 3 or obs_traj.shape[-1] != 2:
        raise ValueError(
            "obs_traj must have shape [N, T_obs, 2], "
            f"got {tuple(obs_traj.shape)}"
        )
    if obs_traj.shape[1] < 1:
        raise ValueError("obs_traj must contain at least one observed timestep")
    if not torch.is_floating_point(obs_traj):
        raise TypeError("obs_traj must be a floating-point tensor")
    if not torch.isfinite(obs_traj).all():
        raise ValueError("obs_traj contains NaN or Inf")
    if radius <= 0:
        raise ValueError("radius must be positive")
    if ttc_threshold < 0:
        raise ValueError("ttc_threshold must be non-negative")
    if dt <= 0:
        raise ValueError("dt must be positive")

    num_agents = obs_traj.shape[0]
    if scene_index is None:
        return torch.zeros(num_agents, dtype=torch.long, device=obs_traj.device)
    if scene_index.ndim != 1 or scene_index.shape[0] != num_agents:
        raise ValueError(
            "scene_index must have shape [N], "
            f"got {tuple(scene_index.shape)} for N={num_agents}"
        )
    return scene_index.to(device=obs_traj.device, dtype=torch.long)


def _last_velocity(obs_traj: torch.Tensor, dt: float) -> torch.Tensor:
    """Estimate the current velocity from the final observed displacement."""
    if obs_traj.shape[1] == 1:
        return torch.zeros_like(obs_traj[:, -1])
    return (obs_traj[:, -1] - obs_traj[:, -2]) / dt


def build_interaction_graph(
    obs_traj: torch.Tensor,
    scene_index: Optional[torch.Tensor] = None,
    graph_type: str = "radius+TTC",
    radius: float = 3.0,
    ttc_threshold: float = 5.0,
    dt: float = DEFAULT_OBSERVATION_DT,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a sparse directed graph from observed world-coordinate tracks.

    Parameters
    ----------
    obs_traj:
        World-coordinate observations with shape ``[N, T_obs, 2]``.
    scene_index:
        Optional integer scene assignment with shape ``[N]``.  Edges never
        cross scene boundaries.  If omitted, all agents belong to one scene.
    graph_type:
        ``"full"``, ``"radius-only"`` or ``"radius+TTC"``.  The latter
        retains currently nearby pairs as well as pairs predicted to pass
        within ``radius`` during the TTC horizon.  Keeping the radius branch
        ensures parallel co-walkers are not discarded merely because their
        relative velocity is close to zero.
    radius:
        Interaction radius in the same world units as ``obs_traj``.
    ttc_threshold:
        Maximum time-to-closest-approach, in units implied by ``dt``.
    dt:
        Time interval between the final two observations.

    Returns
    -------
    edge_index:
        Canonical undirected pairs, shape ``[2, E]``.  Every pair occurs once
        and satisfies ``edge_index[0] < edge_index[1]``.
    edge_feat:
        Oriented geometric and motion features, shape
        ``[E, EDGE_FEATURE_DIM]``.
    edge_weight:
        Neutral edge weights (all ones), shape ``[E]``.  Relation-specific
        attraction, coordination or avoidance is intentionally not encoded.
    """
    graph_type = _canonical_graph_type(graph_type)
    scene_index = _validate_inputs(
        obs_traj, scene_index, radius, ttc_threshold, dt
    )
    num_agents, _, _ = obs_traj.shape
    empty_edge_index = torch.empty(
        (2, 0), dtype=torch.long, device=obs_traj.device
    )
    empty_edge_feat = obs_traj.new_empty((0, EDGE_FEATURE_DIM))
    empty_edge_weight = obs_traj.new_empty((0,))
    if num_agents < 2:
        return empty_edge_index, empty_edge_feat, empty_edge_weight

    last_pos = obs_traj[:, -1]  # [N, 2]
    velocity = _last_velocity(obs_traj, dt)  # [N, 2]

    # Candidate pairs are canonical (source < target), occur exactly once and
    # are restricted to their own scene.
    src_grid = torch.arange(num_agents, device=obs_traj.device)[:, None]
    dst_grid = torch.arange(num_agents, device=obs_traj.device)[None, :]
    same_scene = scene_index[:, None].eq(scene_index[None, :])
    candidate_mask = same_scene & src_grid.lt(dst_grid)
    src, dst = candidate_mask.nonzero(as_tuple=True)
    if src.numel() == 0:
        return empty_edge_index, empty_edge_feat, empty_edge_weight

    # All relative quantities follow the directed convention target - source.
    relative_pos = last_pos[dst] - last_pos[src]  # [E_all, 2]
    relative_vel = velocity[dst] - velocity[src]  # [E_all, 2]
    distance = torch.linalg.vector_norm(relative_pos, dim=-1)  # [E_all]

    rel_speed_sq = relative_vel.square().sum(dim=-1)
    closing_dot = (relative_pos * relative_vel).sum(dim=-1)
    valid_ttc = (rel_speed_sq > eps) & (closing_dot < 0)
    raw_ttc = -closing_dot / rel_speed_sq.clamp_min(eps)
    ttc = torch.where(valid_ttc, raw_ttc.clamp_min(0), torch.zeros_like(raw_ttc))
    closest_offset = relative_pos + ttc.unsqueeze(-1) * relative_vel
    closest_distance = torch.linalg.vector_norm(closest_offset, dim=-1)

    if graph_type == "full":
        keep = torch.ones_like(distance, dtype=torch.bool)
    elif graph_type == "radius-only":
        keep = distance <= radius
    else:
        nearby = distance <= radius
        future_encounter = (
            valid_ttc
            & (raw_ttc <= ttc_threshold)
            & (closest_distance <= radius)
        )
        keep = nearby | future_encounter

    src, dst = src[keep], dst[keep]
    if src.numel() == 0:
        return empty_edge_index, empty_edge_feat, empty_edge_weight

    relative_pos = relative_pos[keep]
    relative_vel = relative_vel[keep]
    distance = distance[keep]
    valid_ttc = valid_ttc[keep]
    ttc = ttc[keep]
    closest_distance = closest_distance[keep]

    source_speed = torch.linalg.vector_norm(velocity[src], dim=-1)
    target_speed = torch.linalg.vector_norm(velocity[dst], dim=-1)
    heading_denom = (source_speed * target_speed).clamp_min(eps)
    heading_similarity = (velocity[src] * velocity[dst]).sum(dim=-1) / heading_denom
    both_moving = (source_speed > eps) & (target_speed > eps)
    heading_similarity = torch.where(
        both_moving, heading_similarity.clamp(-1, 1), torch.zeros_like(distance)
    )

    relative_history = obs_traj[dst] - obs_traj[src]  # [E, T_obs, 2]
    historical_distance = torch.linalg.vector_norm(relative_history, dim=-1)
    minimum_historical_distance = historical_distance.min(dim=-1).values
    historical_distance_variance = historical_distance.var(dim=-1, unbiased=False)
    relative_formation_change = relative_history[:, -1] - relative_history[:, 0]

    # Invalid TTC is represented by a finite capped value plus an explicit
    # validity feature.  This keeps downstream softmax/energy paths NaN-safe.
    ttc_cap = max(float(ttc_threshold), eps)
    ttc_feature = torch.where(
        valid_ttc,
        ttc.clamp(max=ttc_cap),
        torch.full_like(ttc, ttc_cap),
    )

    edge_feat = torch.cat(
        (
            relative_pos,
            relative_vel,
            distance[:, None],
            heading_similarity[:, None],
            (target_speed - source_speed)[:, None],
            ttc_feature[:, None],
            valid_ttc.to(obs_traj.dtype)[:, None],
            closest_distance[:, None],
            minimum_historical_distance[:, None],
            historical_distance_variance[:, None],
            relative_formation_change,
        ),
        dim=-1,
    )
    edge_index = torch.stack((src, dst), dim=0)
    edge_weight = torch.ones(src.shape[0], dtype=obs_traj.dtype, device=obs_traj.device)
    return edge_index, edge_feat, edge_weight


class SparseInteractionGraph(nn.Module):
    """Radius/TTC candidate graph with an optional learned symmetric gate.

    The geometric builder remains the safety envelope: adaptive gating can
    only down-weight or prune an already valid same-scene radius/TTC edge.  It
    never invents a long-range or cross-scene interaction.  When ``adaptive``
    is disabled this module is exactly the historical parameter-free wrapper.

    ``top_k`` is a per-node proposal budget.  Each node nominates its strongest
    incident edges and the nominations are combined by union to retain a
    symmetric canonical graph.  Consequently the graph has at most ``N*k``
    edges (and average degree at most ``2*k``), while an agent can receive more
    than ``k`` edges when several neighbours nominate it.
    """

    def __init__(
        self,
        graph_type: str = "radius+TTC",
        radius: float = 3.0,
        ttc_threshold: float = 5.0,
        dt: float = DEFAULT_OBSERVATION_DT,
        adaptive: bool = False,
        gate_hidden_dim: int = 64,
        top_k: int = 0,
        gate_floor: float = 0.05,
    ) -> None:
        super().__init__()
        if gate_hidden_dim <= 0:
            raise ValueError("gate_hidden_dim must be positive")
        if top_k < 0:
            raise ValueError("top_k cannot be negative")
        if not 0 <= gate_floor < 1:
            raise ValueError("gate_floor must lie in [0, 1)")
        self.graph_type = _canonical_graph_type(graph_type)
        self.radius = radius
        self.ttc_threshold = ttc_threshold
        self.dt = dt
        self.adaptive = bool(adaptive)
        self.top_k = int(top_k)
        self.gate_floor = float(gate_floor)
        self.last_gate_prob: Optional[torch.Tensor] = None
        self.last_diagnostics: Dict[str, float] = {}

        if self.adaptive:
            self.gate_mlp = nn.Sequential(
                nn.LayerNorm(EDGE_FEATURE_DIM),
                nn.Linear(EDGE_FEATURE_DIM, gate_hidden_dim),
                nn.SiLU(),
                nn.Linear(gate_hidden_dim, gate_hidden_dim),
                nn.SiLU(),
                nn.Linear(gate_hidden_dim, 1),
            )
            # Begin near the original all-edges graph while retaining a small
            # feature-dependent perturbation so top-k does not start from ties.
            final_layer = self.gate_mlp[-1]
            nn.init.normal_(final_layer.weight, mean=0.0, std=1e-3)
            nn.init.constant_(final_layer.bias, 2.0)

    @staticmethod
    def _symmetric_top_k_mask(
        edge_index: torch.Tensor,
        score: torch.Tensor,
        num_agents: int,
        top_k: int,
    ) -> torch.Tensor:
        """Return the union of every node's strongest ``top_k`` proposals."""
        edge_count = edge_index.shape[1]
        if top_k <= 0 or edge_count == 0:
            return torch.ones(
                edge_count, dtype=torch.bool, device=edge_index.device)
        src, dst = edge_index
        keep = torch.zeros(
            edge_count, dtype=torch.bool, device=edge_index.device)
        for agent_index in range(num_agents):
            incident = ((src == agent_index) | (dst == agent_index)).nonzero(
                as_tuple=False).flatten()
            if incident.numel() <= top_k:
                keep[incident] = True
            else:
                local_top = torch.topk(
                    score[incident], k=top_k, sorted=False).indices
                keep[incident[local_top]] = True
        return keep

    def _symmetric_gate(self, edge_feat: torch.Tensor) -> torch.Tensor:
        """Score a physical edge identically under endpoint reversal."""
        forward_logit = self.gate_mlp(edge_feat).squeeze(-1)
        reverse_logit = self.gate_mlp(
            reverse_edge_features(edge_feat)).squeeze(-1)
        gate = torch.sigmoid(0.5 * (forward_logit + reverse_logit))
        return self.gate_floor + (1.0 - self.gate_floor) * gate

    def forward(
        self,
        obs_traj: torch.Tensor,
        scene_index: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``edge_index``, ``edge_feat`` and ``edge_weight``."""
        edge_index, edge_feat, edge_weight = build_interaction_graph(
            obs_traj=obs_traj,
            scene_index=scene_index,
            graph_type=self.graph_type,
            radius=self.radius,
            ttc_threshold=self.ttc_threshold,
            dt=self.dt,
        )
        candidate_edge_count = int(edge_index.shape[1])
        if not self.adaptive or candidate_edge_count == 0:
            self.last_gate_prob = edge_weight
            self.last_diagnostics = {
                "candidate_edges": float(candidate_edge_count),
                "active_edges": float(candidate_edge_count),
                "mean_gate": (float(edge_weight.mean().detach().cpu())
                              if candidate_edge_count else 0.0),
                "active_edge_fraction": (1.0 if candidate_edge_count else 0.0),
            }
            return edge_index, edge_feat, edge_weight

        gate_prob = self._symmetric_gate(edge_feat)
        active_mask = self._symmetric_top_k_mask(
            edge_index, gate_prob.detach(), obs_traj.shape[0], self.top_k)
        self.last_gate_prob = gate_prob
        active_count = int(active_mask.sum().item())
        self.last_diagnostics = {
            "candidate_edges": float(candidate_edge_count),
            "active_edges": float(active_count),
            "mean_gate": float(gate_prob.mean().detach().cpu()),
            "active_edge_fraction": (
                active_count / max(candidate_edge_count, 1)),
        }
        # Compact immediately so relation inference and KxK pair energies pay
        # only for active edges. The density objective still reads all soft
        # gates through ``last_gate_prob`` and trains even pruned candidates.
        return (
            edge_index[:, active_mask],
            edge_feat[active_mask],
            gate_prob[active_mask],
        )


# Descriptive alias for callers that prefer a builder-oriented class name.
InteractionGraphBuilder = SparseInteractionGraph


__all__ = [
    "DEFAULT_OBSERVATION_DT",
    "EDGE_FEATURE_DIM",
    "EDGE_FEATURE_NAMES",
    "InteractionGraphBuilder",
    "SparseInteractionGraph",
    "build_interaction_graph",
    "reverse_edge_features",
]
