"""Trajectory-pair-conditioned latent relations for multiway coupling.

This module deliberately operates on a *fixed* bank of complete future
trajectories.  It does not move trajectory coordinates.  For every sparse
edge ``e=(i,j)`` and candidate pair ``(a,b)`` it provides

* a shared temporal encoding ``h_i^a`` of the complete predicted path;
* geometric evidence ``phi(Y_i^a, Y_j^b)``; and
* a posterior relation distribution obtained by adding a learned residual to
  the observation-only prior ``q0(r_ij | X)``.

All pair tensors are sparse in the agent dimension: their leading dimension
is the number of supplied graph edges ``E``, never a dense ``N x N`` grid.
The orientation convention is row/source candidate first and
column/destination candidate second.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn

from src.models.interaction_graph import EDGE_FEATURE_DIM, reverse_edge_features
from src.models.joint_goal import canonicalize_scene_index


TRAJECTORY_PAIR_GEOMETRY_NAMES = (
    "endpoint_distance",
    "minimum_path_distance",
    "mean_path_distance",
    "formation_change_x",
    "formation_change_y",
    "displacement_cosine",
    "mean_velocity_cosine",
    "velocity_divergence",
    "predicted_closest_time",
    "endpoint_heading_cosine",
    "closing_rate",
    "endpoint_distance_change",
)
TRAJECTORY_PAIR_GEOMETRY_DIM = len(TRAJECTORY_PAIR_GEOMETRY_NAMES)

# Under endpoint reversal, destination-source formation vectors change sign;
# every other feature above is a physical-pair scalar and remains unchanged.
TRAJECTORY_PAIR_GEOMETRY_REVERSE_SIGN_INDICES = (3, 4)


def _safe_cosine(
        left: torch.Tensor,
        right: torch.Tensor,
        eps: float,
) -> torch.Tensor:
    """Return a finite cosine, using zero when either vector is stationary."""
    numerator = (left * right).sum(dim=-1)
    left_norm = torch.linalg.vector_norm(left, dim=-1)
    right_norm = torch.linalg.vector_norm(right, dim=-1)
    denominator = left_norm * right_norm
    cosine = numerator / denominator.clamp_min(eps)
    moving = (left_norm > eps) & (right_norm > eps)
    return torch.where(moving, cosine.clamp(-1.0, 1.0), torch.zeros_like(cosine))


def _future_velocity(
        trajectories: torch.Tensor,
        last_pos: torch.Tensor,
        dt: float,
) -> torch.Tensor:
    """Finite-difference candidate velocity with shape ``[N,K,T,2]``."""
    previous = torch.cat((
        last_pos[:, None, None, :].expand(-1, trajectories.shape[1], 1, -1),
        trajectories[:, :, :-1],
    ), dim=2)
    return (trajectories - previous) / dt


def _validate_trajectory_bank(
        trajectories: torch.Tensor,
        last_pos: torch.Tensor,
) -> None:
    if trajectories.ndim != 4 or trajectories.shape[-1] != 2:
        raise ValueError("trajectories must have shape [N,K,T,2]")
    num_agents, num_candidates, num_steps, _ = trajectories.shape
    if num_agents <= 0 or num_candidates <= 0 or num_steps <= 0:
        raise ValueError("trajectories must contain at least one agent/candidate/step")
    if not torch.is_floating_point(trajectories):
        raise TypeError("trajectories must be floating point")
    if last_pos.shape != (num_agents, 2):
        raise ValueError("last_pos must have shape [N,2]")
    if last_pos.device != trajectories.device:
        raise ValueError("trajectories and last_pos must share a device")
    if last_pos.dtype != trajectories.dtype:
        raise ValueError("trajectories and last_pos must share a dtype")
    if not torch.isfinite(trajectories).all() or not torch.isfinite(last_pos).all():
        raise ValueError("trajectory coordinates must be finite")


def _validate_sparse_edges(
        edge_index: torch.Tensor,
        num_agents: int,
        device: torch.device,
        scene_index: Optional[torch.Tensor],
        require_canonical: bool = False,
) -> torch.Tensor:
    """Validate oriented sparse edges and return compact scene ids.

    Geometry construction accepts either orientation so callers can explicitly
    reverse an edge for equivariance tests.  Model-facing code normally passes
    canonical ``source < destination`` edges and can request that invariant.
    Self edges are never meaningful trajectory-pair interactions.
    """
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2,E]")
    if edge_index.dtype != torch.long:
        raise TypeError("edge_index must use torch.long indices")
    if edge_index.device != device:
        raise ValueError("edge_index and trajectories must share a device")
    edge_count = edge_index.shape[1]
    if edge_count:
        if bool((edge_index < 0).any()) or bool((edge_index >= num_agents).any()):
            raise IndexError("edge_index contains an invalid agent id")
        source, destination = edge_index
        if bool((source == destination).any()):
            raise ValueError("self edges are not valid trajectory-pair edges")
        if require_canonical and not bool((source < destination).all()):
            raise ValueError("canonical edges must satisfy source < destination")

    _, compact_scene = canonicalize_scene_index(
        scene_index, num_agents, device)
    if edge_count:
        source, destination = edge_index
        if bool((compact_scene[source] != compact_scene[destination]).any()):
            raise ValueError("trajectory-pair edges cannot cross scene boundaries")
    return compact_scene


def reverse_trajectory_pair_geometry(geometry: torch.Tensor) -> torch.Tensor:
    """Reverse endpoint orientation without changing candidate-axis order.

    ``geometry[..., a, b, :]`` for ``i -> j`` becomes the features for
    ``j -> i`` while the caller still stores old ``(a,b)`` axes.  To compare a
    separately constructed reverse-edge tensor, also transpose its two
    candidate axes.
    """
    if geometry.ndim < 1 or geometry.shape[-1] != TRAJECTORY_PAIR_GEOMETRY_DIM:
        raise ValueError(
            f"geometry must end in {TRAJECTORY_PAIR_GEOMETRY_DIM} features")
    reversed_geometry = geometry.clone()
    reversed_geometry[..., TRAJECTORY_PAIR_GEOMETRY_REVERSE_SIGN_INDICES] *= -1
    return reversed_geometry


def build_trajectory_pair_geometry(
        trajectories: torch.Tensor,
        last_pos: torch.Tensor,
        edge_index: torch.Tensor,
        scene_index: Optional[torch.Tensor] = None,
        dt: float = 0.4,
        eps: float = 1e-6,
) -> torch.Tensor:
    """Build complete-path geometric evidence with shape ``[E,K,K,P]``.

    Parameters
    ----------
    trajectories:
        Fixed future bank ``[N,K,T,2]`` in common world coordinates.
    last_pos:
        Last observed position ``[N,2]``.
    edge_index:
        Oriented sparse pairs ``[2,E]``.  Rows in the output index source
        candidates and columns index destination candidates.
    scene_index:
        Optional packed-scene id per agent.  Cross-scene edges are rejected.
    dt:
        Prediction interval used for velocity and acceleration features.

    Notes
    -----
    Distance is *evidence*, not a hand-coded repulsion penalty.  Formation,
    co-motion, crossing and following semantics are learned by the downstream
    latent-relation and energy networks.
    """
    _validate_trajectory_bank(trajectories, last_pos)
    if dt <= 0 or eps <= 0:
        raise ValueError("dt and eps must be positive")
    num_agents, num_candidates, num_steps, _ = trajectories.shape
    _validate_sparse_edges(
        edge_index, num_agents, trajectories.device, scene_index)
    edge_count = edge_index.shape[1]
    if edge_count == 0:
        return trajectories.new_empty((
            0, num_candidates, num_candidates,
            TRAJECTORY_PAIR_GEOMETRY_DIM))

    source, destination = edge_index
    source_path = trajectories[source][:, :, None, :, :]       # [E,K,1,T,2]
    destination_path = trajectories[destination][:, None, :, :, :]  # [E,1,K,T,2]
    relative_path = destination_path - source_path
    path_distance = torch.linalg.vector_norm(relative_path, dim=-1)

    endpoint_distance = path_distance[..., -1]
    minimum_distance = path_distance.min(dim=-1).values
    mean_distance = path_distance.mean(dim=-1)

    observed_relative = (
        last_pos[destination] - last_pos[source])[:, None, None, :]
    formation_change = relative_path[..., -1, :] - observed_relative

    source_displacement = (
        trajectories[source, :, -1] - last_pos[source, None, :])
    destination_displacement = (
        trajectories[destination, :, -1] - last_pos[destination, None, :])
    displacement_cosine = _safe_cosine(
        source_displacement[:, :, None, :],
        destination_displacement[:, None, :, :], eps)

    all_velocity = _future_velocity(trajectories, last_pos, dt)
    source_velocity = all_velocity[source][:, :, None, :, :]
    destination_velocity = all_velocity[destination][:, None, :, :, :]
    velocity_cosine = _safe_cosine(
        source_velocity, destination_velocity, eps).mean(dim=-1)
    relative_velocity = destination_velocity - source_velocity
    velocity_divergence = torch.linalg.vector_norm(
        relative_velocity, dim=-1).mean(dim=-1)

    # Discrete time of closest predicted approach.  It is normalized to [0,1]
    # so feature scale is invariant to the prediction horizon.  Min/mean path
    # distances provide differentiable evidence even though argmin itself is
    # intentionally used only as a descriptor.
    closest_step = path_distance.argmin(dim=-1).to(trajectories.dtype)
    closest_time = (closest_step + 1.0) / float(num_steps)

    heading_cosine = _safe_cosine(
        source_velocity[..., -1, :],
        destination_velocity[..., -1, :], eps)

    relative_unit = relative_path / path_distance.clamp_min(eps).unsqueeze(-1)
    closing_rate = (
        relative_velocity * relative_unit).sum(dim=-1).mean(dim=-1)
    observed_distance = torch.linalg.vector_norm(
        last_pos[destination] - last_pos[source], dim=-1)[:, None, None]
    endpoint_distance_change = endpoint_distance - observed_distance

    geometry = torch.cat((
        endpoint_distance.unsqueeze(-1),
        minimum_distance.unsqueeze(-1),
        mean_distance.unsqueeze(-1),
        formation_change,
        displacement_cosine.unsqueeze(-1),
        velocity_cosine.unsqueeze(-1),
        velocity_divergence.unsqueeze(-1),
        closest_time.unsqueeze(-1),
        heading_cosine.unsqueeze(-1),
        closing_rate.unsqueeze(-1),
        endpoint_distance_change.unsqueeze(-1),
    ), dim=-1)
    if geometry.shape != (
            edge_count, num_candidates, num_candidates,
            TRAJECTORY_PAIR_GEOMETRY_DIM):
        raise RuntimeError("internal trajectory-pair geometry shape error")
    if not torch.isfinite(geometry).all():
        raise FloatingPointError("trajectory-pair geometry is non-finite")
    return geometry


class TrajectoryEncoder(nn.Module):
    """Encode every complete candidate path into a shared ``[N,K,D]`` space.

    Per-step features contain displacement from the final observation (2),
    finite-difference velocity (2), acceleration (2), and normalized future
    time (1).  A lightweight shared GRU sees all prediction steps; parameters
    are shared across every agent and candidate, preserving agent/sample
    permutation equivariance.
    """

    def __init__(
            self,
            output_dim: int = 128,
            hidden_dim: Optional[int] = None,
            dt: float = 0.4,
    ) -> None:
        super().__init__()
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")
        hidden_dim = output_dim if hidden_dim is None else hidden_dim
        if hidden_dim <= 0 or dt <= 0:
            raise ValueError("hidden_dim and dt must be positive")
        self.output_dim = int(output_dim)
        self.hidden_dim = int(hidden_dim)
        self.dt = float(dt)
        self.step_projection = nn.Sequential(
            nn.Linear(7, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
        )
        self.temporal_encoder = nn.GRU(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            batch_first=True,
        )
        self.output_projection = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.output_dim),
            nn.LayerNorm(self.output_dim),
        )

    def forward(
            self,
            trajectories: torch.Tensor,
            last_pos: torch.Tensor,
    ) -> torch.Tensor:
        """Return full-trajectory embeddings with shape ``[N,K,output_dim]``."""
        _validate_trajectory_bank(trajectories, last_pos)
        num_agents, num_candidates, num_steps, _ = trajectories.shape
        velocity = _future_velocity(trajectories, last_pos, self.dt)
        previous_velocity = torch.cat((
            torch.zeros_like(velocity[:, :, :1]), velocity[:, :, :-1]), dim=2)
        acceleration = (velocity - previous_velocity) / self.dt
        displacement = trajectories - last_pos[:, None, None, :]
        normalized_time = torch.linspace(
            1.0 / num_steps, 1.0, num_steps,
            dtype=trajectories.dtype, device=trajectories.device,
        ).view(1, 1, num_steps, 1).expand(
            num_agents, num_candidates, -1, -1)
        step_feature = torch.cat((
            displacement, velocity, acceleration, normalized_time), dim=-1)
        projected = self.step_projection(
            step_feature.reshape(num_agents * num_candidates, num_steps, 7))
        _, final_hidden = self.temporal_encoder(projected)
        encoded = self.output_projection(final_hidden[-1])
        return encoded.reshape(num_agents, num_candidates, self.output_dim)


def _reverse_edge_context(edge_feat: torch.Tensor) -> torch.Tensor:
    """Reverse known graph geometry; generic learned contexts stay unchanged."""
    if edge_feat.shape[-1] == EDGE_FEATURE_DIM:
        return reverse_edge_features(edge_feat)
    return edge_feat


class CandidateConditionedRelation(nn.Module):
    """Refine an observation-only relation prior for each trajectory pair.

    The learned residual is evaluated in both endpoint orientations with a
    shared MLP and averaged, ensuring that reindexing agents only transposes
    the candidate-pair axes.  Disabling ``trajectory_conditioned`` exactly
    expands the prior ``q0`` and yields zero prior KL, providing a clean
    edge-level-relation ablation.
    """

    def __init__(
            self,
            trajectory_dim: int,
            edge_dim: int,
            num_relation_modes: int = 4,
            geometry_dim: int = TRAJECTORY_PAIR_GEOMETRY_DIM,
            hidden_dim: int = 128,
            trajectory_conditioned: bool = True,
            temperature: float = 1.0,
            eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if trajectory_dim <= 0 or edge_dim < 0 or geometry_dim <= 0:
            raise ValueError("invalid trajectory/edge/geometry dimension")
        if num_relation_modes <= 0 or hidden_dim <= 0:
            raise ValueError("relation modes and hidden_dim must be positive")
        if temperature <= 0 or eps <= 0:
            raise ValueError("temperature and eps must be positive")
        self.trajectory_dim = int(trajectory_dim)
        self.edge_dim = int(edge_dim)
        self.num_relation_modes = int(num_relation_modes)
        self.geometry_dim = int(geometry_dim)
        self.hidden_dim = int(hidden_dim)
        self.trajectory_conditioned = bool(trajectory_conditioned)
        self.temperature = float(temperature)
        self.eps = float(eps)
        input_dim = 2 * self.trajectory_dim + self.edge_dim + self.geometry_dim
        self.residual_mlp = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.num_relation_modes),
        )
        # Start as the observation relation prior, while tiny non-zero weights
        # avoid identical mode gradients once task learning begins.
        nn.init.normal_(self.residual_mlp[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.residual_mlp[-1].bias)

    def _validate(
            self,
            trajectory_feat: torch.Tensor,
            edge_index: torch.Tensor,
            edge_feat: torch.Tensor,
            relation_prior: Optional[torch.Tensor],
            pair_geometry: torch.Tensor,
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
        if edge_feat.device != trajectory_feat.device or \
                pair_geometry.device != trajectory_feat.device:
            raise ValueError("all relation inputs must share a device")
        if edge_feat.dtype != trajectory_feat.dtype or \
                pair_geometry.dtype != trajectory_feat.dtype:
            raise ValueError("all floating relation inputs must share a dtype")
        if not torch.isfinite(edge_feat).all() or not torch.isfinite(
                pair_geometry).all():
            raise ValueError("edge_feat and pair_geometry must be finite")

        if relation_prior is None:
            prior = trajectory_feat.new_full(
                (edge_count, self.num_relation_modes),
                1.0 / self.num_relation_modes)
        else:
            if relation_prior.shape != (edge_count, self.num_relation_modes):
                raise ValueError("relation_prior must have shape [E,M]")
            if relation_prior.device != trajectory_feat.device or \
                    relation_prior.dtype != trajectory_feat.dtype:
                raise ValueError("relation_prior must share input device and dtype")
            if not torch.isfinite(relation_prior).all() or bool(
                    (relation_prior < 0).any()):
                raise ValueError("relation_prior must be finite and non-negative")
            prior_sum = relation_prior.sum(dim=-1, keepdim=True)
            uniform = torch.full_like(
                relation_prior, 1.0 / self.num_relation_modes)
            prior = torch.where(
                prior_sum > self.eps,
                relation_prior / prior_sum.clamp_min(self.eps), uniform)
        return prior, pair_geometry

    def forward(
            self,
            trajectory_feat: torch.Tensor,
            edge_index: torch.Tensor,
            edge_feat: torch.Tensor,
            relation_prior: Optional[torch.Tensor],
            pair_geometry: torch.Tensor,
            scene_index: Optional[torch.Tensor] = None,
            trajectory_conditioned: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        """Return posterior/residual/logits and ``KL(q_ab || q0)``.

        Shapes are posterior/logits ``[E,K,K,M]``, edge prior ``[E,M]``,
        pointwise KL ``[E,K,K]``, and scalar ``relation_prior_kl``.
        """
        prior, pair_geometry = self._validate(
            trajectory_feat, edge_index, edge_feat, relation_prior,
            pair_geometry, scene_index)
        edge_count = edge_index.shape[1]
        num_candidates = trajectory_feat.shape[1]
        expanded_prior = prior[:, None, None, :].expand(
            -1, num_candidates, num_candidates, -1)
        use_conditioning = (
            self.trajectory_conditioned if trajectory_conditioned is None
            else bool(trajectory_conditioned))

        if not use_conditioning or edge_count == 0:
            residual_logits = trajectory_feat.new_zeros((
                edge_count, num_candidates, num_candidates,
                self.num_relation_modes))
        else:
            source, destination = edge_index
            source_hidden = trajectory_feat[source][:, :, None, :].expand(
                -1, -1, num_candidates, -1)
            destination_hidden = trajectory_feat[destination][:, None, :, :].expand(
                -1, num_candidates, -1, -1)
            expanded_edge = edge_feat[:, None, None, :].expand(
                -1, num_candidates, num_candidates, -1)
            forward_input = torch.cat((
                source_hidden, destination_hidden, expanded_edge,
                pair_geometry), dim=-1)

            reverse_edge = _reverse_edge_context(edge_feat)
            expanded_reverse_edge = reverse_edge[:, None, None, :].expand(
                -1, num_candidates, num_candidates, -1)
            reverse_geometry = (
                reverse_trajectory_pair_geometry(pair_geometry)
                if self.geometry_dim == TRAJECTORY_PAIR_GEOMETRY_DIM
                else pair_geometry)
            reverse_input = torch.cat((
                destination_hidden, source_hidden, expanded_reverse_edge,
                reverse_geometry), dim=-1)
            residual_logits = 0.5 * (
                self.residual_mlp(forward_input) +
                self.residual_mlp(reverse_input))

        prior_logits = expanded_prior.clamp_min(self.eps).log()
        relation_logits = prior_logits + residual_logits
        # The off ablation must be *exactly* q0 expansion, independent of the
        # candidate-posterior temperature.  Temperature only controls the
        # learned candidate-conditioned distribution.
        posterior = (
            torch.softmax(relation_logits / self.temperature, dim=-1)
            if use_conditioning else expanded_prior)
        pointwise_kl = (
            posterior * (
                posterior.clamp_min(self.eps).log() - prior_logits)
        ).sum(dim=-1)
        # Empty-edge batches contribute a differentiable zero to aggregated
        # losses without producing mean(empty)=NaN.
        relation_prior_kl = (
            pointwise_kl.mean() if pointwise_kl.numel()
            else trajectory_feat.sum() * 0.0)
        if not torch.isfinite(posterior).all() or not torch.isfinite(
                relation_prior_kl):
            raise FloatingPointError("candidate relation posterior is non-finite")
        return {
            # Short semantic names are convenient for the V4 wrapper; the
            # explicit aliases below remain self-documenting in diagnostics.
            "posterior": posterior,
            "residual": residual_logits,
            "logits": relation_logits,
            "prior_kl": relation_prior_kl,
            "relation_posterior": posterior,
            "relation_prob": posterior,
            "residual_logits": residual_logits,
            "relation_logits": relation_logits,
            "relation_prior": prior,
            "relation_prior_expanded": expanded_prior,
            "relation_prior_kl_pointwise": pointwise_kl,
            "relation_prior_kl": relation_prior_kl,
        }


# Explicit aliases keep integration code and ablation configs descriptive.
CompleteTrajectoryEncoder = TrajectoryEncoder
CandidateConditionedRelationInference = CandidateConditionedRelation
TrajectoryPairRelation = CandidateConditionedRelation
build_pair_geometry = build_trajectory_pair_geometry


__all__ = [
    "CandidateConditionedRelation",
    "CandidateConditionedRelationInference",
    "CompleteTrajectoryEncoder",
    "TRAJECTORY_PAIR_GEOMETRY_DIM",
    "TRAJECTORY_PAIR_GEOMETRY_NAMES",
    "TRAJECTORY_PAIR_GEOMETRY_REVERSE_SIGN_INDICES",
    "TrajectoryEncoder",
    "TrajectoryPairRelation",
    "build_pair_geometry",
    "build_trajectory_pair_geometry",
    "reverse_trajectory_pair_geometry",
]
