"""JMM collision metrics for synchronized joint trajectory worlds.

This module follows the released ICCV 2023 ``joint-metrics-matter`` evaluator
at commit 894182b5ea7dac3c93bbe0bcc18c156c932398b7.  In particular, collision
distance is the closest approach of the *relative motion* over each future
time interval, not the distance between two ordinary spatial line segments.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SceneCollisionStatistics:
    """Unaggregated statistics for one scene."""

    collision: torch.Tensor  # [K,N] bool
    jade_world_error: torch.Tensor  # [K]
    jade_world_index: int
    cr_jade: float
    cr_mean: float
    collision_numerator_jade: int
    collision_denominator_jade: int
    collision_numerator_mean: int
    collision_denominator_mean: int


def _closest_origin_distance(start: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
    """Distance from the origin to relative-motion segments ``[...,2]``.

    This is algebraically equivalent to the official NumPy ``_lineseg_dist``
    for two-dimensional relative trajectories, including zero-length motion.
    """

    delta = end - start
    length_sq = (delta * delta).sum(dim=-1)
    safe = length_sq.clamp_min(torch.finfo(start.dtype).tiny)
    fraction = (-(start * delta).sum(dim=-1) / safe).clamp(0.0, 1.0)
    closest = start + fraction[..., None] * delta
    distance = torch.linalg.vector_norm(closest, dim=-1)
    return torch.where(
        length_sq > 0,
        distance,
        torch.linalg.vector_norm(start, dim=-1),
    )


def collision_flags(
    prediction: torch.Tensor,
    valid: torch.Tensor | None = None,
    threshold_m: float = 0.2,
) -> torch.Tensor:
    """Return JMM collision flags for ``prediction`` shaped ``[K,N,T,2]``.

    The threshold comparison is strict (``distance < threshold_m``), matching
    the official implementation.  Only the twelve future positions and their
    eleven intervening segments are considered; no observation-to-future
    segment is inserted.
    """

    if prediction.ndim != 4 or prediction.shape[-1] != 2:
        raise ValueError("prediction must be [K,N,T,2]")
    worlds, agents, timesteps, _ = prediction.shape
    if timesteps < 1:
        raise ValueError("prediction must contain at least one future timestep")
    if valid is None:
        valid = torch.ones(agents, dtype=torch.bool, device=prediction.device)
    if valid.shape != (agents,):
        raise ValueError("valid must be [N]")
    if not torch.isfinite(prediction).all():
        raise ValueError("prediction contains non-finite coordinates")
    if threshold_m <= 0:
        raise ValueError("threshold_m must be positive")

    valid = valid.bool()
    flags = torch.zeros(worlds, agents, dtype=torch.bool, device=prediction.device)
    if int(valid.sum()) < 2:
        return flags

    indices = valid.nonzero(as_tuple=False).flatten()
    trajectories = prediction.index_select(1, indices).float()
    count = trajectories.shape[1]
    left, right = torch.triu_indices(count, count, offset=1, device=prediction.device)
    relative = trajectories[:, left] - trajectories[:, right]  # [K,pairs,T,2]
    collided = torch.linalg.vector_norm(relative[..., 0, :], dim=-1) < threshold_m
    if timesteps > 1:
        segment_distance = _closest_origin_distance(
            relative[..., :-1, :], relative[..., 1:, :])
        collided = collided | (segment_distance < threshold_m).any(dim=-1)
    local = torch.zeros(worlds, count, dtype=torch.bool, device=prediction.device)
    # A pedestrian may occur in several pairs.  Explicit OR reduction avoids
    # the overwrite semantics of scatter when duplicate indices are present.
    for agent in range(count):
        pair_mask = (left == agent) | (right == agent)
        local[:, agent] = collided[:, pair_mask].any(dim=1)
    flags[:, indices] = local
    return flags


def scene_collision_statistics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor | None = None,
    threshold_m: float = 0.2,
) -> SceneCollisionStatistics:
    """Compute official-code CRJADE and CRmean for one scene.

    The common CRJADE world minimizes mean Euclidean displacement over valid
    agents and future timesteps.  ``torch.argmin`` supplies the deterministic
    first-index tie break used by NumPy ``argmin`` in the released evaluator.
    """

    if prediction.ndim != 4 or prediction.shape[-1] != 2:
        raise ValueError("prediction must be [K,N,T,2]")
    _, agents, timesteps, _ = prediction.shape
    if target.shape != (agents, timesteps, 2):
        raise ValueError("target must be [N,T,2]")
    if valid is None:
        valid = torch.ones(agents, dtype=torch.bool, device=prediction.device)
    if valid.shape != (agents,) or int(valid.sum()) == 0:
        raise ValueError("valid must identify at least one agent")
    if not torch.isfinite(target).all():
        raise ValueError("target contains non-finite coordinates")

    valid = valid.bool()
    flags = collision_flags(prediction, valid, threshold_m)
    distance = torch.linalg.vector_norm(
        prediction.float() - target[None].float(), dim=-1)
    world_error = distance[:, valid].mean(dim=(1, 2))
    world_index = int(torch.argmin(world_error).item())
    valid_count = int(valid.sum())
    jade_numerator = int(flags[world_index, valid].sum())
    mean_numerator = int(flags[:, valid].sum())
    mean_denominator = prediction.shape[0] * valid_count
    return SceneCollisionStatistics(
        collision=flags,
        jade_world_error=world_error,
        jade_world_index=world_index,
        cr_jade=jade_numerator / valid_count,
        cr_mean=mean_numerator / mean_denominator,
        collision_numerator_jade=jade_numerator,
        collision_denominator_jade=valid_count,
        collision_numerator_mean=mean_numerator,
        collision_denominator_mean=mean_denominator,
    )


def ground_truth_collision_rate(
    target: torch.Tensor,
    valid: torch.Tensor | None = None,
    threshold_m: float = 0.2,
) -> tuple[float, int, int]:
    """Return the JMM collision fraction for one ground-truth future."""

    if target.ndim != 3 or target.shape[-1] != 2:
        raise ValueError("target must be [N,T,2]")
    if valid is None:
        valid = torch.ones(target.shape[0], dtype=torch.bool, device=target.device)
    flags = collision_flags(target[None], valid, threshold_m)[0]
    denominator = int(valid.sum())
    if denominator == 0:
        raise ValueError("valid must identify at least one agent")
    numerator = int(flags[valid].sum())
    return numerator / denominator, numerator, denominator
