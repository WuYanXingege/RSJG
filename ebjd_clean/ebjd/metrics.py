"""World-preserving metre-space marginal and joint metrics."""

from __future__ import annotations

from dataclasses import dataclass

import torch


def _softmin(values: torch.Tensor, temperature: float, dim: int) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return -float(temperature) * torch.logsumexp(-values / float(temperature), dim=dim)


def per_world_errors(
    predictions: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ADE/FDE errors as [B,N,P], once, in input metric units."""
    if predictions.ndim != 5 or predictions.shape[-2:] != (12, 2):
        raise ValueError("predictions must be [B,P,N,12,2]")
    if targets.shape != predictions.shape[0:1] + predictions.shape[2:]:
        raise ValueError("targets must be [B,N,12,2]")
    distance = torch.linalg.vector_norm(
        predictions.float() - targets[:, None].float(), dim=-1)
    ade = distance.mean(-1).permute(0, 2, 1)
    fde = distance[..., -1].permute(0, 2, 1)
    mask = valid[..., None].bool()
    return ade.masked_fill(~mask, 0), fde.masked_fill(~mask, 0)


def scene_soft_metrics(
    predictions: torch.Tensor,
    goals: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del goals  # literal endpoint is already predictions[..., -1, :]
    ade, fde = per_world_errors(predictions, targets, valid)
    counts = valid.sum(1).clamp_min(1).float()
    marginal_ade = (_softmin(ade, temperature, 2) * valid).sum(1) / counts
    marginal_fde = (_softmin(fde, temperature, 2) * valid).sum(1) / counts
    world_ade = (ade * valid[..., None]).sum(1) / counts[:, None]
    world_fde = (fde * valid[..., None]).sum(1) / counts[:, None]
    joint_ade = _softmin(world_ade, temperature, 1)
    joint_fde = _softmin(world_fde, temperature, 1)
    return tuple(x.mean() for x in (
        marginal_ade, marginal_fde, joint_ade, joint_fde))


@dataclass(frozen=True)
class MetricSummary:
    minADE: float
    minFDE: float
    JADE: float
    JFDE: float
    coordination_gap_ADE: float
    coordination_gap_FDE: float


def hard_metrics(
    predictions: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor
) -> MetricSummary:
    ade, fde = per_world_errors(predictions, targets, valid)
    counts = valid.sum(1).clamp_min(1).float()
    marginal_ade = (ade.min(2).values * valid).sum(1) / counts
    marginal_fde = (fde.min(2).values * valid).sum(1) / counts
    world_ade = (ade * valid[..., None]).sum(1) / counts[:, None]
    world_fde = (fde * valid[..., None]).sum(1) / counts[:, None]
    joint_ade = world_ade.min(1).values
    joint_fde = world_fde.min(1).values
    ma, mf, ja, jf = [float(x.mean()) for x in (
        marginal_ade, marginal_fde, joint_ade, joint_fde)]
    return MetricSummary(ma, mf, ja, jf, ja - ma, jf - mf)


def packed_metric_adapter(
    predictions: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Export each scene as legacy-compatible [P,20,A,2]/[20,A,2]."""
    packed = []
    for scene in range(predictions.shape[0]):
        mask = valid[scene]
        pred = predictions[scene, :, mask].permute(0, 2, 1, 3)
        truth = targets[scene, mask].permute(1, 0, 2)
        observed_padding = torch.zeros(
            pred.shape[0], 8, pred.shape[2], 2, device=pred.device, dtype=pred.dtype)
        truth_padding = torch.zeros(8, truth.shape[1], 2, device=truth.device, dtype=truth.dtype)
        packed.append((torch.cat((observed_padding, pred), 1),
                       torch.cat((truth_padding, truth), 0)))
    return packed
