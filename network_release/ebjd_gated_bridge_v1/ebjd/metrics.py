"""World-preserving metre-space marginal and joint metric statistics."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch


def _softmin(values: torch.Tensor, temperature: float, dim: int) -> torch.Tensor:
    """Normalized soft minimum: zero errors map to exactly zero."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    worlds = values.shape[dim]
    return -float(temperature) * (
        torch.logsumexp(-values / float(temperature), dim=dim) - math.log(worlds))


def per_world_errors(
    predictions: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ADE/FDE errors as [B,N,P], once, in input metric units."""
    if predictions.ndim != 5 or predictions.shape[-2:] != (12, 2):
        raise ValueError("predictions must be [B,P,N,12,2]")
    if targets.shape != predictions.shape[0:1] + predictions.shape[2:]:
        raise ValueError("targets must be [B,N,12,2]")
    if valid.shape != predictions.shape[:1] + predictions.shape[2:3]:
        raise ValueError("valid must be [B,N]")
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
    """Training risks remain scene-weighted by design."""
    del goals
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
class BatchMetricStatistics:
    agent_minade: torch.Tensor
    agent_minfde: torch.Tensor
    agent_valid: torch.Tensor
    scene_jade: torch.Tensor
    scene_jfde: torch.Tensor
    scene_marginal_ade: torch.Tensor
    scene_marginal_fde: torch.Tensor


def hard_metric_statistics(
    predictions: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor
) -> BatchMetricStatistics:
    """Expose unaggregated values so batching/sharding cannot alter results."""
    ade, fde = per_world_errors(predictions, targets, valid)
    counts = valid.sum(1).clamp_min(1).float()
    agent_ade = ade.min(2).values
    agent_fde = fde.min(2).values
    scene_marginal_ade = (agent_ade * valid).sum(1) / counts
    scene_marginal_fde = (agent_fde * valid).sum(1) / counts
    scene_jade = ((ade * valid[..., None]).sum(1) / counts[:, None]).min(1).values
    scene_jfde = ((fde * valid[..., None]).sum(1) / counts[:, None]).min(1).values
    return BatchMetricStatistics(
        agent_ade, agent_fde, valid.bool(), scene_jade, scene_jfde,
        scene_marginal_ade, scene_marginal_fde)


@dataclass(frozen=True)
class MetricSummary:
    minADE: float
    minFDE: float
    JADE: float
    JFDE: float
    scene_weighted_minADE: float
    scene_weighted_minFDE: float
    coordination_gap_ADE: float
    coordination_gap_FDE: float
    agent_count: int
    scene_count: int
    sums: dict[str, float]


class MetricAccumulator:
    """Aggregate formal marginals by agents and joint/gaps by scenes."""

    def __init__(self) -> None:
        self.agent_ade_sum = 0.0
        self.agent_fde_sum = 0.0
        self.agent_count = 0
        self.scene_jade_sum = 0.0
        self.scene_jfde_sum = 0.0
        self.scene_marginal_ade_sum = 0.0
        self.scene_marginal_fde_sum = 0.0
        self.scene_count = 0

    def update_statistics(self, statistics: BatchMetricStatistics) -> None:
        valid = statistics.agent_valid
        self.agent_ade_sum += float(statistics.agent_minade[valid].double().sum())
        self.agent_fde_sum += float(statistics.agent_minfde[valid].double().sum())
        self.agent_count += int(valid.sum())
        self.scene_jade_sum += float(statistics.scene_jade.double().sum())
        self.scene_jfde_sum += float(statistics.scene_jfde.double().sum())
        self.scene_marginal_ade_sum += float(statistics.scene_marginal_ade.double().sum())
        self.scene_marginal_fde_sum += float(statistics.scene_marginal_fde.double().sum())
        self.scene_count += int(statistics.scene_jade.numel())

    def update(self, predictions: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor) -> None:
        self.update_statistics(hard_metric_statistics(predictions, targets, valid))

    def summary(self) -> MetricSummary:
        if self.agent_count == 0 or self.scene_count == 0:
            raise ValueError("metric accumulator is empty")
        minade = self.agent_ade_sum / self.agent_count
        minfde = self.agent_fde_sum / self.agent_count
        jade = self.scene_jade_sum / self.scene_count
        jfde = self.scene_jfde_sum / self.scene_count
        scene_ade = self.scene_marginal_ade_sum / self.scene_count
        scene_fde = self.scene_marginal_fde_sum / self.scene_count
        return MetricSummary(
            minade, minfde, jade, jfde, scene_ade, scene_fde,
            max(0.0, jade - scene_ade), max(0.0, jfde - scene_fde),
            self.agent_count, self.scene_count,
            {
                "agent_minADE": self.agent_ade_sum,
                "agent_minFDE": self.agent_fde_sum,
                "scene_JADE": self.scene_jade_sum,
                "scene_JFDE": self.scene_jfde_sum,
                "scene_marginal_ADE": self.scene_marginal_ade_sum,
                "scene_marginal_FDE": self.scene_marginal_fde_sum,
            })

    def as_dict(self) -> dict:
        return asdict(self.summary())


def hard_metrics(
    predictions: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor
) -> MetricSummary:
    accumulator = MetricAccumulator()
    accumulator.update(predictions, targets, valid)
    return accumulator.summary()


def packed_metric_adapter(
    predictions: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor,
    observed: torch.Tensor | None = None,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Export each scene as legacy-compatible [P,20,A,2]/[20,A,2]."""
    packed = []
    if observed is None:
        observed = torch.zeros(
            targets.shape[0], targets.shape[1], 8, 2,
            device=targets.device, dtype=targets.dtype)
    for scene in range(predictions.shape[0]):
        mask = valid[scene]
        pred = predictions[scene, :, mask].permute(0, 2, 1, 3)
        truth = targets[scene, mask].permute(1, 0, 2)
        history = observed[scene, mask].permute(1, 0, 2)
        packed.append((
            torch.cat((history[None].expand(pred.shape[0], -1, -1, -1), pred), 1),
            torch.cat((history, truth), 0)))
    return packed
