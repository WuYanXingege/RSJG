"""EBJD diffusion, geometry, map and free-rollout objectives."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from .encoders import EBJDContext
from .metrics import scene_soft_metrics


def _masked_mean(value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    mask = valid
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(value).to(value.dtype)
    return (value * mask).sum() / mask.sum().clamp_min(1)


def diffusion_loss(
    final: torch.Tensor, coarse: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    def split_loss(prediction: torch.Tensor) -> torch.Tensor:
        goal = _masked_mean((prediction[..., 11, :] - target[..., 11, :]).square(), valid[:, None])
        bridge = _masked_mean((prediction[..., :11, :] - target[..., :11, :]).square(), valid[:, None])
        return goal + bridge
    return split_loss(final) + 0.25 * split_loss(coarse)


def supervision_edges(context: EBJDContext) -> torch.Tensor:
    position = context.observed[..., -1, :]
    rel_p = position[:, :, None] - position[:, None, :]
    rel_v = context.mean_velocity[:, :, None] - context.mean_velocity[:, None, :]
    distance = torch.linalg.vector_norm(rel_p, dim=-1)
    cpa_t = (-(rel_p * rel_v).sum(-1) / rel_v.square().sum(-1).clamp_min(1e-6)).clamp(0, 4.8)
    cpa_distance = torch.linalg.vector_norm(rel_p + cpa_t[..., None] * rel_v, dim=-1)
    valid_pair = context.valid[:, :, None] & context.valid[:, None, :]
    eye = torch.eye(position.shape[1], device=position.device, dtype=torch.bool)[None]
    return valid_pair & ~eye & ((distance <= 6) | (cpa_distance <= 2))


def relative_motion_loss(
    prediction: torch.Tensor, target: torch.Tensor, context: EBJDContext
) -> torch.Tensor:
    """Per-scene relative position/velocity smooth-L1, zero for no edges."""
    if prediction.ndim == 5:
        prediction = prediction.mean(1)
    edges = supervision_edges(context)
    pred_velocity = torch.empty_like(prediction)
    true_velocity = torch.empty_like(target)
    pred_velocity[..., 0, :] = (
        prediction[..., 0, :] - context.observed[..., -1, :]) / 0.4
    true_velocity[..., 0, :] = (
        target[..., 0, :] - context.observed[..., -1, :]) / 0.4
    pred_velocity[..., 1:, :] = (prediction[..., 1:, :] - prediction[..., :-1, :]) / 0.4
    true_velocity[..., 1:, :] = (target[..., 1:, :] - target[..., :-1, :]) / 0.4
    rel_position_error = (
        prediction[:, :, None] - prediction[:, None, :]
        - target[:, :, None] + target[:, None, :])
    rel_velocity_error = (
        pred_velocity[:, :, None] - pred_velocity[:, None, :]
        - true_velocity[:, :, None] + true_velocity[:, None, :])
    pos = F.smooth_l1_loss(rel_position_error, torch.zeros_like(rel_position_error), reduction="none").mean((-1, -2))
    vel = F.smooth_l1_loss(rel_velocity_error, torch.zeros_like(rel_velocity_error), reduction="none").mean((-1, -2))
    scene_values = []
    for index in range(prediction.shape[0]):
        if edges[index].any():
            scene_values.append((pos[index][edges[index]] + 0.5 * vel[index][edges[index]]).mean())
        else:
            scene_values.append(prediction[index].sum() * 0)
    return torch.stack(scene_values).mean()


def future_heatmap_targets(target: torch.Tensor, context: EBJDContext, sigma_m: float = 0.5) -> torch.Tensor:
    height, width = context.map_logits.shape[-2:]
    relative = target - context.map_center[..., None, :]
    gx = torch.linspace(-1, 1, width, device=target.device)
    gy = torch.linspace(-1, 1, height, device=target.device)
    yy, xx = torch.meshgrid(gy, gx, indexing="ij")
    x = relative[..., 0] / (context.crop_width_m / 2)
    y = relative[..., 1] / (context.crop_width_m / 2)
    sigma = sigma_m / (context.crop_width_m / 2)
    heatmap = torch.exp(-0.5 * (
        (xx - x[..., None, None]) ** 2 + (yy - y[..., None, None]) ** 2) / sigma ** 2)
    center_in_crop = ((x.abs() <= 1) & (y.abs() <= 1))[..., None, None]
    return heatmap * center_in_crop


def map_loss(target: torch.Tensor, context: EBJDContext) -> torch.Tensor:
    heatmap = future_heatmap_targets(target.float(), context)
    raw = F.binary_cross_entropy_with_logits(
        context.map_logits.float(), heatmap, reduction="none")
    return _masked_mean(raw, context.valid)


def rollout_loss(
    prediction: torch.Tensor,
    goal: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    metrics = scene_soft_metrics(prediction, goal, target, valid, temperature)
    marginal_ade, marginal_fde, joint_ade, joint_fde = metrics
    loss = joint_ade + 0.5 * joint_fde + marginal_ade + 0.5 * marginal_fde
    return loss, metrics
