"""EBJD diffusion, geometry, map and free-rollout objectives."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from .encoders import EBJDContext
from .metrics import scene_soft_metrics
from .representation import fixed_endpoint_trajectory


def _masked_mean(value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    mask = valid
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(value).to(value.dtype)
    return (value * mask).sum() / mask.sum().clamp_min(1)


def diffusion_loss(
    final: torch.Tensor, coarse: torch.Tensor, target: torch.Tensor,
    valid: torch.Tensor, coarse_weight: float = 0.25,
    goal_weight: float = 1.0, return_components: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Weighted endpoint/bridge velocity loss with unweighted diagnostics.

    The ``goal_weight == 1`` branch deliberately preserves the historical
    floating-point operation order.  Endpoint and bridge terms keep their
    separate masked means, so two endpoint coordinates are not diluted by the
    22 bridge coordinates.
    """
    def split_loss(
        prediction: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        goal = _masked_mean((prediction[..., 11, :] - target[..., 11, :]).square(), valid[:, None])
        bridge = _masked_mean((prediction[..., :11, :] - target[..., :11, :]).square(), valid[:, None])
        return goal, bridge

    fine_goal, fine_bridge = split_loss(final)
    coarse_goal, coarse_bridge = split_loss(coarse)
    if float(goal_weight) == 1.0:
        loss = (fine_goal + fine_bridge) + float(coarse_weight) * (
            coarse_goal + coarse_bridge)
    else:
        loss = (
            float(goal_weight) * fine_goal + fine_bridge
            + float(coarse_weight) * (
                float(goal_weight) * coarse_goal + coarse_bridge))
    if not return_components:
        return loss
    return loss, {
        "goal_fine_unweighted": fine_goal,
        "bridge_fine_unweighted": fine_bridge,
        "goal_coarse_unweighted": coarse_goal,
        "bridge_coarse_unweighted": coarse_bridge,
    }


def fixed_anchor_path_loss(
    final_trajectory: torch.Tensor,
    coarse_trajectory: torch.Tensor,
    target: torch.Tensor,
    anchors: torch.Tensor,
    valid: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """Return alpha-squared, valid-agent mean path loss after endpoint fixation."""
    if final_trajectory.shape != coarse_trajectory.shape:
        raise ValueError("final and coarse trajectories must have identical shape")
    if final_trajectory.ndim != 5 or final_trajectory.shape[1] != 1:
        raise ValueError("fixed-anchor path training requires [B,1,N,12,2]")
    if target.shape != final_trajectory.shape[:1] + final_trajectory.shape[2:]:
        raise ValueError("target must have shape [B,N,12,2]")
    if anchors.shape != final_trajectory.shape[:-2] + (2,):
        raise ValueError("anchors must have shape [B,1,N,2]")
    if valid.shape != target.shape[:2]:
        raise ValueError("valid must have shape [B,N]")
    alpha_scene = alpha.reshape(alpha.shape[0], -1)
    if alpha_scene.shape != (target.shape[0], 1):
        raise ValueError("alpha must contain exactly one scalar per scene")
    fixed_final = fixed_endpoint_trajectory(final_trajectory, anchors)
    fixed_coarse = fixed_endpoint_trajectory(coarse_trajectory, anchors)
    with torch.autocast(device_type=target.device.type, enabled=False):
        truth = target.float()[:, None, :, :11, :]
        fine = F.smooth_l1_loss(
            fixed_final[..., :11, :], truth, beta=1.0, reduction="none")
        coarse = F.smooth_l1_loss(
            fixed_coarse[..., :11, :], truth, beta=1.0, reduction="none")
        per_agent = (fine + 0.25 * coarse).mean(dim=(-1, -2)).squeeze(1)
        mask = valid.to(per_agent.dtype)
        gated = alpha_scene[:, 0].float().square()[:, None] * per_agent.float()
        return (gated * mask).sum() / mask.sum().clamp_min(1)


def supervision_edges(
    context: EBJDContext, radius_m: float = 6.0,
    cpa_radius_m: float = 2.0, cpa_horizon_s: float = 4.8,
) -> torch.Tensor:
    position = context.observed[..., -1, :]
    rel_p = position[:, :, None] - position[:, None, :]
    rel_v = context.mean_velocity[:, :, None] - context.mean_velocity[:, None, :]
    distance = torch.linalg.vector_norm(rel_p, dim=-1)
    cpa_t = (-(rel_p * rel_v).sum(-1) / rel_v.square().sum(-1).clamp_min(1e-6)).clamp(0, cpa_horizon_s)
    cpa_distance = torch.linalg.vector_norm(rel_p + cpa_t[..., None] * rel_v, dim=-1)
    valid_pair = context.valid[:, :, None] & context.valid[:, None, :]
    eye = torch.eye(position.shape[1], device=position.device, dtype=torch.bool)[None]
    return valid_pair & ~eye & (
        (distance <= radius_m) | (cpa_distance <= cpa_radius_m))


def relative_motion_loss(
    prediction: torch.Tensor, target: torch.Tensor, context: EBJDContext,
    radius_m: float = 6.0, cpa_radius_m: float = 2.0,
    cpa_horizon_s: float = 4.8,
) -> torch.Tensor:
    """Return [B] relative position/velocity losses; no-edge scenes are zero."""
    if prediction.ndim == 5:
        prediction = prediction.mean(1)
    edges = supervision_edges(context, radius_m, cpa_radius_m, cpa_horizon_s)
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
    return torch.stack(scene_values)


def gated_geometry_mean(
    per_scene_loss: torch.Tensor, alpha: torch.Tensor,
) -> torch.Tensor:
    """Apply the diffusion alpha-squared gate independently to each scene."""
    if per_scene_loss.ndim != 1:
        raise ValueError("per_scene_loss must be [B]")
    alpha = alpha.reshape(alpha.shape[0], -1)
    if alpha.shape[0] != per_scene_loss.shape[0] or alpha.shape[1] != 1:
        raise ValueError("alpha must contain exactly one scalar per scene")
    return (alpha[:, 0].float().square() * per_scene_loss.float()).mean()


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


def map_loss(
    target: torch.Tensor, context: EBJDContext, sigma_m: float = 0.5,
) -> torch.Tensor:
    heatmap = future_heatmap_targets(target.float(), context, sigma_m)
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
