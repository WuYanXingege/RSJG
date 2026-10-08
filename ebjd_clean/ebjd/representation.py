"""Invertible endpoint--Brownian-bridge trajectory representations."""

from __future__ import annotations

import torch
from torch import nn


def cosine_vp(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Cosine VP coefficients with literal terminal values."""
    t = torch.as_tensor(t, dtype=torch.float32)
    alpha = torch.cos(torch.pi * t / 2)
    sigma = torch.sin(torch.pi * t / 2)
    alpha = torch.where(t >= 1, torch.zeros_like(alpha), alpha)
    sigma = torch.where(t >= 1, torch.ones_like(sigma), sigma)
    alpha = torch.where(t <= 0, torch.ones_like(alpha), alpha)
    sigma = torch.where(t <= 0, torch.zeros_like(sigma), sigma)
    return alpha, sigma


def cv_baseline(observed: torch.Tensor, dt: float = 0.4) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the 12-frame constant-velocity baseline and recent mean velocity."""
    if observed.ndim != 4 or observed.shape[-2:] != (8, 2):
        raise ValueError("observed must have shape [B,N,8,2]")
    velocity_last = (observed[..., -1, :] - observed[..., -2, :]) / dt
    velocity_prev = (observed[..., -2, :] - observed[..., -3, :]) / dt
    mean_velocity = (velocity_last + velocity_prev) / 2
    tau = torch.arange(
        1, 13, device=observed.device, dtype=observed.dtype)
    baseline = observed[..., -1:, :] + (
        tau.view(1, 1, 12, 1) * dt * mean_velocity.unsqueeze(-2))
    return baseline, mean_velocity


class EndpointBridgeRepresentation(nn.Module):
    """Exact goal2 + bridge22 representation with Brownian bridge whitening."""

    def __init__(
        self,
        goal_scale: float = 1.0,
        bridge_scale: float = 1.0,
        pred_length: int = 12,
    ) -> None:
        super().__init__()
        if pred_length != 12:
            raise ValueError("EBJD v1 requires a 12-frame prediction horizon")
        if goal_scale <= 0 or bridge_scale <= 0:
            raise ValueError("representation scales must be positive")
        a = torch.arange(1, pred_length, dtype=torch.float64) / pred_length
        covariance = torch.minimum(a[:, None], a[None, :]) - a[:, None] * a[None, :]
        cholesky = torch.linalg.cholesky(covariance)
        self.register_buffer("a", a.float())
        self.register_buffer("bridge_cholesky", cholesky.float())
        self.register_buffer("goal_scale", torch.tensor(float(goal_scale)))
        self.register_buffer("bridge_scale", torch.tensor(float(bridge_scale)))

    def set_scales(self, goal_scale: float, bridge_scale: float) -> None:
        if goal_scale <= 0 or bridge_scale <= 0:
            raise ValueError("representation scales must be positive")
        self.goal_scale.fill_(max(float(goal_scale), 0.5))
        self.bridge_scale.fill_(max(float(bridge_scale), 0.1))

    @torch.no_grad()
    def fit_scales(
        self, targets: torch.Tensor, baseline: torch.Tensor, valid: torch.Tensor
    ) -> tuple[float, float]:
        """Fit fold-only scalar RMS values in FP64 and apply configured floors."""
        encoded_unscaled = self._encode_unscaled(targets.double(), baseline.double())
        mask = valid.bool()[..., None, None]
        goal = encoded_unscaled[..., -1:, :]
        bridge = encoded_unscaled[..., :11, :]
        goal_values = goal.expand_as(encoded_unscaled[..., -1:, :])[mask.expand_as(goal)]
        bridge_values = bridge[mask.expand_as(bridge)]
        sg = max(float(goal_values.square().mean().sqrt()), 0.5)
        sr = max(float(bridge_values.square().mean().sqrt()), 0.1)
        self.set_scales(sg, sr)
        return sg, sr

    def _encode_unscaled(
        self, targets: torch.Tensor, baseline: torch.Tensor
    ) -> torch.Tensor:
        with torch.autocast(device_type=targets.device.type, enabled=False):
            goal_delta = targets[..., -1, :] - baseline[..., -1, :]
            residual = (
                targets[..., :11, :] - baseline[..., :11, :]
                - self.a.to(targets).view(*([1] * (targets.ndim - 2)), 11, 1)
                * goal_delta.unsqueeze(-2)
            )
            flat = residual.movedim(-2, 0).reshape(11, -1)
            whitened = torch.linalg.solve_triangular(
                self.bridge_cholesky.to(flat), flat, upper=False)
            whitened = whitened.reshape(
                11, *residual.movedim(-2, 0).shape[1:]).movedim(0, -2)
            return torch.cat((whitened, goal_delta.unsqueeze(-2)), dim=-2)

    def encode_target(
        self, targets: torch.Tensor, baseline: torch.Tensor
    ) -> torch.Tensor:
        if targets.shape != baseline.shape or targets.shape[-2:] != (12, 2):
            raise ValueError("targets and baseline must have shape [...,12,2]")
        with torch.autocast(device_type=targets.device.type, enabled=False):
            unscaled = self._encode_unscaled(targets.float(), baseline.float())
            bridge = unscaled[..., :11, :] / self.bridge_scale.float()
            goal = unscaled[..., 11:, :] / self.goal_scale.float()
            return torch.cat((bridge, goal), dim=-2)

    def decode(
        self, latent: torch.Tensor, baseline: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode ``[...,N,12,2]`` latent with baseline ``[B,N,12,2]``."""
        if latent.shape[-2:] != (12, 2) or baseline.shape[-2:] != (12, 2):
            raise ValueError("latent and baseline must end in [12,2]")
        with torch.autocast(device_type=latent.device.type, enabled=False):
            baseline = baseline.float()
            while baseline.ndim < latent.ndim:
                baseline = baseline.unsqueeze(1)
            bridge_z = latent.float()[..., :11, :]
            goal_z = latent.float()[..., 11, :]
            goal_delta = self.goal_scale.float() * goal_z
            goal = baseline[..., -1, :] + goal_delta
            flat = bridge_z.movedim(-2, 0).reshape(11, -1)
            residual = torch.matmul(
                self.bridge_cholesky.float(), flat).reshape(
                    11, *bridge_z.movedim(-2, 0).shape[1:]).movedim(0, -2)
            residual = self.bridge_scale.float() * residual
            a = self.a.float().view(*([1] * (latent.ndim - 2)), 11, 1)
            first = baseline[..., :11, :] + a * goal_delta.unsqueeze(-2) + residual
            trajectory = torch.cat((first, goal.unsqueeze(-2)), dim=-2)
            return trajectory, goal


class CartesianVelocityRepresentation(nn.Module):
    """Capacity-matched velocity-latent ablation without bridge whitening."""

    def __init__(self, scale: float = 1.0, dt: float = 0.4) -> None:
        super().__init__()
        self.register_buffer("scale", torch.tensor(float(scale)))
        self.dt = float(dt)

    @property
    def goal_scale(self) -> torch.Tensor:
        return self.scale

    def encode_target(self, targets: torch.Tensor, baseline: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=targets.device.type, enabled=False):
            targets, baseline = targets.float(), baseline.float()
            base_step = baseline[..., 1:2, :] - baseline[..., :1, :]
            x0 = baseline[..., :1, :] - base_step
            previous = torch.cat((x0, targets[..., :-1, :]), dim=-2)
            velocities = (targets - previous) / self.dt
            baseline_previous = torch.cat((x0, baseline[..., :-1, :]), dim=-2)
            baseline_velocity = (baseline - baseline_previous) / self.dt
            return (velocities - baseline_velocity) / self.scale.float()

    def decode(self, latent: torch.Tensor, baseline: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.autocast(device_type=latent.device.type, enabled=False):
            baseline = baseline.float()
            while baseline.ndim < latent.ndim:
                baseline = baseline.unsqueeze(1)
            base_velocity = torch.cat((
                baseline[..., 1:2, :] - baseline[..., :1, :],
                baseline[..., 1:, :] - baseline[..., :-1, :],
            ), dim=-2) / self.dt
            velocity = base_velocity + self.scale.float() * latent.float()
            start = baseline[..., :1, :] - self.dt * base_velocity[..., :1, :]
            trajectory = start + self.dt * velocity.cumsum(dim=-2)
            return trajectory, trajectory[..., -1, :]
