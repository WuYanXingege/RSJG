"""History/map conditioning for the independent EBJD model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .layers import PreNormSocialBlock, sinusoidal_embedding
from .representation import cv_baseline
from .vendor import TrainableMapUNet


@dataclass
class EBJDContext:
    memory: torch.Tensor
    social_h: torch.Tensor
    observed_bias: torch.Tensor
    baseline: torch.Tensor
    mean_velocity: torch.Tensor
    map_features: torch.Tensor
    map_logits: torch.Tensor
    map_center: torch.Tensor
    valid: torch.Tensor
    observed: torch.Tensor
    crop_width_m: float


def _safe_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x.square().sum(dim=-1).add(eps).sqrt()


def pairwise_observed_geometry(
    observed: torch.Tensor, mean_velocity: torch.Tensor, goal_scale: torch.Tensor
) -> torch.Tensor:
    """Return ordered pair features [B,N,N,10], using only history."""
    with torch.autocast(device_type=observed.device.type, enabled=False):
        observed, mean_velocity = observed.float(), mean_velocity.float()
        position = observed[..., -1, :]
        rel_p = position[:, :, None] - position[:, None, :]
        rel_v = mean_velocity[:, :, None] - mean_velocity[:, None, :]
        distance = _safe_norm(rel_p)
        rel_speed2 = rel_v.square().sum(-1)
        closing = -(rel_p * rel_v).sum(-1) / distance
        cpa_t = (-(rel_p * rel_v).sum(-1) / rel_speed2.clamp_min(1e-6)).clamp(0, 4.8)
        cpa_p = rel_p + cpa_t[..., None] * rel_v
        cpa_d = _safe_norm(cpa_p)
        vi = mean_velocity[:, :, None].expand_as(rel_p)
        vj = mean_velocity[:, None, :].expand_as(rel_p)
        ni, nj = _safe_norm(vi), _safe_norm(vj)
        direction_valid = ((ni > 1e-3) & (nj > 1e-3)).float()
        cosine = (vi * vj).sum(-1) / (ni * nj).clamp_min(1e-6)
        cosine = cosine * direction_valid
        scale = goal_scale.float().clamp_min(0.5)
        return torch.cat((
            rel_p / scale,
            rel_v,
            (distance / scale)[..., None],
            closing[..., None],
            (cpa_t / 4.8)[..., None],
            (cpa_d / scale)[..., None],
            cosine[..., None],
            direction_valid[..., None],
        ), dim=-1).clamp(-10, 10)


class HistoryMapEncoder(nn.Module):
    def __init__(
        self,
        goal_scale: float = 1.0,
        crop_width_m: float = 32.0,
        map_input_channels: int = 14,
        dim: int = 128,
        map_agent_chunk: int = 8,
    ) -> None:
        super().__init__()
        self.crop_width_m = float(crop_width_m)
        self.register_buffer("goal_scale", torch.tensor(float(goal_scale)))
        if map_agent_chunk <= 0:
            raise ValueError("map_agent_chunk must be positive")
        self.map_agent_chunk = int(map_agent_chunk)
        self.activation_checkpointing = True
        self.map_unet = TrainableMapUNet(map_input_channels)
        self.history_stem = nn.Sequential(
            nn.Linear(6, dim), nn.LayerNorm(dim), nn.SiLU())
        self.history_gru = nn.GRU(dim, dim, batch_first=True)
        self.observed_geometry_mlp = nn.Sequential(
            nn.Linear(10, 64), nn.SiLU(), nn.Linear(64, 4))
        self.history_social = nn.ModuleList([
            PreNormSocialBlock(dim, 4, 512) for _ in range(2)])
        self.map_projection = nn.Linear(32, dim)
        self.map_coordinate_mlp = nn.Sequential(
            nn.Linear(2, 64), nn.SiLU(), nn.Linear(64, dim))
        self.history_type = nn.Parameter(torch.zeros(dim))
        self.map_type = nn.Parameter(torch.zeros(dim))
        self.memory_norm = nn.LayerNorm(dim)

    def _history_heatmaps(self, observed: torch.Tensor, height: int, width: int) -> torch.Tensor:
        center = observed[..., -1:, :]
        relative = observed - center
        x = relative[..., 0] / (self.crop_width_m / 2)
        y = relative[..., 1] / (self.crop_width_m / 2)
        gx = torch.linspace(-1, 1, width, device=observed.device, dtype=observed.dtype)
        gy = torch.linspace(-1, 1, height, device=observed.device, dtype=observed.dtype)
        yy, xx = torch.meshgrid(gy, gx, indexing="ij")
        sigma = 0.5 / (self.crop_width_m / 2)
        squared = ((xx - x[..., None, None]) ** 2 + (yy - y[..., None, None]) ** 2)
        return torch.exp(-0.5 * squared / (sigma * sigma))

    def _map_input(self, observed: torch.Tensor, semantic_maps: torch.Tensor) -> torch.Tensor:
        if semantic_maps.ndim != 5 or semantic_maps.shape[:2] != observed.shape[:2]:
            raise ValueError("semantic_maps must have shape [B,N,C,H,W]")
        channels = semantic_maps.shape[2]
        if channels == 14:
            return semantic_maps.float()
        if channels != 6:
            raise ValueError("semantic_maps must contain 6 semantics or complete 14 channels")
        heatmaps = self._history_heatmaps(
            observed.float(), semantic_maps.shape[-2], semantic_maps.shape[-1])
        return torch.cat((semantic_maps.float(), heatmaps), dim=2)

    def forward(
        self, observed: torch.Tensor, semantic_maps: torch.Tensor, valid: torch.Tensor
    ) -> EBJDContext:
        if observed.ndim != 4 or observed.shape[-2:] != (8, 2):
            raise ValueError("observed must be [B,N,8,2]")
        batch, agents = observed.shape[:2]
        if valid.shape != (batch, agents) or not valid.any(dim=1).all():
            raise ValueError("valid must be [B,N] with at least one agent per scene")
        observed = observed.float()
        valid = valid.bool()
        baseline, mean_velocity = cv_baseline(observed)
        velocity = torch.zeros_like(observed)
        velocity[..., 1:, :] = (observed[..., 1:, :] - observed[..., :-1, :]) / 0.4
        velocity[..., 0, :] = velocity[..., 1, :]
        acceleration = torch.zeros_like(velocity)
        acceleration[..., 1:, :] = (velocity[..., 1:, :] - velocity[..., :-1, :]) / 0.4
        features = torch.cat((
            (observed - observed[..., -1:, :]) / self.goal_scale.clamp_min(0.5),
            velocity,
            acceleration,
        ), dim=-1)
        hist_tokens, hidden = self.history_gru(
            self.history_stem(features).reshape(batch * agents, 8, -1))
        hist_tokens = hist_tokens.reshape(batch, agents, 8, -1)
        social_h = hidden[-1].reshape(batch, agents, -1)
        geometry = pairwise_observed_geometry(observed, mean_velocity, self.goal_scale)
        observed_bias = 2 * torch.tanh(self.observed_geometry_mlp(geometry))
        observed_bias = observed_bias.permute(0, 3, 1, 2).contiguous()
        pair_valid = valid[:, None, :, None] & valid[:, None, None, :]
        observed_bias = observed_bias * pair_valid
        for block in self.history_social:
            social_h = block(social_h, observed_bias, valid)

        map_input = self._map_input(observed, semantic_maps)
        height, width = map_input.shape[-2:]
        flat_map = map_input.reshape(batch * agents, 14, height, width)
        features, logits_parts = [], []
        for offset in range(0, len(flat_map), self.map_agent_chunk):
            part = flat_map[offset:offset + self.map_agent_chunk]
            if self.activation_checkpointing and self.training and torch.is_grad_enabled():
                f32_part, logits_part = checkpoint(
                    self.map_unet, part, use_reentrant=False)
            else:
                f32_part, logits_part = self.map_unet(part)
            features.append(f32_part)
            logits_parts.append(logits_part)
        f32, logits = torch.cat(features), torch.cat(logits_parts)
        pooled = F.adaptive_avg_pool2d(f32, (8, 8))
        pooled = pooled.flatten(2).transpose(1, 2).reshape(batch, agents, 64, 32)
        axis = torch.linspace(-1, 1, 8, device=observed.device)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        coords = torch.stack((xx, yy), -1).reshape(1, 1, 64, 2)
        map_tokens = self.map_projection(pooled) + self.map_coordinate_mlp(coords)
        history_time = sinusoidal_embedding(
            torch.arange(8, device=observed.device, dtype=torch.float32), 128)
        history_tokens = hist_tokens + history_time.view(1, 1, 8, 128) + self.history_type
        map_tokens = map_tokens + self.map_type
        memory = self.memory_norm(torch.cat((history_tokens, map_tokens), dim=2))
        mask = valid[..., None, None].to(memory.dtype)
        return EBJDContext(
            memory=memory * mask,
            social_h=social_h * valid[..., None],
            observed_bias=observed_bias,
            baseline=baseline,
            mean_velocity=mean_velocity,
            map_features=f32.reshape(batch, agents, 32, height, width),
            map_logits=logits.reshape(batch, agents, 12, height, width),
            map_center=observed[..., -1, :],
            valid=valid,
            observed=observed,
            crop_width_m=self.crop_width_m,
        )
