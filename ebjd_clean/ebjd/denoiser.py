"""Six-block factorized joint denoiser with clean-future geometry."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.profiler import record_function
from torch.utils.checkpoint import checkpoint

from .encoders import EBJDContext
from .layers import BiasedMultiheadAttention, sinusoidal_embedding
from .representation import cosine_vp


class FactorizedFutureBlock(nn.Module):
    """Temporal, social, per-agent context, then FFN; never mixes worlds."""

    def __init__(self, dim: int = 128, heads: int = 4, ff_dim: int = 512) -> None:
        super().__init__()
        self.dim = dim
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(4)])
        self.temporal = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=0)
        self.social = BiasedMultiheadAttention(dim, heads)
        self.cross = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=0)
        self.ff = nn.Sequential(nn.Linear(dim, ff_dim), nn.GELU(), nn.Linear(ff_dim, dim))
        self.modulation = nn.Sequential(nn.Linear(2 * dim, dim), nn.SiLU(), nn.Linear(dim, 2 * dim))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def _modulate(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.modulation(condition).chunk(2, dim=-1)
        return (1 + gamma) * x + beta

    def forward(
        self,
        hidden: torch.Tensor,
        memory: torch.Tensor,
        social_h: torch.Tensor,
        time_embedding: torch.Tensor,
        bias: torch.Tensor,
        valid: torch.Tensor,
        social_enabled: bool = True,
    ) -> torch.Tensor:
        batch, worlds, agents, times, dim = hidden.shape
        condition = torch.cat((
            time_embedding[:, None, None, None, :].expand(batch, 1, agents, 1, dim),
            social_h[:, None, :, None, :],
        ), dim=-1)
        query = self._modulate(self.norms[0](hidden), condition)
        temporal = query.reshape(batch * worlds * agents, times, dim)
        temporal = self.temporal(temporal, temporal, temporal, need_weights=False)[0]
        hidden = hidden + temporal.reshape_as(hidden)

        if social_enabled:
            query = self._modulate(self.norms[1](hidden), condition)
            social = query.permute(0, 1, 3, 2, 4).reshape(batch * worlds * times, agents, dim)
            flat_bias = bias.reshape(batch * worlds * times, 4, agents, agents)
            flat_valid = valid[:, None, None, :].expand(batch, worlds, times, agents)
            social = self.social(
                social, flat_bias, flat_valid.reshape(batch * worlds * times, agents))
            hidden = hidden + social.reshape(batch, worlds, times, agents, dim).permute(0, 1, 3, 2, 4)

        query = self._modulate(self.norms[2](hidden), condition)
        flat_query = query.reshape(batch * worlds * agents, times, dim)
        flat_memory = memory[:, None].expand(batch, worlds, agents, 72, dim).reshape(
            batch * worlds * agents, 72, dim)
        cross = self.cross(flat_query, flat_memory, flat_memory, need_weights=False)[0]
        hidden = hidden + cross.reshape_as(hidden)
        query = self._modulate(self.norms[3](hidden), condition)
        hidden = hidden + self.ff(query)
        return hidden * valid[:, None, :, None, None].to(hidden.dtype)


def _norm(x: torch.Tensor) -> torch.Tensor:
    return x.square().sum(-1).add(1e-6).sqrt()


def future_pair_geometry(
    trajectories: torch.Tensor,
    goals: torch.Tensor,
    goal_scale: torch.Tensor,
    last_observed: torch.Tensor | None = None,
    dt: float = 0.4,
) -> torch.Tensor:
    """Compute ordered pair geometry [B,P,T,N,N,12] from predicted clean paths."""
    with torch.autocast(device_type=trajectories.device.type, enabled=False):
        trajectories, goals = trajectories.float(), goals.float()
        velocity = torch.empty_like(trajectories)
        if last_observed is None:
            velocity[..., 0, :] = (trajectories[..., 1, :] - trajectories[..., 0, :]) / dt
        else:
            velocity[..., 0, :] = (
                trajectories[..., 0, :] - last_observed.float()[:, None, :, :]) / dt
        velocity[..., 1:, :] = (trajectories[..., 1:, :] - trajectories[..., :-1, :]) / dt
        y = trajectories.permute(0, 1, 3, 2, 4)
        v = velocity.permute(0, 1, 3, 2, 4)
        rel_p = y[..., :, None, :] - y[..., None, :, :]
        rel_v = v[..., :, None, :] - v[..., None, :, :]
        distance = _norm(rel_p)
        closing = -(rel_p * rel_v).sum(-1) / distance
        goal_rel = goals[..., :, None, :] - goals[..., None, :, :]
        goal_distance = _norm(goal_rel)
        rel_speed2 = rel_v.square().sum(-1)
        cpa_t = (-(rel_p * rel_v).sum(-1) / rel_speed2.clamp_min(1e-6)).clamp(0, 4.8)
        cpa_d = _norm(rel_p + cpa_t[..., None] * rel_v)
        time_ratio = torch.linspace(
            1 / 12, 1, 12, device=y.device, dtype=torch.float32).view(1, 1, 12, 1, 1, 1)
        scale = goal_scale.float().clamp_min(0.5)
        return torch.cat((
            rel_p / scale,
            rel_v,
            (distance / scale)[..., None],
            closing[..., None],
            (goal_rel[:, :, None].expand_as(rel_p) / scale),
            (goal_distance[:, :, None].expand_as(distance) / scale)[..., None],
            time_ratio.expand(*distance.shape, 1),
            (cpa_t / 4.8)[..., None],
            (cpa_d / scale)[..., None],
        ), dim=-1).clamp(-10, 10)


class JointEndpointBridgeDenoiser(nn.Module):
    def __init__(
        self,
        representation: nn.Module,
        dim: int = 128,
        future_social: bool = True,
        clean_geometry: bool = True,
        activation_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        self.representation = representation
        self.future_social = bool(future_social)
        self.clean_geometry = bool(clean_geometry)
        self.activation_checkpointing = bool(activation_checkpointing)
        self.state_stem = nn.Sequential(nn.Linear(2, dim), nn.LayerNorm(dim))
        self.time_mlp = nn.Sequential(nn.Linear(dim, 256), nn.SiLU(), nn.Linear(256, dim))
        self.token_type = nn.Parameter(torch.zeros(2, dim))
        self.register_buffer(
            "future_time", sinusoidal_embedding(torch.arange(12).float(), dim))
        self.blocks = nn.ModuleList([FactorizedFutureBlock(dim, 4, 512) for _ in range(6)])
        self.pre_bridge_head = self._head(dim)
        self.pre_goal_head = self._head(dim)
        self.final_bridge_head = self._head(dim)
        self.final_goal_head = self._head(dim)
        self.clean_injection = nn.Sequential(nn.Linear(6, dim), nn.LayerNorm(dim))
        self.map_injection = nn.Sequential(nn.Linear(33, dim), nn.LayerNorm(dim))
        self.future_geometry_mlp = nn.Sequential(
            nn.Linear(12, 64), nn.SiLU(), nn.Linear(64, 4))
        for head in (
            self.pre_bridge_head, self.pre_goal_head,
            self.final_bridge_head, self.final_goal_head,
        ):
            nn.init.normal_(head[-1].weight, std=1e-3)
            nn.init.zeros_(head[-1].bias)

    @staticmethod
    def _head(dim: int) -> nn.Sequential:
        return nn.Sequential(nn.Linear(dim, 64), nn.SiLU(), nn.Linear(64, 2))

    @staticmethod
    def _combine(bridge: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        return torch.cat((bridge, goal.unsqueeze(-2)), dim=-2)

    def _sample_map(self, context: EBJDContext, trajectory: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=trajectory.device.type, enabled=False):
            batch, worlds, agents, times = trajectory.shape[:4]
            relative = trajectory.float() - context.map_center.float()[:, None, :, None, :]
            grid = relative / (context.crop_width_m / 2)
            grid_bn = grid.permute(0, 2, 1, 3, 4).reshape(batch * agents, worlds * times, 1, 2)
            sampled = F.grid_sample(
                context.map_features.float().reshape(
                    batch * agents, 32, *context.map_features.shape[-2:]),
                grid_bn, mode="bilinear", padding_mode="zeros", align_corners=True)
            sampled = sampled.squeeze(-1).reshape(batch, agents, 32, worlds, times)
            sampled = sampled.permute(0, 3, 1, 4, 2)
            in_crop = (grid.abs() <= 1).all(-1, keepdim=True).float()
            return torch.cat((sampled * in_crop, in_crop), dim=-1)

    def _clean_features(
        self, trajectory: torch.Tensor, goal: torch.Tensor, context: EBJDContext
    ) -> torch.Tensor:
        with torch.autocast(device_type=trajectory.device.type, enabled=False):
            trajectory, goal = trajectory.float(), goal.float()
            baseline = context.baseline.float()[:, None]
            velocity = torch.empty_like(trajectory)
            velocity[..., 0, :] = (
                trajectory[..., 0, :] - context.observed.float()[:, None, :, -1, :]) / 0.4
            velocity[..., 1:, :] = (trajectory[..., 1:, :] - trajectory[..., :-1, :]) / 0.4
            position = (trajectory - baseline) / self.representation.goal_scale.float().clamp_min(0.5)
            velocity_delta = velocity - context.mean_velocity.float()[:, None, :, None, :]
            goal_delta = (goal - baseline[..., -1, :]) / self.representation.goal_scale.float().clamp_min(0.5)
            return torch.cat((
                position,
                velocity_delta,
                goal_delta.unsqueeze(-2).expand(*position.shape[:-1], 2),
            ), dim=-1)

    def forward(
        self,
        latent: torch.Tensor,
        time: torch.Tensor | float,
        context: EBJDContext,
        checkpoint_blocks: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if latent.ndim != 5 or latent.shape[-2:] != (12, 2):
            raise ValueError("latent must have shape [B,P,N,12,2]")
        batch, worlds, agents = latent.shape[:3]
        time = torch.as_tensor(time, device=latent.device, dtype=torch.float32)
        if time.ndim == 0:
            time = time.expand(batch)
        if time.shape != (batch,):
            raise ValueError("time must be scalar or [B]")
        use_block_checkpoints = (
            self.activation_checkpointing
            if checkpoint_blocks is None else bool(checkpoint_blocks))
        alpha, sigma = cosine_vp(time)
        alpha = alpha.to(latent.device).view(batch, 1, 1, 1, 1)
        sigma = sigma.to(latent.device).view(batch, 1, 1, 1, 1)
        time_embedding = self.time_mlp(sinusoidal_embedding(time, 128))
        types = torch.zeros(12, dtype=torch.long, device=latent.device)
        types[-1] = 1
        hidden = (
            self.state_stem(latent)
            + self.future_time.view(1, 1, 1, 12, 128)
            + self.token_type[types].view(1, 1, 1, 12, 128)
            + context.social_h[:, None, :, None, :]
        )
        observed = context.observed_bias[:, None, None].expand(
            batch, worlds, 12, 4, agents, agents)
        for block in self.blocks[:3]:
            if use_block_checkpoints and self.training and torch.is_grad_enabled():
                with record_function("ebjd.denoiser_block_checkpoint"):
                    hidden = checkpoint(
                        lambda value, module=block: module(
                            value, context.memory, context.social_h, time_embedding,
                            observed, context.valid, social_enabled=self.future_social),
                        hidden, use_reentrant=False)
            else:
                hidden = block(
                    hidden, context.memory, context.social_h, time_embedding,
                    observed, context.valid, social_enabled=self.future_social)
        pre = self._combine(
            self.pre_bridge_head(hidden[..., :11, :]),
            self.pre_goal_head(hidden[..., 11, :]),
        )
        with torch.autocast(device_type=latent.device.type, enabled=False):
            clean_latent = alpha.float() * latent.float() - sigma.float() * pre.float()
        trajectory, goal = self.representation.decode(clean_latent, context.baseline)
        gate = alpha.square()
        hidden = hidden + gate * (
            self.clean_injection(self._clean_features(trajectory, goal, context))
            + self.map_injection(self._sample_map(context, trajectory)))
        geometry_source = trajectory if self.clean_geometry else self.representation.decode(latent, context.baseline)[0]
        geometry_goal = goal if self.clean_geometry else geometry_source[..., -1, :]
        future = 2 * torch.tanh(self.future_geometry_mlp(
            future_pair_geometry(
                geometry_source, geometry_goal, self.representation.goal_scale,
                context.observed[..., -1, :])))
        future = future.permute(0, 1, 2, 5, 3, 4).contiguous()
        combined = observed + gate.view(batch, 1, 1, 1, 1, 1) * future
        for block in self.blocks[3:]:
            if use_block_checkpoints and self.training and torch.is_grad_enabled():
                with record_function("ebjd.denoiser_block_checkpoint"):
                    hidden = checkpoint(
                        lambda value, module=block: module(
                            value, context.memory, context.social_h, time_embedding,
                            combined, context.valid, social_enabled=self.future_social),
                        hidden, use_reentrant=False)
            else:
                hidden = block(
                    hidden, context.memory, context.social_h, time_embedding,
                    combined, context.valid, social_enabled=self.future_social)
        final = self._combine(
            self.final_bridge_head(hidden[..., :11, :]),
            self.final_goal_head(hidden[..., 11, :]),
        )
        mask = context.valid[:, None, :, None, None].to(final.dtype)
        return final * mask, pre * mask, trajectory, goal
