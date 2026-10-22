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


class NativeGoalConditioner(nn.Module):
    """Cross-attend future tokens to frozen GDTS endpoint proposals."""

    def __init__(
        self, dim: int = 128, heads: int = 4,
        anchor_conditioning: bool = False,
    ) -> None:
        super().__init__()
        self.candidate_mlp = nn.Sequential(
            nn.Linear(4, 64), nn.SiLU(), nn.Linear(64, dim))
        self.query_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(
            dim, heads, dropout=0.0, batch_first=True)
        self.gate = nn.Parameter(torch.zeros(()))
        self.goal_anchor_gate = (
            nn.Parameter(torch.zeros(())) if anchor_conditioning else None)
        self.bridge_anchor_gate = (
            nn.Parameter(torch.zeros(())) if anchor_conditioning else None)
        self.enabled = True
        self.last_condition_norm_ratio: torch.Tensor | None = None

    def candidate_features(
        self, context: EBJDContext, goals: torch.Tensor,
        goal_scale: torch.Tensor,
    ) -> torch.Tensor:
        origin = context.observed.float()[..., -1, :]
        centre = context.baseline.float()[..., -1, :]
        if goals.ndim != 4:
            raise ValueError("candidate or anchor goals must have four dimensions")
        if goals.shape[1] == origin.shape[1]:
            # Candidate set [B,N,K,2].
            origin, centre = origin[:, :, None], centre[:, :, None]
        elif goals.shape[2] == origin.shape[1]:
            # Assigned anchors [B,P,N,2].
            origin, centre = origin[:, None], centre[:, None]
        else:
            raise ValueError("goal axes do not match the context agent dimension")
        scale = goal_scale.float().clamp_min(0.5)
        return torch.cat((
            (goals.float() - origin) / scale,
            (goals.float() - centre) / scale,
        ), dim=-1).clamp(-20, 20)

    def encode_candidates(
        self, context: EBJDContext, goal_scale: torch.Tensor,
    ) -> torch.Tensor:
        if context.native_goals is None:
            raise ValueError("native goal conditioning requires native_goals")
        with torch.autocast(device_type=context.native_goals.device.type, enabled=False):
            features = self.candidate_features(
                context, context.native_goals, goal_scale)
        return self.candidate_mlp(features.to(self.candidate_mlp[0].weight.dtype))

    def encode_anchors(
        self, context: EBJDContext, goal_scale: torch.Tensor,
    ) -> torch.Tensor:
        if context.world_anchors is None:
            raise ValueError("world-anchor conditioning requires assigned anchors")
        with torch.autocast(device_type=context.world_anchors.device.type, enabled=False):
            features = self.candidate_features(
                context, context.world_anchors, goal_scale)
        return self.candidate_mlp(features.to(self.candidate_mlp[0].weight.dtype))

    def forward(
        self,
        hidden: torch.Tensor,
        context: EBJDContext,
        goal_scale: torch.Tensor,
    ) -> torch.Tensor:
        if not self.enabled:
            self.last_condition_norm_ratio = hidden.new_zeros(())
            return hidden
        goals = context.native_goals
        if goals is None:
            raise ValueError("native goal conditioning requires native_goals")
        if goals.ndim != 4 or goals.shape[:2] != hidden.shape[:1] + hidden.shape[2:3]:
            raise ValueError("native_goals must have shape [B,N,K,2]")
        if goals.shape[-2:] != (20, 2):
            raise ValueError("native_goals must contain exactly 20 endpoint proposals")
        candidates = context.native_candidate_embeddings
        if candidates is None:
            with torch.autocast(device_type=hidden.device.type, enabled=False):
                features = self.candidate_features(context, goals, goal_scale)
            candidates = self.candidate_mlp(features.to(hidden.dtype))
        else:
            candidates = candidates.to(hidden.dtype)
        batch, worlds, agents, times, dim = hidden.shape
        query = self.query_norm(hidden).reshape(batch * worlds * agents, times, dim)
        keys = candidates[:, None].expand(batch, worlds, agents, 20, dim).reshape(
            batch * worlds * agents, 20, dim)
        delta = self.attention(query, keys, keys, need_weights=False)[0].reshape_as(hidden)
        valid = context.valid[:, None, :, None, None].to(hidden.dtype)
        delta = delta * valid
        gated = torch.tanh(self.gate).to(hidden.dtype) * delta
        with torch.no_grad():
            denominator = hidden.detach().float().norm().clamp_min(1e-12)
            # Keep diagnostics on device; the trainer transfers all update
            # scalars together after the graph is complete.
            self.last_condition_norm_ratio = (
                gated.detach().float().norm().div(denominator))
        return (hidden + gated) * valid


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

    @staticmethod
    def _self_only_bias(bias: torch.Tensor) -> torch.Tensor:
        """Retain only each valid agent's self-attention connection."""
        agents = bias.shape[-1]
        if bias.shape[-2] != agents:
            raise ValueError("social attention bias must be square")
        diagonal = torch.eye(
            agents, dtype=torch.bool, device=bias.device).view(
                *((1,) * (bias.ndim - 2)), agents, agents)
        return bias.masked_fill(~diagonal, -torch.inf)

    def _social_output(
        self,
        query: torch.Tensor,
        bias: torch.Tensor,
        valid: torch.Tensor,
        gamma: float,
    ) -> torch.Tensor:
        """Interpolate projected self-only/full attention from one input."""
        gamma = float(gamma)
        if gamma == 1.0:
            return self.social(query, bias, valid)
        self_only = self.social(query, self._self_only_bias(bias), valid)
        if gamma == 0.0:
            return self_only
        if not 0.0 < gamma < 1.0:
            raise ValueError("future social gamma must be in [0, 1]")
        full = self.social(query, bias, valid)
        return self_only + gamma * (full - self_only)

    def forward(
        self,
        hidden: torch.Tensor,
        memory: torch.Tensor,
        social_h: torch.Tensor,
        time_embedding: torch.Tensor,
        bias: torch.Tensor,
        valid: torch.Tensor,
        social_enabled: bool = True,
        social_self_only: bool = False,
        social_gamma: float | None = None,
        memory_mode: str = "legacy_joint",
        motion_memory: torch.Tensor | None = None,
        map_memory: torch.Tensor | None = None,
        memory_gate: torch.Tensor | None = None,
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
            gamma = (0.0 if social_self_only else 1.0) \
                if social_gamma is None else float(social_gamma)
            social = self._social_output(
                social, flat_bias,
                flat_valid.reshape(batch * worlds * times, agents), gamma)
            hidden = hidden + social.reshape(batch, worlds, times, agents, dim).permute(0, 1, 3, 2, 4)

        query = self._modulate(self.norms[2](hidden), condition)
        flat_query = query.reshape(batch * worlds * agents, times, dim)
        def flatten(source: torch.Tensor) -> torch.Tensor:
            count = source.shape[-2]
            return source[:, None].expand(
                batch, worlds, agents, count, dim).reshape(
                    batch * worlds * agents, count, dim)

        if memory_mode == "split_gated":
            if motion_memory is None or map_memory is None or memory_gate is None:
                raise ValueError("split-gated attention requires both memories and gate")
            motion = flatten(motion_memory)
            map_value = flatten(map_memory)
            cross_h = self.cross(
                flat_query, motion, motion, need_weights=False)[0]
            cross_m = self.cross(
                flat_query, map_value, map_value, need_weights=False)[0]
            flat_gate = memory_gate.reshape(batch * worlds * agents, 2)
            cross = (
                flat_gate[:, 0, None, None] * cross_h
                + flat_gate[:, 1, None, None] * cross_m)
        else:
            flat_memory = flatten(memory)
            attention_mask = None
            if memory_mode == "count_normalized_joint":
                if motion_memory is None or map_memory is None:
                    raise ValueError("count-normalized attention requires memory boundaries")
                motion_count, map_count = (
                    motion_memory.shape[-2], map_memory.shape[-2])
                attention_mask = flat_query.new_zeros(times, motion_count + map_count)
                attention_mask[:, :motion_count] = torch.log(torch.as_tensor(
                    map_count / motion_count, device=flat_query.device,
                    dtype=flat_query.dtype))
            cross = self.cross(
                flat_query, flat_memory, flat_memory,
                attn_mask=attention_mask, need_weights=False)[0]
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
        native_goal_conditioning: bool = False,
        memory_mode: str = "legacy_joint",
        anchor_conditioning: bool = False,
        geometry_mapping: str = "direct_index",
    ) -> None:
        super().__init__()
        self.representation = representation
        self.future_social = bool(future_social)
        # Inference-only mechanism intervention. This is deliberately runtime
        # state rather than a parameter/buffer so checkpoint identity is unchanged.
        self.future_social_self_only = False
        self.future_social_gamma = 1.0
        self.clean_geometry = bool(clean_geometry)
        self.activation_checkpointing = bool(activation_checkpointing)
        self.memory_mode = str(memory_mode)
        self.anchor_conditioning = bool(anchor_conditioning)
        if geometry_mapping not in {"direct_index", "decoder_jacobian"}:
            raise ValueError(f"unsupported geometry mapping: {geometry_mapping}")
        self.geometry_mapping = geometry_mapping
        self.static_context_cache_enabled = False
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
        if self.memory_mode == "split_gated":
            self.memory_group_gate = nn.Sequential(
                nn.Linear(3 * dim, 64), nn.SiLU(), nn.Linear(64, 2))
            nn.init.zeros_(self.memory_group_gate[-1].weight)
            nn.init.zeros_(self.memory_group_gate[-1].bias)
        else:
            self.memory_group_gate = None
        self.register_buffer(
            "future_geometry_weights",
            representation.decoder_jacobian_weights()
            if hasattr(representation, "decoder_jacobian_weights")
            else torch.eye(12), persistent=False)
        # Registered last so legacy parameter/state ordering remains an exact
        # prefix during the explicit model-extension migration.
        self.native_goal_conditioner = (
            NativeGoalConditioner(
                dim=dim, heads=4,
                anchor_conditioning=self.anchor_conditioning)
            if native_goal_conditioning else None)
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

    def prepare_static_context(self, context: EBJDContext) -> EBJDContext:
        """Cache time/latent-independent trainable encodings for one graph."""
        from dataclasses import replace
        if self.native_goal_conditioner is None or not self.static_context_cache_enabled:
            return context
        candidates = context.native_candidate_embeddings
        if candidates is None:
            candidates = self.native_goal_conditioner.encode_candidates(
                context, self.representation.goal_scale)
        anchors = context.world_anchor_embeddings
        if self.anchor_conditioning and anchors is None:
            anchors = self.native_goal_conditioner.encode_anchors(
                context, self.representation.goal_scale)
        return replace(
            context, native_candidate_embeddings=candidates,
            world_anchor_embeddings=anchors)

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
        anchor_embedding = None
        if self.native_goal_conditioner is not None:
            hidden = self.native_goal_conditioner(
                hidden, context, self.representation.goal_scale)
            if self.anchor_conditioning:
                if context.world_anchors is None or context.world_anchors.shape != (
                        batch, worlds, agents, 2):
                    raise ValueError(
                        "assigned world anchors must match latent [B,P,N,2]")
                anchor_embedding = context.world_anchor_embeddings
                if anchor_embedding is None:
                    anchor_embedding = self.native_goal_conditioner.encode_anchors(
                        context, self.representation.goal_scale)
                anchor_embedding = anchor_embedding.to(hidden.dtype)
                if self.native_goal_conditioner.goal_anchor_gate is None or \
                        self.native_goal_conditioner.bridge_anchor_gate is None:
                    raise RuntimeError("anchor gates were not constructed")
                valid_anchor = context.valid[:, None, :, None].to(hidden.dtype)
                anchor_embedding = anchor_embedding * valid_anchor
                hidden[..., :11, :] = hidden[..., :11, :] + (
                    torch.tanh(self.native_goal_conditioner.bridge_anchor_gate)
                    .to(hidden.dtype) * anchor_embedding[..., None, :])
                hidden[..., 11, :] = hidden[..., 11, :] + (
                    torch.tanh(self.native_goal_conditioner.goal_anchor_gate)
                    .to(hidden.dtype) * anchor_embedding)
        memory_gate = None
        if self.memory_group_gate is not None:
            if anchor_embedding is None:
                anchor_embedding = hidden.new_zeros(batch, worlds, agents, 128)
            gate_input = torch.cat((
                context.social_h[:, None].expand(batch, worlds, agents, 128),
                time_embedding[:, None, None].expand(batch, worlds, agents, 128),
                anchor_embedding,
            ), dim=-1)
            memory_gate = torch.softmax(self.memory_group_gate(gate_input), dim=-1)
        checkpoint_motion = (
            context.motion_memory if context.motion_memory is not None
            else context.memory[..., :0, :])
        checkpoint_map = (
            context.map_memory if context.map_memory is not None
            else context.memory[..., :0, :])
        checkpoint_gate = (
            memory_gate if memory_gate is not None
            else hidden.new_zeros(batch, worlds, agents, 2))

        def checkpointed_block(
            module: FactorizedFutureBlock, value: torch.Tensor,
            block_bias: torch.Tensor,
        ) -> torch.Tensor:
            # Cached/static context tensors are explicit checkpoint inputs.
            # This prevents recomputation from observing a mutated closure and
            # keeps gradient edges to trainable memory/gate encoders visible.
            return checkpoint(
                lambda h, memory, social, time_value, bias_value, valid_value,
                       motion, map_value, group_gate: module(
                           h, memory, social, time_value, bias_value, valid_value,
                           social_enabled=self.future_social,
                           social_gamma=self.future_social_gamma,
                           memory_mode=self.memory_mode,
                           motion_memory=motion, map_memory=map_value,
                           memory_gate=group_gate),
                value, context.memory, context.social_h, time_embedding,
                block_bias, context.valid, checkpoint_motion, checkpoint_map,
                checkpoint_gate, use_reentrant=False)
        observed = context.observed_bias[:, None, None].expand(
            batch, worlds, 12, 4, agents, agents)
        for block in self.blocks[:3]:
            if use_block_checkpoints and self.training and torch.is_grad_enabled():
                with record_function("ebjd.denoiser_block_checkpoint"):
                    hidden = checkpointed_block(block, hidden, observed)
            else:
                hidden = block(
                    hidden, context.memory, context.social_h, time_embedding,
                    observed, context.valid, social_enabled=self.future_social,
                    social_gamma=self.future_social_gamma,
                    memory_mode=self.memory_mode,
                    motion_memory=context.motion_memory,
                    map_memory=context.map_memory,
                    memory_gate=memory_gate)
        pre = self._combine(
            self.pre_bridge_head(hidden[..., :11, :]),
            self.pre_goal_head(hidden[..., 11, :]),
        )
        with torch.autocast(device_type=latent.device.type, enabled=False):
            clean_latent = alpha.float() * latent.float() - sigma.float() * pre.float()
        trajectory, goal = self.representation.decode(
            clean_latent, context.baseline, context.world_anchors)
        gate = alpha.square()
        hidden = hidden + gate * (
            self.clean_injection(self._clean_features(trajectory, goal, context))
            + self.map_injection(self._sample_map(context, trajectory)))
        geometry_source = trajectory if self.clean_geometry else self.representation.decode(
            latent, context.baseline, context.world_anchors)[0]
        geometry_goal = goal if self.clean_geometry else geometry_source[..., -1, :]
        future = 2 * torch.tanh(self.future_geometry_mlp(
            future_pair_geometry(
                geometry_source, geometry_goal, self.representation.goal_scale,
                context.observed[..., -1, :])))
        future = future.permute(0, 1, 2, 5, 3, 4).contiguous()
        if self.geometry_mapping == "decoder_jacobian":
            future = torch.einsum(
                "lt,bpthij->bplhij",
                self.future_geometry_weights.to(future), future)
        combined = observed + gate.view(batch, 1, 1, 1, 1, 1) * future
        for block in self.blocks[3:]:
            if use_block_checkpoints and self.training and torch.is_grad_enabled():
                with record_function("ebjd.denoiser_block_checkpoint"):
                    hidden = checkpointed_block(block, hidden, combined)
            else:
                hidden = block(
                    hidden, context.memory, context.social_h, time_embedding,
                    combined, context.valid, social_enabled=self.future_social,
                    social_gamma=self.future_social_gamma,
                    memory_mode=self.memory_mode,
                    motion_memory=context.motion_memory,
                    map_memory=context.map_memory,
                    memory_gate=memory_gate)
        final = self._combine(
            self.final_bridge_head(hidden[..., :11, :]),
            self.final_goal_head(hidden[..., 11, :]),
        )
        mask = context.valid[:, None, :, None, None].to(final.dtype)
        return final * mask, pre * mask, trajectory, goal
