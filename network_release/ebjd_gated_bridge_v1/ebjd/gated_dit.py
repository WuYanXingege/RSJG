"""Endpoint-conditioned, gated factorized diffusion-transformer blocks.

These blocks are an explicit research architecture.  The historical FJAB is
kept in ``denoiser.py`` and never instantiates these parameters implicitly.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class NormalizedAttention(nn.Module):
    """Multihead attention with per-head RMS-normalized Q/K and FP32 logits.

    ``valid`` masks keys; self-attention also masks the corresponding queries.
    Geometry remains an additive logit bias in its original units.  Fully
    masked rows return zero, including the output projection's bias.
    """

    def __init__(self, dim: int, heads: int, eps: float = 1e-6) -> None:
        super().__init__()
        if dim < 1 or heads < 1 or dim % heads:
            raise ValueError("attention dim must be positive and divisible by heads")
        self.dim, self.heads = int(dim), int(heads)
        self.head_dim = dim // heads
        self.eps = float(eps)
        if self.eps <= 0:
            raise ValueError("attention RMS epsilon must be positive")
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.q_norm = nn.Parameter(torch.ones(self.head_dim))
        self.k_norm = nn.Parameter(torch.ones(self.head_dim))

    def _normalize(self, value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        value = value.float()
        return value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.eps) * weight.float()

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
        valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if query.ndim != 3 or query.shape[-1] != self.dim:
            raise ValueError("attention query must be [batch,length,dim]")
        source = query if memory is None else memory
        if source.ndim != 3 or source.shape[0] != query.shape[0] or source.shape[-1] != self.dim:
            raise ValueError("attention memory must have matching batch and dim")
        batch, queries, _ = query.shape
        keys = source.shape[1]
        q = self.query(query).view(batch, queries, self.heads, self.head_dim).transpose(1, 2)
        k = self.key(source).view(batch, keys, self.heads, self.head_dim).transpose(1, 2)
        v = self.value(source).view(batch, keys, self.heads, self.head_dim).transpose(1, 2)
        with torch.autocast(device_type=query.device.type, enabled=False):
            logits = torch.matmul(
                self._normalize(q, self.q_norm),
                self._normalize(k, self.k_norm).transpose(-1, -2),
            ) / math.sqrt(self.head_dim)
            if bias is not None:
                if bias.shape != (batch, self.heads, queries, keys):
                    raise ValueError("attention geometry bias has incompatible dimensions")
                logits = logits + bias.float()
            if valid is not None:
                if valid.shape != (batch, keys) or valid.dtype != torch.bool:
                    raise ValueError("attention valid must be boolean [batch,key_length]")
                logits = logits.masked_fill(~valid[:, None, None, :], -torch.inf)
                # Softmax(-inf,...,-inf) has NaN backward even if its output
                # is later replaced by zero.  Make empty rows finite first.
                logits = torch.where(
                    valid.any(-1)[:, None, None, None], logits, torch.zeros_like(logits))
            weights = torch.nan_to_num(torch.softmax(logits, dim=-1))
            if valid is not None:
                weights = weights * valid[:, None, None, :].to(weights.dtype)
        value = torch.matmul(weights.to(v.dtype), v)
        value = value.transpose(1, 2).reshape(batch, queries, self.dim)
        result = self.out(value)
        if valid is not None:
            result = result * valid.any(-1)[:, None, None].to(result.dtype)
            if memory is None:
                result = result * valid[:, :, None].to(result.dtype)
        return result


class EndpointGatedJointBlock(nn.Module):
    """Factorized joint attention with endpoint-conditioned adaLN-Zero gates.

    Four independently modulated residual branches process temporal slots,
    within-world people, person-specific context, and the FFN.  World labels
    never enter an attention axis.  Zero gates give an initial identity block;
    branch weights begin learning after the gates open, as in adaLN-Zero.
    """

    def __init__(self, dim: int = 192, heads: int = 4, ff_dim: int = 768) -> None:
        super().__init__()
        self.dim = int(dim)
        self.heads = int(heads)
        self.norms = nn.ModuleList([
            nn.LayerNorm(dim, elementwise_affine=False) for _ in range(4)])
        self.temporal = NormalizedAttention(dim, heads)
        self.social = NormalizedAttention(dim, heads)
        self.cross = NormalizedAttention(dim, heads)
        self.ff = nn.Sequential(nn.Linear(dim, ff_dim), nn.GELU(), nn.Linear(ff_dim, dim))
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 12 * dim))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(
        self,
        hidden: torch.Tensor,
        memory: torch.Tensor,
        social_h: torch.Tensor,
        time_embedding: torch.Tensor,
        bias: torch.Tensor,
        valid: torch.Tensor,
        endpoint_condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hidden.ndim != 5 or hidden.shape[-1] != self.dim:
            raise ValueError("hidden must be [batch,worlds,agents,times,dim]")
        batch, worlds, agents, times, dim = hidden.shape
        if social_h.shape != (batch, agents, dim) or time_embedding.shape != (batch, dim):
            raise ValueError("history/time conditioning dimensions differ")
        if valid.shape != (batch, agents) or valid.dtype != torch.bool:
            raise ValueError("valid must be boolean [batch,agents]")
        if memory.ndim != 4 or memory.shape[:2] != (batch, agents) or memory.shape[-1] != dim:
            raise ValueError("memory must be [batch,agents,tokens,dim]")
        if bias.shape != (batch, worlds, times, self.heads, agents, agents):
            raise ValueError("joint geometry bias dimensions differ")
        condition = time_embedding[:, None, None, :] + social_h[:, None, :, :]
        condition = condition.expand(batch, worlds, agents, dim)
        if endpoint_condition is not None:
            if endpoint_condition.shape != (batch, worlds, agents, dim):
                raise ValueError("endpoint condition dimensions differ")
            condition = condition + endpoint_condition
        modulation = self.modulation(condition).view(batch, worlds, agents, 4, 3, dim)

        def prepare(branch: int):
            shift, scale, raw_gate = modulation[..., branch, :, :].unbind(-2)
            query = self.norms[branch](hidden)
            query = query * (1 + scale.unsqueeze(-2)) + shift.unsqueeze(-2)
            return query, raw_gate.tanh().unsqueeze(-2)

        query, gate = prepare(0)
        value = self.temporal(query.reshape(batch * worlds * agents, times, dim))
        hidden = hidden + gate * value.reshape_as(hidden)

        query, gate = prepare(1)
        social = query.permute(0, 1, 3, 2, 4).reshape(batch * worlds * times, agents, dim)
        flat_bias = bias.reshape(batch * worlds * times, self.heads, agents, agents)
        flat_valid = valid[:, None, None].expand(batch, worlds, times, agents).reshape(-1, agents)
        social = self.social(social, bias=flat_bias, valid=flat_valid)
        social = social.reshape(batch, worlds, times, agents, dim).permute(0, 1, 3, 2, 4)
        hidden = hidden + gate * social

        query, gate = prepare(2)
        flat_query = query.reshape(batch * worlds * agents, times, dim)
        flat_memory = memory[:, None].expand(
            batch, worlds, agents, memory.shape[-2], dim).reshape(
                batch * worlds * agents, memory.shape[-2], dim)
        cross = self.cross(flat_query, memory=flat_memory).reshape_as(hidden)
        hidden = hidden + gate * cross

        query, gate = prepare(3)
        hidden = hidden + gate * self.ff(query)
        return hidden * valid[:, None, :, None, None].to(hidden.dtype)
