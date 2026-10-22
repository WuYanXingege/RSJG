"""Shared tensor-safe layers used by the independent EBJD network."""

from __future__ import annotations

import math

import torch
from torch import nn


def sinusoidal_embedding(values: torch.Tensor, dim: int) -> torch.Tensor:
    """Return a stable sinusoidal embedding with an explicit final width."""
    if dim < 2:
        raise ValueError("sinusoidal embedding dim must be at least two")
    values = values.float()
    half = dim // 2
    exponent = -math.log(10000.0) * torch.arange(
        half, device=values.device, dtype=torch.float32) / max(half - 1, 1)
    phases = values.unsqueeze(-1) * exponent.exp()
    result = torch.cat((phases.sin(), phases.cos()), dim=-1)
    if result.shape[-1] < dim:
        result = torch.nn.functional.pad(result, (0, dim - result.shape[-1]))
    return result


class BiasedMultiheadAttention(nn.Module):
    """Self attention supporting a distinct additive bias for every head."""

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("attention dim must be divisible by heads")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)

    def forward(
        self,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply attention to ``[B,L,D]`` with bias ``[B,H,L,L]``."""
        if x.ndim != 3 or x.shape[-1] != self.dim:
            raise ValueError("x must have shape [batch,length,dim]")
        batch, length, _ = x.shape
        qkv = self.qkv(x).view(
            batch, length, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        logits = torch.matmul(q.float(), k.float().transpose(-1, -2))
        logits = logits / math.sqrt(self.head_dim)
        if bias is not None:
            if bias.shape != (batch, self.heads, length, length):
                raise ValueError(
                    f"bias must be {(batch, self.heads, length, length)}")
            logits = logits + bias.float()
        query_valid = None
        if valid is not None:
            if valid.shape != (batch, length):
                raise ValueError("valid must have shape [batch,length]")
            key_valid = valid[:, None, None, :]
            logits = logits.masked_fill(~key_valid, -torch.inf)
            query_valid = valid[:, :, None]
        weights = torch.softmax(logits, dim=-1).to(v.dtype)
        weights = torch.nan_to_num(weights)
        result = torch.matmul(weights, v).transpose(1, 2).reshape(
            batch, length, self.dim)
        result = self.out(result)
        if query_valid is not None:
            result = result * query_valid.to(result.dtype)
        return result


class PreNormSocialBlock(nn.Module):
    """Pre-LN social attention plus a 4x expansion feed-forward network."""

    def __init__(self, dim: int, heads: int, ff_dim: int) -> None:
        super().__init__()
        self.ln_attn = nn.LayerNorm(dim)
        self.attn = BiasedMultiheadAttention(dim, heads)
        self.ln_ff = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim), nn.GELU(), nn.Linear(ff_dim, dim))

    def forward(
        self, x: torch.Tensor, bias: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        x = x + self.attn(self.ln_attn(x), bias=bias, valid=valid)
        x = x + self.ff(self.ln_ff(x))
        return x * valid.unsqueeze(-1).to(x.dtype)
