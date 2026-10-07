"""Pure-CPU reference algebra for diagnostics; no project/model imports."""

from __future__ import annotations

import numpy as np


def decompose_uniform(cost: np.ndarray):
    """Return I, a, b, mu for a full rectangular candidate support."""
    value = np.asarray(cost, dtype=np.float64)
    if value.ndim != 2 or not value.size:
        raise ValueError("cost must be a non-empty matrix")
    row = value.mean(axis=1)
    col = value.mean(axis=0)
    mu = float(value.mean())
    interaction = value - row[:, None] - col[None, :] + mu
    a = row - mu
    b = col
    return interaction, a, b, mu


def reconstruct(interaction, a, b):
    return interaction + a[:, None] + b[None, :]


def logsumexp(x: np.ndarray, axis: int = -1):
    x = np.asarray(x, dtype=np.float64)
    peak = x.max(axis=axis, keepdims=True)
    return np.squeeze(peak, axis=axis) + np.log(np.exp(x - peak).sum(axis=axis))


def sample_jensen_gap(logits: np.ndarray) -> float:
    """mean_s LSE(z_s) - LSE(mean_s z_s), non-negative in exact arithmetic."""
    z = np.asarray(logits, dtype=np.float64)
    if z.ndim != 2:
        raise ValueError("logits must have [draw,candidate] shape")
    return float(logsumexp(z, axis=1).mean() - logsumexp(z.mean(axis=0)))
