"""Training/inference shared deterministic differentiable DDIM sampler."""

from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint

from .encoders import EBJDContext
from .model import EBJDModel
from .representation import cosine_vp


def differentiable_sample(
    model: EBJDModel,
    context: EBJDContext,
    initial_noise: torch.Tensor,
    steps: int = 20,
    checkpoint_steps: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return trajectory, literal goal and final latent without detaching steps."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    latent = initial_noise.float()
    batch = latent.shape[0]
    for index in range(steps, 0, -1):
        time = torch.full(
            (batch,), index / steps, device=latent.device, dtype=torch.float32)
        next_time = torch.full_like(time, (index - 1) / steps)
        if checkpoint_steps and torch.is_grad_enabled():
            velocity = checkpoint(
                lambda z, t: model.denoiser(z, t, context)[0],
                latent, time, use_reentrant=False)
        else:
            velocity = model.denoiser(latent, time, context)[0]
        with torch.autocast(device_type=latent.device.type, enabled=False):
            alpha, sigma = cosine_vp(time)
            next_alpha, next_sigma = cosine_vp(next_time)
            alpha = alpha.to(latent).float().view(batch, 1, 1, 1, 1)
            sigma = sigma.to(latent).float().view(batch, 1, 1, 1, 1)
            next_alpha = next_alpha.to(latent).float().view(batch, 1, 1, 1, 1)
            next_sigma = next_sigma.to(latent).float().view(batch, 1, 1, 1, 1)
            clean = alpha * latent.float() - sigma * velocity.float()
            noise = sigma * latent.float() + alpha * velocity.float()
            latent = next_alpha * clean + next_sigma * noise
    trajectory, goal = model.representation.decode(latent, context.baseline)
    return trajectory, goal, latent


@torch.no_grad()
def predict(
    model: EBJDModel,
    observed: torch.Tensor,
    semantic_maps: torch.Tensor,
    valid: torch.Tensor,
    worlds: int = 20,
    steps: int = 20,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    context = model.encode_context(observed, semantic_maps, valid)
    noise = torch.randn(
        observed.shape[0], worlds, observed.shape[1], 12, 2,
        device=observed.device, dtype=torch.float32, generator=generator)
    trajectory, goal, _ = differentiable_sample(model, context, noise, steps)
    return trajectory, goal
