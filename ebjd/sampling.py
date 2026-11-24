"""Training/inference shared deterministic differentiable DDIM sampler."""

from __future__ import annotations

from collections import OrderedDict
import torch
from torch.profiler import record_function
from torch.utils.checkpoint import checkpoint

from .encoders import EBJDContext
from .model import EBJDModel
from .representation import cosine_vp
from .execution import enabled


_SCHEDULES = OrderedDict()


def _ddim_schedule(steps, batch, device, dtype):
    """Bounded, parameter-free schedule cache; callers must not mutate tensors."""
    key = (steps, batch, str(device), dtype)
    if not enabled('schedule') or key not in _SCHEDULES:
        schedule = []
        for index in range(steps, 0, -1):
            time = torch.full((batch,), index / steps, device=device, dtype=torch.float32)
            next_time = torch.full_like(time, (index - 1) / steps)
            # Preserve original tensor shapes and per-step trigonometric arithmetic.
            coefficients = (*cosine_vp(time), *cosine_vp(next_time))
            schedule.append((time, *(x.to(dtype).float().view(batch, 1, 1, 1, 1) for x in coefficients)))
        if not enabled('schedule'):
            return tuple(schedule)
        _SCHEDULES[key] = tuple(schedule)
        if len(_SCHEDULES) > 16:
            _SCHEDULES.popitem(last=False)
    _SCHEDULES.move_to_end(key)
    return _SCHEDULES[key]


def differentiable_sample(
    model: EBJDModel,
    context: EBJDContext,
    initial_noise: torch.Tensor,
    steps: int = 20,
    checkpoint_steps: bool = False,
    checkpoint_blocks: bool | None = None,
    checkpoint_preserve_rng_state: bool = True,
    proposal_goals: torch.Tensor | None = None,
    proposal_eta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return trajectory, literal goal and final latent without detaching steps."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    latent = initial_noise.float()
    batch = latent.shape[0]
    proposal_condition = None
    if model.denoiser.proposal_conditioned and enabled('proposal'):
        if proposal_goals is None:
            raise ValueError("proposal-conditioned sampling requires proposal_goals")
        proposal_condition = model.denoiser.encode_proposal_condition(context, proposal_goals, latent.dtype)
    for time, alpha, sigma, next_alpha, next_sigma in _ddim_schedule(steps, batch, latent.device, latent.dtype):
        if checkpoint_steps and torch.is_grad_enabled():
            # Capture an immutable per-call policy.  In rollout training the
            # outer DDIM-step checkpoint is sufficient; nesting six block
            # checkpoints inside each recomputation only adds overhead.
            block_policy = checkpoint_blocks
            with record_function("ebjd.ddim_step_checkpoint"):
                velocity = checkpoint(
                    lambda z, t, policy=block_policy: model.denoiser(
                        z, t, context, checkpoint_blocks=policy,
                        proposal_goals=proposal_goals,
                        proposal_eta=proposal_eta,
                        proposal_condition=proposal_condition)[0],
                    latent, time, use_reentrant=False,
                    preserve_rng_state=checkpoint_preserve_rng_state)
        else:
            velocity = model.denoiser(
                latent, time, context, checkpoint_blocks=checkpoint_blocks,
                proposal_goals=proposal_goals,
                proposal_eta=proposal_eta,
                proposal_condition=proposal_condition)[0]
        with torch.autocast(device_type=latent.device.type, enabled=False):
            clean = alpha * latent.float() - sigma * velocity.float()
            noise = sigma * latent.float() + alpha * velocity.float()
            latent = next_alpha * clean + next_sigma * noise
    if model.denoiser.proposal_conditioned:
        if proposal_goals is None:
            raise ValueError("proposal-conditioned sampling requires proposal_goals")
        trajectory, goal = model.representation.decode_conditioned(
            latent, context.baseline, proposal_goals, proposal_eta)
    else:
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
    proposal_goals: torch.Tensor | None = None,
    proposal_eta: float = 1.0,
    initial_noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    context = model.encode_context(observed, semantic_maps, valid)
    if initial_noise is None:
        noise = torch.randn(
            observed.shape[0], worlds, observed.shape[1], 12, 2,
            device=observed.device, dtype=torch.float32, generator=generator)
    else:
        expected = (observed.shape[0], worlds, observed.shape[1], 12, 2)
        if initial_noise.shape != expected:
            raise ValueError(f"initial_noise must have shape {expected}")
        noise = initial_noise.to(device=observed.device, dtype=torch.float32)
    trajectory, goal, _ = differentiable_sample(
        model, context, noise, steps,
        proposal_goals=proposal_goals, proposal_eta=proposal_eta)
    return trajectory, goal
