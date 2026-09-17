"""Sparse branch-specific joint denoising dependency residual."""

import math
from typing import Optional

import torch
from torch import nn


def sinusoidal_timestep_embedding(
    timestep: torch.Tensor,
    dimension: int = 32,
) -> torch.Tensor:
    """Return the standard fixed sinusoidal embedding ``[...,32]``."""
    if dimension % 2:
        raise ValueError("dimension must be even")
    timestep = timestep.float()
    half = dimension // 2
    frequency = torch.exp(
        torch.arange(half, device=timestep.device, dtype=torch.float32) *
        (-math.log(10000.0) / max(half - 1, 1)))
    phase = timestep[..., None] * frequency
    return torch.cat((phase.sin(), phase.cos()), dim=-1)


class DependencyCorrector(nn.Module):
    """Predict ``Delta epsilon`` from noisy pair geometry and relation state."""

    def __init__(self, dt: float = 0.4, hidden_dim: int = 64,
                 relation_dim: int = 16, eps: float = 1e-8) -> None:
        super().__init__()
        if dt <= 0 or hidden_dim != 64 or relation_dim != 16:
            raise ValueError("JDV2 requires dt>0, hidden=64, relation=16")
        self.dt = float(dt)
        self.eps = float(eps)
        self.state_encoder = nn.Sequential(
            nn.Linear(21, 64), nn.SiLU(), nn.Linear(64, 64),
            nn.LayerNorm(64))
        self.timestep_network = nn.Sequential(
            nn.Linear(32, 64), nn.SiLU(), nn.Linear(64, 128))
        nn.init.zeros_(self.timestep_network[-1].weight)
        nn.init.zeros_(self.timestep_network[-1].bias)
        self.gate = nn.Sequential(
            nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 1), nn.Sigmoid())
        self.value = nn.Sequential(
            nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 64))
        self.output = nn.Sequential(
            nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 2))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(
        self,
        noisy_velocity_world: torch.Tensor,
        last_position_world: torch.Tensor,
        edge_index: torch.Tensor,
        relation_embedding: torch.Tensor,
        diffusion_timestep: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return exact-zero-safe residual ``[N,T,2]``.

        The input velocity is the current noisy diffusion state expressed in
        metres/second.  Position is reconstructed as
        ``x_last + dt*cumsum(v)``; no ``x0`` estimate is used.
        """
        if noisy_velocity_world.ndim != 3 or \
                noisy_velocity_world.shape[-1] != 2:
            raise ValueError("noisy_velocity_world must have shape [N,T,2]")
        num_agents, num_steps, _ = noisy_velocity_world.shape
        if last_position_world.shape != (num_agents, 2):
            raise ValueError("last_position_world must be [N,2]")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must be [2,E]")
        edge_count = edge_index.shape[1]
        if relation_embedding.shape != (edge_count, 16):
            raise ValueError("relation_embedding must be [E,16]")
        if edge_count == 0:
            # Bypass every learned bias so the zero-edge contract is exact.
            return torch.zeros_like(noisy_velocity_world)
        velocity = noisy_velocity_world.float()
        position = last_position_world.float()[:, None, :] + self.dt * \
            torch.cumsum(velocity, dim=1)
        src, dst = edge_index.long()
        relative_position = position[dst] - position[src]
        relative_velocity = velocity[dst] - velocity[src]
        distance = torch.linalg.vector_norm(
            relative_position, dim=-1, keepdim=True)
        relation = relation_embedding.float()[:, None, :].expand(
            -1, num_steps, -1)
        forward_state = torch.cat((
            relative_position, relative_velocity, distance, relation), dim=-1)
        reverse_state = torch.cat((
            -relative_position, -relative_velocity, distance, relation), dim=-1)
        state = torch.cat((forward_state, reverse_state), dim=0)
        hidden = self.state_encoder(state)

        timestep = torch.as_tensor(
            diffusion_timestep, device=velocity.device, dtype=torch.float32)
        if timestep.ndim == 0:
            timestep = timestep.expand(num_agents)
        if timestep.shape != (num_agents,):
            raise ValueError("diffusion_timestep must be scalar or [N]")
        receiver = torch.cat((src, dst), dim=0)
        film = self.timestep_network(
            sinusoidal_timestep_embedding(timestep))[receiver]
        gamma, beta = film.chunk(2, dim=-1)
        hidden = (1.0 + gamma[:, None, :]) * hidden + beta[:, None, :]
        gate = self.gate(hidden).float()
        value = self.value(hidden).float()
        if edge_weight is not None:
            if edge_weight.shape != (edge_count,):
                raise ValueError("edge_weight must be [E]")
            directed_weight = torch.cat((edge_weight, edge_weight)).float()
            gate = gate * directed_weight[:, None, None]
        numerator = velocity.new_zeros((num_agents, num_steps, 64))
        denominator = velocity.new_zeros((num_agents, num_steps, 1))
        numerator.index_add_(0, receiver, gate * value)
        denominator.index_add_(0, receiver, gate)
        aggregated = numerator / (denominator + self.eps)
        residual = self.output(aggregated)
        residual = torch.where(
            denominator > 0, residual, torch.zeros_like(residual))
        return residual.to(noisy_velocity_world.dtype)


__all__ = ["DependencyCorrector", "sinusoidal_timestep_embedding"]
