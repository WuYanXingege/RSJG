"""Bounded CTF modulation from this call's actual goal and coarse physical path."""
import torch
from torch import nn


class GeometryReliabilityGate(nn.Module):
    def __init__(self, dim=192):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim + 10, 64), nn.SiLU(), nn.Linear(64, 1))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, context, trajectory, goals, time, base_gate, scale):
        b, p, n = goals.shape[:3]
        scale = scale.float().clamp_min(0.5)
        observed = context.observed.float()
        last = observed[:, :, -1]
        velocity = torch.cat(((trajectory[..., :1, :] - last[:, None, :, None]) / 0.4,
                              trajectory.diff(dim=-2) / 0.4), -2)
        dev = (trajectory - context.baseline[:, None]).square().sum(-1).mean(-1).add(1e-8).sqrt() / scale
        # Per-world synchronous distances, with no cross-world information.
        pos = trajectory.permute(0, 1, 3, 2, 4)
        dist = (pos[..., :, None, :] - pos[..., None, :, :]).square().sum(-1).add(1e-8).sqrt()
        legal = context.valid[:, :, None] & context.valid[:, None, :] & ~torch.eye(n, device=goals.device, dtype=torch.bool)
        near = dist.masked_fill(~legal[:, None, None], float('inf')).amin((2, 4))
        near = torch.where(torch.isfinite(near), near / scale, 0.0)
        features = torch.cat(((goals - last[:, None]) / scale,
                              (goals - context.baseline[:, None, :, -1]) / scale,
                              velocity.mean(-2) - context.mean_velocity[:, None],
                              dev[..., None], near[..., None],
                              time[:, None, None, None].expand(b, p, n, 1),
                              context.valid[:, None, :, None].expand(b, p, n, 1).float()), -1)
        r = self.net(torch.cat((context.social_h[:, None].expand(b, p, n, -1), features), -1)).squeeze(-1)
        gate = base_gate.reshape(b, 1, 1) * (1 + 0.5 * r.tanh())
        gate = gate * context.valid[:, None]
        # Arithmetic mean is symmetric; diagonal irrelevant to pair social bias.
        pair = (gate[..., :, None] + gate[..., None, :]) * 0.5
        pair = pair * (context.valid[:, None, :, None] & context.valid[:, None, None, :])
        return gate, pair
