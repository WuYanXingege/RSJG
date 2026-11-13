"""Trainable endpoint density for scratch EBJD; no checkpoint/bank dependency.

An equal-weight diagonal Gaussian mixture in train-fold-normalized CV residual
coordinates. One reparameterized draw per component forms 20 stratified
candidates. This is explicitly NOT 20 IID mixture draws or TTST.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from .unified import UnifiedCouplingBridge


class LearnedGoalMixture(nn.Module):
    def __init__(self, dim=192, components=20, min_std=0.05, max_std=4.0):
        super().__init__()
        if components != 20 or not 0 < min_std < max_std:
            raise ValueError("expected 20 components and ordered positive std bounds")
        self.components, self.min_std, self.max_std = components, min_std, max_std
        self.trunk = nn.Sequential(nn.LayerNorm(2 * dim), nn.Linear(2 * dim, dim), nn.SiLU())
        self.mean = nn.Linear(dim, components * 2)
        self.scale = nn.Linear(dim, components * 2)
        # Random small independent means break component symmetry; no prior weights.
        nn.init.normal_(self.mean.weight, std=0.01)
        nn.init.normal_(self.mean.bias, std=0.1)
        nn.init.normal_(self.scale.weight, std=0.001)
        initial_std = 0.75
        ratio = (initial_std - min_std) / (max_std - min_std)
        nn.init.constant_(self.scale.bias, math.log(ratio / (1 - ratio)))

    def forward(self, context):
        # Shared history/social and spatial map-memory features, no future inputs.
        with torch.autocast(device_type=context.social_h.device.type, enabled=False):
            feature = torch.cat((context.social_h.float(), context.memory[..., 8:, :].float().mean(-2)), -1)
            hidden = self.trunk(feature)
            shape = (*hidden.shape[:2], self.components, 2)
            mean = self.mean(hidden).reshape(shape)
            std = self.min_std + (self.max_std - self.min_std) * self.scale(hidden).reshape(shape).sigmoid()
        return mean, std

    def sample(self, context, goal_scale, *, generator=None, noise=None, distribution=None):
        mean, std = self(context) if distribution is None else distribution
        if noise is None:
            noise = torch.randn(mean.shape, device=mean.device, dtype=torch.float32, generator=generator)
        if noise.shape != mean.shape:
            raise ValueError("endpoint noise must be [B,N,20,2]")
        residual = (mean + std * noise).permute(0, 2, 1, 3)
        goals = context.baseline[:, None, :, -1, :].float() + goal_scale.float() * residual
        return goals.masked_fill(~context.valid[:, None, :, None], 0)

    def nll(self, context, truth, goal_scale, *, distribution=None):
        mean, std = self(context) if distribution is None else distribution
        target = (truth.float() - context.baseline[..., -1, :].float()) / goal_scale.float()
        standardized = (target[:, :, None] - mean) / std
        log_density = (-0.5 * standardized.square() - std.log() - 0.5 * math.log(2 * math.pi)).sum(-1)
        nll = math.log(self.components) - torch.logsumexp(log_density, dim=-1)
        # Density in normalized coordinates; scale is fixed from TRAIN only.
        return nll[context.valid].mean()


class LearnedGoalCouplingBridge(UnifiedCouplingBridge):
    def __init__(self, bridge, *, components=20, min_std=0.05, max_std=4.0):
        super().__init__(bridge, worlds=components, singleton_backward="identity")
        self.goal = LearnedGoalMixture(bridge.encoder.history_stem[0].out_features,
                                      components, min_std, max_std)
