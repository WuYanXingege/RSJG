"""Position-preserving endpoint queries; research variant of the learned prior.

Retains the uniform mixture, NLL, reparameterization and unbounded CV residual
coordinates. Each component reads all 64 spatially encoded map tokens instead
of using only their mean. No hard crop support or external goal weights.
"""
from __future__ import annotations

import copy
import math
import torch
from torch import nn
from .learned_goal import LearnedGoalMixture


class SpatialGoalMixture(LearnedGoalMixture):
    def __init__(self, dim=192, components=20, min_std=0.05, max_std=4.0):
        super().__init__(dim, components, min_std, max_std)
        self.spatial_queries = nn.Parameter(torch.randn(components, dim) * 0.02)
        self.spatial_history = nn.Linear(dim, dim, bias=False)
        self.spatial_center = nn.Linear(2, dim, bias=False)
        self.spatial_key = nn.Linear(dim, dim, bias=False)
        self.spatial_value = nn.Linear(dim, dim, bias=False)
        self.spatial_norm = nn.LayerNorm(dim)
        self.spatial_output = nn.Linear(dim, 4, bias=False)
        # Exact functional migration at update zero. The readout learns first;
        # attention receives gradients as soon as this readout becomes nonzero.
        nn.init.zeros_(self.spatial_output.weight)

    def forward(self, context):
        with torch.autocast(device_type=context.social_h.device.type, enabled=False):
            memory = context.memory[..., 8:, :].float()
            if memory.shape[-2] != 64:
                raise ValueError('spatial goal head requires the existing 8x8 map memory')
            pooled = memory.mean(-2)
            hidden = self.trunk(torch.cat((context.social_h.float(), pooled), -1))
            shape = (*hidden.shape[:2], self.components, 2)
            mean = self.mean(hidden).reshape(shape)
            logit = self.scale(hidden).reshape(shape)
            query = (self.spatial_queries + self.spatial_history(context.social_h.float())[..., None, :]
                     + self.spatial_center(mean))
            # Attention acts only along spatial tokens, never across people/worlds.
            attention = torch.softmax(torch.einsum('bnkd,bntd->bnkt', query,
                self.spatial_key(memory)) / math.sqrt(query.shape[-1]), -1)
            value = self.spatial_value(memory)
            local = torch.einsum('bnkt,bntd->bnkd', attention, value)
            correction = self.spatial_output(self.spatial_norm(local - value.mean(-2, keepdim=True)))
            mean = mean + correction[..., :2]
            std = self.min_std + (self.max_std - self.min_std) * (logit + correction[..., 2:]).sigmoid()
        return mean, std


def install_spatial_goal(model, optimizer=None, optimizer_factory=None):
    """Explicit migration. Preserve every existing tensor and optimizer moment.

    New parameters have no historical Adam state. No strict=False loading and
    no claim of architecture-identical continuation. Also usable at random init.
    """
    old = model.goal
    old_parameters = dict(model.named_parameters())
    head = SpatialGoalMixture(old.mean.in_features, old.components, old.min_std, old.max_std)
    head = head.to(device=next(old.parameters()).device)
    state = head.state_dict()
    source = old.state_dict()
    additions = set(state) - set(source)
    if not additions or not all(key.startswith('spatial_') for key in additions):
        raise RuntimeError('unexpected goal parameter migration')
    for key, tensor in source.items():
        if key not in state or state[key].shape != tensor.shape:
            raise RuntimeError(f'incompatible inherited tensor: {key}')
        state[key] = tensor.detach().clone()
    head.load_state_dict(state, strict=True)
    model.goal = head
    if optimizer is None:
        return None
    if optimizer_factory is None:
        raise ValueError('optimizer migration needs the existing group factory')
    migrated = optimizer_factory(model)
    if len(migrated.param_groups) != len(optimizer.param_groups):
        raise RuntimeError('optimizer group count changed')
    for old_group, new_group in zip(optimizer.param_groups, migrated.param_groups):
        for key, value in old_group.items():
            if key != 'params': new_group[key] = copy.deepcopy(value)
    for name, parameter in model.named_parameters():
        previous = old_parameters.get(name)
        if previous is not None and previous in optimizer.state:
            migrated.state[parameter] = copy.deepcopy(optimizer.state[previous])
    return migrated
