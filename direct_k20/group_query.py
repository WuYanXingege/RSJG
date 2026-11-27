"""Add one observed-group query offset before the native personal encoder."""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .group_slot_recombination.matcher import groups_from_observed
from .model import K, Student


@dataclass(frozen=True)
class GroupInputs:
    """Packed group gather indices, with each singleton repeated in members."""

    group_index: torch.Tensor
    members: torch.Tensor

    def to(self, device):
        return GroupInputs(self.group_index.to(device), self.members.to(device))


def prepare_groups(observed):
    """Reuse the fixed observed-only greedy partition."""
    if torch.is_tensor(observed):
        device = observed.device
        observed = observed.detach().cpu().numpy()
    else:
        device = 'cpu'
    indices = groups_from_observed(observed)
    order = np.argsort(indices, kind='stable')
    sizes = np.bincount(indices)
    starts = np.cumsum(sizes) - sizes
    members = np.stack((order[starts], order[starts + sizes - 1]), axis=-1)
    return GroupInputs(torch.from_numpy(indices), torch.from_numpy(members)).to(device)


class GroupQueryStudent(Student):
    """The native student with group-specific offsets to each shared slot."""

    def __init__(self):
        super().__init__()
        self.group_query = nn.Sequential(nn.Linear(32 + 128 + 4, 64), nn.SiLU(), nn.Linear(64, 32))
        nn.init.zeros_(self.group_query[-1].weight)
        nn.init.zeros_(self.group_query[-1].bias)

    def load_s0_state_dict(self, state):
        """Strictly load every native tensor and keep the new module's initialization."""
        group = {f'group_query.{name}': value for name, value in self.group_query.state_dict().items()}
        if set(state) != set(self.state_dict()) - set(group):
            raise ValueError('S0 checkpoint must contain exactly the native student tensors')
        self.load_state_dict(dict(state) | group, strict=True)

    def query_offsets(self, observed, context, groups):
        """Compute [K,G,32] offsets from the same observations as the native student."""
        left, right = groups.members.to(observed.device).unbind(-1)
        h = (context[left] + context[right]) / 2
        position = observed[:, -1]
        recent_velocity = (position - observed[:, -4]) / 1.2
        speed = torch.linalg.vector_norm(observed.diff(dim=1), dim=-1).mean(dim=1) / .4
        motion = torch.stack((
            1 + (left != right).to(observed.dtype),
            (speed[left] + speed[right]) / 2,
            torch.linalg.vector_norm(position[left] - position[right], dim=-1),
            torch.linalg.vector_norm(recent_velocity[left] - recent_velocity[right], dim=-1),
        ), dim=-1)
        observation_features = torch.cat((h, motion), dim=-1)
        inputs = torch.cat((
            self.queries[:, None].expand(-1, len(left), -1),
            observation_features[None].expand(K, -1, -1),
        ), dim=-1)
        return self.group_query(inputs)

    def _queries_by_agent(self, observed, context, groups):
        groups = prepare_groups(observed) if groups is None else groups
        offsets = self.query_offsets(observed, context, groups)
        by_agent = offsets.index_select(1, groups.group_index.to(offsets.device))
        return self.queries[:, None] + by_agent


@torch.no_grad()
def summarize_query_offsets(offsets):
    """Summarize L2 norms in query space and within-slot distances across groups."""
    group_count = offsets.shape[1]
    pairs = torch.triu_indices(group_count, group_count, offset=1, device=offsets.device)
    norms = torch.linalg.vector_norm(offsets, dim=-1)
    pair_distances = torch.linalg.vector_norm(offsets[:, pairs[0]] - offsets[:, pairs[1]], dim=-1)
    return dict(
        groups=group_count,
        dq_norm_mean=float(norms.mean()),
        dq_group_pair_mean=float(pair_distances.mean()) if pair_distances.numel() else 0.,
        dq_group_pair_max=float(pair_distances.max()) if pair_distances.numel() else 0.,
    )
