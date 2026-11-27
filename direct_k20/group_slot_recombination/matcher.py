"""Score permutations of native slots while keeping observed local groups intact."""

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn import functional as F

from bcep.motion_frame import frame
from direct_k20.interaction_diagnosis.statistics import greedy_observed_pairs
from direct_k20.model import FRAMES, K


DT = 0.4
QUERY_WIDTH = 32
HIDDEN_WIDTH = 64
ROW_WIDTH = 24
COLUMN_WIDTH = 22
DIAGONAL_BIAS = 0.1
SINKHORN_TEMPERATURE = 0.5
SINKHORN_STEPS = 20


def groups_from_observed(observed):
    """Get packed member-to-group indices using the fixed diagnostic partition."""
    positions = np.asarray(observed, dtype=np.float64)
    n = positions.shape[0]
    if n < 1 or positions.shape != (n, 8, 2) or not np.isfinite(positions).all():
        raise ValueError('expected one finite packed window of eight observed positions')
    pairs = greedy_observed_pairs(positions)
    paired = np.zeros(n, dtype=bool)
    paired[pairs.ravel()] = True
    members = sorted(
        [tuple(pair) for pair in pairs] + [(i,) for i in np.flatnonzero(~paired)],
        key=lambda group: group[0],
    )
    group_index = np.empty(n, dtype=np.int64)
    for group, indices in enumerate(members):
        group_index[list(indices)] = group
    return group_index


def window_features(observed, goals):
    """Construct row/column inputs from observations and native S0 goals only.

    The source column uses the receiving group's feature summary at source
    slot j. The destination row pools every other group at native slot k.
    Both group/member reductions are symmetric. Group ordinals are retained
    separately for exact gathering, and never provided to the matcher.
    """
    observed = np.asarray(observed, dtype=np.float64)
    goals = np.asarray(goals, dtype=np.float64)
    group_index = groups_from_observed(observed)
    n = len(group_index)
    if goals.shape != (K, n, 2) or not np.isfinite(goals).all():
        raise ValueError('native goals must be finite [20,N,2] aligned to the observation')

    # Follow the frozen student's FP32 motion-frame convention, including the
    # native positive-x fallback when the final three-segment mean is still.
    observed_native = torch.tensor(observed, dtype=torch.float32)
    _, rotation, _ = frame(observed_native, FRAMES, DT, observed_native.new_ones(()))
    rotation = rotation.numpy().astype(np.float64)
    position = observed[:, -1]
    recent_velocity = (position - observed[:, -4]) / (3 * DT)
    full_velocity = (position - observed[:, 0]) / (7 * DT)
    displacement = goals - position[None]
    local_displacement = np.einsum('knd,ndc->knc', displacement, rotation)
    cv_residual = displacement - (FRAMES * DT) * recent_velocity[None]
    local_residual = np.einsum('knd,ndc->knc', cv_residual, rotation)
    step_speed = np.linalg.norm(np.diff(observed, axis=1), axis=-1).mean(1) / DT
    centered_position = position - position.mean(axis=0)
    per_person_obs = np.column_stack((
        centered_position, recent_velocity, full_velocity,
        np.linalg.norm(recent_velocity, axis=-1),
        np.linalg.norm(recent_velocity - full_velocity, axis=-1),
        step_speed,
    ))

    count = int(group_index.max()) + 1
    obs_features, slot_features = [], []
    for group in range(count):
        member = np.flatnonzero(group_index == group)
        # A singleton contributes zeros to the three pair-only terms below.
        pair_position = pair_velocity = 0.0
        pair_distance_change = np.zeros(K)
        pair_motion = np.zeros((K, 2))
        if len(member) == 2:
            a, b = member
            pair_position = np.linalg.norm(position[b] - position[a])
            pair_velocity = np.linalg.norm(recent_velocity[b] - recent_velocity[a])
            separation = np.linalg.norm(goals[:, b] - goals[:, a], axis=-1)
            pair_distance_change = separation - pair_position
            difference = displacement[:, b] - displacement[:, a]
            directed = np.stack((difference, -difference), axis=1)
            in_each_frame = np.einsum('knd,ndc->knc', directed, rotation[member])
            pair_motion = np.abs(in_each_frame).mean(axis=1)
        obs_features.append(np.concatenate((
            per_person_obs[member].mean(axis=0),
            [len(member)/2, pair_position, pair_velocity],
        )))
        slot_features.append(np.column_stack((
            displacement[:, member].mean(axis=1),
            local_displacement[:, member].mean(axis=1),
            np.linalg.norm(displacement[:, member], axis=-1).mean(axis=1),
            pair_distance_change,
            pair_motion,
            local_residual[:, member].mean(axis=1),
        )))

    own_obs = np.asarray(obs_features)
    own_motion = np.asarray(slot_features)
    column = np.concatenate((np.broadcast_to(own_obs[:, None], (count, K, 12)), own_motion), axis=-1)
    row = np.zeros((count, K, ROW_WIDTH))
    if count > 1:
        others_obs = (own_obs.sum(axis=0) - own_obs) / (count - 1)
        others_motion = (own_motion.sum(axis=0) - own_motion) / (count - 1)
        centers = own_obs[:, :2]
        distance = np.linalg.norm(centers[:, None] - centers[None], axis=-1)
        distance_mean = distance.sum(axis=1) / (count - 1)
        np.fill_diagonal(distance, np.inf)
        distances = np.column_stack((distance_mean, distance.min(axis=1)))
        static = np.concatenate((others_obs, distances), axis=-1)
        row = np.concatenate((np.broadcast_to(static[:, None], (count, K, 14)), others_motion), axis=-1)
    return dict(row_features=row.astype(np.float32), column_features=column.astype(np.float32),
                group_index=group_index)


class Matcher(nn.Module):
    """A group-shared row/column scorer with non-additive slot interactions."""

    def __init__(self):
        super().__init__()
        self.destination = nn.Sequential(nn.Linear(ROW_WIDTH + QUERY_WIDTH, HIDDEN_WIDTH), nn.SiLU())
        self.source = nn.Sequential(nn.Linear(COLUMN_WIDTH, HIDDEN_WIDTH), nn.SiLU())
        self.row_hidden = nn.Linear(HIDDEN_WIDTH, HIDDEN_WIDTH, bias=False)
        self.column_hidden = nn.Linear(HIDDEN_WIDTH, HIDDEN_WIDTH, bias=False)
        self.interaction = nn.Linear(HIDDEN_WIDTH, HIDDEN_WIDTH)
        self.output = nn.Linear(HIDDEN_WIDTH, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, row_features, column_features, queries, group_valid=None):
        if (row_features.ndim not in (3, 4) or row_features.shape[-2:] != (K, ROW_WIDTH)
                or column_features.shape != (*row_features.shape[:-1], COLUMN_WIDTH)
                or queries.shape != (K, QUERY_WIDTH)):
            raise ValueError('expected padded or packed group features and 20 frozen slot queries')
        prefix = row_features.shape[:-2]
        if group_valid is not None:
            if group_valid.shape != prefix or group_valid.dtype != torch.bool:
                raise ValueError('group mask and features are not aligned')
            row_features = torch.where(group_valid[..., None, None], row_features, 0.)
            column_features = torch.where(group_valid[..., None, None], column_features, 0.)
        query = queries.detach().expand(*prefix, K, QUERY_WIDTH)
        destination = self.destination(torch.cat((row_features, query), dim=-1))
        source = self.source(column_features)
        # This is an affine map of [destination, source, destination*source].
        # Evaluate the row and column contributions before broadcasting.
        pairwise = (
            self.row_hidden(destination)[..., :, None, :]
            + self.column_hidden(source)[..., None, :, :]
            + self.interaction(destination[..., :, None, :] * source[..., None, :, :])
        )
        score = self.output(F.silu(pairwise)).squeeze(-1)
        if group_valid is not None:
            score = torch.where(group_valid[..., None, None], score, 0.)
        return score + DIAGONAL_BIAS * torch.eye(K, dtype=score.dtype, device=score.device)


def _active_groups(scores, group_valid):
    if scores.ndim not in (3, 4) or scores.shape[-2:] != (K, K):
        raise ValueError('expected [G,20,20] or [B,G,20,20] score matrices')
    if group_valid is None:
        group_valid = torch.ones(scores.shape[:-2], device=scores.device, dtype=torch.bool)
    if (group_valid.shape != scores.shape[:-2] or group_valid.dtype != torch.bool
            or group_valid.device != scores.device):
        raise ValueError('group mask and score matrices are not aligned')
    return group_valid & (group_valid.sum(dim=-1, keepdim=True) > 1)


def hard_permutation(scores, group_valid=None):
    """Maximize each group's score and return destination-to-source indices."""
    active = _active_groups(scores, group_valid).cpu().numpy()
    matrices = scores.detach().cpu().numpy()
    permutation = np.broadcast_to(np.arange(K), matrices.shape[:-1]).copy()
    for index in zip(*np.nonzero(active)):
        matrix = matrices[index]
        if not np.isfinite(matrix).all():
            raise ValueError('nonfinite assignment score')
        rows, columns = linear_sum_assignment(matrix, maximize=True)
        permutation[index][rows] = columns
    return torch.as_tensor(permutation, device=scores.device, dtype=torch.int64)


def log_sinkhorn(scores):
    """Twenty alternating row/column log-normalizations at fixed temperature."""
    normalized = scores / SINKHORN_TEMPERATURE
    for _ in range(SINKHORN_STEPS):
        normalized = normalized - normalized.logsumexp(dim=-1, keepdim=True)
        normalized = normalized - normalized.logsumexp(dim=-2, keepdim=True)
    return normalized.exp()


def straight_through_assignment(scores, group_valid=None):
    """Pair Hungarian forward values with log-Sinkhorn gradients."""
    active = _active_groups(scores, group_valid)
    permutation = hard_permutation(scores, group_valid)
    hard = F.one_hot(permutation, num_classes=K).to(scores.dtype)
    # Single-group windows and padding always use identity and carry no
    # matcher gradient. Normalize only matrices participating in matching.
    soft = log_sinkhorn(scores[active])
    assignment = hard.clone()
    assignment[active] = hard[active] + (soft - soft.detach())
    return assignment, permutation


def classify_permutation(permutation):
    """Distinguish identity, common renaming, and relative group moves."""
    permutation = np.asarray(permutation)
    if np.all(permutation == np.arange(K)):
        return 'identity'
    if np.all(permutation == permutation[:1]):
        return 'rename'
    return 'relative'


def permute_paths(paths, permutation, group_index, *, inverse=False):
    """Gather complete trajectories for each member with no arithmetic."""
    paths = np.asarray(paths)
    permutation = np.asarray(permutation)
    group_index = np.asarray(group_index)
    n = len(group_index)
    if (paths.shape != (K, n, FRAMES, 2) or n < 1
            or group_index.shape != (n,) or not np.issubdtype(group_index.dtype, np.integer)
            or permutation.ndim != 2 or permutation.shape[1] != K
            or not np.issubdtype(permutation.dtype, np.integer)
            or not np.array_equal(np.unique(group_index), np.arange(len(permutation)))
            or not np.all(np.sort(permutation, axis=-1) == np.arange(K))):
        raise ValueError('trajectories, groups, and slot bijections are not aligned')
    if inverse:
        permutation = np.argsort(permutation, axis=-1)
    return paths[permutation[group_index].T, np.arange(n)]
