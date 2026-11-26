"""Observed-only K20 selection over complete A and B trajectory samples."""

import numpy as np


K = 20
FRAMES = 12
REPLACEMENTS = 4
_PAIR_LEFT, _PAIR_RIGHT = np.triu_indices(K, k=1)


def _check_layout(a, b):
    if not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray):
        raise TypeError('trajectory inputs must be NumPy arrays')
    if (a.ndim != 4 or a.shape != b.shape or a.shape[0] != K or a.shape[1] < 1
            or a.shape[2:] != (FRAMES, 2)):
        raise ValueError('expected matching [20, N, 12, 2] trajectories with N >= 1')
    if a.dtype != np.float32 or b.dtype != np.float32:
        raise ValueError('trajectory sources must be FP32')


def _weighted_distance(left, right):
    """Return the mean/final displacement for each reference, choice and agent."""
    dx = left[..., 0] - right[..., 0]
    dy = left[..., 1] - right[..., 1]
    np.square(dx, out=dx)
    np.square(dy, out=dy)
    dx += dy
    np.sqrt(dx, out=dx)
    return 0.5 * dx.mean(axis=-1) + 0.5 * dx[..., -1]


def _within_model_distance(paths):
    # Euclidean displacement is symmetric. Diagonal matches have zero cost.
    pair = _weighted_distance(paths[_PAIR_LEFT], paths[_PAIR_RIGHT])
    distance = np.zeros((K, K, paths.shape[1]), dtype=np.float64)
    distance[_PAIR_LEFT, _PAIR_RIGHT] = pair
    distance[_PAIR_RIGHT, _PAIR_LEFT] = pair
    return distance


def select(a, b):
    """Return the fixed K20 slot selection and observed-only proxy diagnostics.

    Locations are zero-based source slots. Output order is retained B slots
    in ascending order, then A slots in their greedy admission order. A
    singleton uses all A slots in ascending order. Each objective is evaluated
    in FP64 and exact ties choose the smaller original source slot.
    """
    _check_layout(a, b)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('nonfinite source trajectory')
    if a.shape[1] == 1:
        return dict(slots=[['A', slot] for slot in range(K)], removed_B=[], added_A=[],
                    Jproxy_after_deletions_m=[], Mproxy_before_additions_m=None,
                    Mproxy_after_additions_m=[])

    a64 = a.astype(np.float64)
    b64 = b.astype(np.float64)
    joint_bb = _within_model_distance(b64).mean(axis=-1)
    kept_b = list(range(K))
    removed_b = []
    jproxy = []
    for _ in range(REPLACEMENTS):
        winner = min(
            (float(joint_bb[:, [other for other in kept_b if other != slot]].min(axis=1).mean()), slot)
            for slot in kept_b
        )
        cost, slot = winner
        kept_b.remove(slot)
        removed_b.append(slot)
        jproxy.append(cost)

    dist_ab = _weighted_distance(a64[:, None], b64[None, kept_b])
    nearest = dist_ab.min(axis=1)
    initial_mproxy = float(nearest.mean())
    dist_aa = _within_model_distance(a64)
    unused_a = list(range(K))
    added_a = []
    mproxy = []
    for _ in range(REPLACEMENTS):
        cost, slot = min(
            (float(np.minimum(nearest, dist_aa[:, slot, :]).mean()), slot)
            for slot in unused_a
        )
        nearest = np.minimum(nearest, dist_aa[:, slot, :])
        unused_a.remove(slot)
        added_a.append(slot)
        mproxy.append(cost)
    return dict(slots=[['B', slot] for slot in kept_b] + [['A', slot] for slot in added_a],
                removed_B=removed_b, added_A=added_a, Jproxy_after_deletions_m=jproxy,
                Mproxy_before_additions_m=initial_mproxy, Mproxy_after_additions_m=mproxy)


def gather(a, b, slots):
    """Copy the sealed source slots without changing agents, frames or coordinates."""
    _check_layout(a, b)
    sources = {'A': a, 'B': b}
    keys = [(source, slot) for source, slot in slots]
    if (len(keys) != K or len(set(keys)) != K
            or any(source not in sources or not isinstance(slot, (int, np.integer))
                   or not 0 <= slot < K for source, slot in keys)):
        raise ValueError('expected K20 distinct, valid source slots')
    expected_a = K if a.shape[1] == 1 else REPLACEMENTS
    if sum(source == 'A' for source, _ in keys) != expected_a:
        raise ValueError('invalid A/B ratio for window population')
    return np.ascontiguousarray(np.stack([sources[source][slot] for source, slot in keys]))
