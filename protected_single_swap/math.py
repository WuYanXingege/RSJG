"""Observed-candidate-only FP64 whole-scene single replacement and protection.

The coverage reduction always sums in canonical (raw sample, valid agent)
order. No tolerance is used to decide a positive gain or break a gain tie.
"""
import numpy as np


def _array(x):
    if hasattr(x, 'detach'):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _valid_points(points, valid=None):
    x = _array(points)
    if x.ndim != 3 or x.shape[-1] != 2:
        raise ValueError('endpoints must be [samples, agents, 2]')
    mask = np.ones(x.shape[1], dtype=bool) if valid is None else _array(valid).astype(bool)
    if mask.shape != (x.shape[1],) or not mask.any():
        raise ValueError('at least one valid agent is required')
    x = np.asarray(x[:, mask], dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError('valid endpoints must be finite')
    return x


def _ids(ids, m):
    ids = _array(ids)
    if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError('IDs must be a one-dimensional integer array')
    ids = ids.astype(np.int64, copy=False)
    if len(ids) == 0 or len(np.unique(ids)) != len(ids) or ids.min() < 0 or ids.max() >= m:
        raise ValueError('selected IDs must be nonempty, unique and in range')
    return ids


def _distance(a, b):
    difference = a - b
    return np.sqrt(np.square(difference).sum(axis=-1, dtype=np.float64))


def coverage_cost(raw, ids, valid=None):
    """Direct FP64 P(S), for independent final recomputation or small tests."""
    x = _valid_points(raw, valid)
    ids = _ids(ids, len(x))
    nearest = _distance(x[:, None], x[ids][None]).min(axis=1)
    return float(np.ascontiguousarray(nearest).reshape(-1).mean(dtype=np.float64))


def rank_positive_swaps(raw, ids, valid=None, chunk_size=64):
    """Enumerate all K*(M-K) swaps, return exact strictly-positive candidates.

    Return dict:
      p0: float, original P(S0)
      total_swaps: int, K*(M-K)
      swaps: list of dict(slot, out_id, in_id, p, gain), sorted by
             (-gain, out_id, in_id).

    `slot` is the original S0 position. Insert `in_id` at that slot, retaining
    the other K-1 slots. `p` uses the same FP64 distance values and canonical
    summation order as coverage_cost. Cached nearest and second nearest
    distances avoid recomputing all K representatives for every replacement.
    """
    x = _valid_points(raw, valid)
    ids = _ids(ids, len(x))
    if not isinstance(chunk_size, (int, np.integer)) or chunk_size < 1:
        raise ValueError('chunk_size must be a positive integer')
    m, k = len(x), len(ids)
    # [raw reference, potential representative, valid agent].
    distances = _distance(x[:, None], x[None])
    selected = distances[:, ids]
    nearest = selected.min(axis=1)
    owners = selected.argmin(axis=1)
    second = (np.partition(selected, 1, axis=1)[:, 1]
              if k > 1 else np.full_like(nearest, np.inf))
    p0 = float(np.ascontiguousarray(nearest).reshape(-1).mean(dtype=np.float64))
    outside = np.setdiff1d(np.arange(m, dtype=np.int64), ids)
    swaps = []
    for slot, out_id in enumerate(ids):
        without = np.where(owners == slot, second, nearest)
        for start in range(0, len(outside), chunk_size):
            incoming = outside[start:start+chunk_size]
            # [incoming representative, raw reference, valid agent].
            alternatives = np.ascontiguousarray(distances[:, incoming].transpose(1, 0, 2))
            costs = np.minimum(alternatives, without[None]).reshape(len(incoming), -1).mean(axis=1, dtype=np.float64)
            for in_id, p in zip(incoming, costs):
                gain = float(p0 - p)
                if gain > 0.:
                    swaps.append(dict(slot=int(slot), out_id=int(out_id), in_id=int(in_id),
                                      p=float(p), gain=gain))
    swaps.sort(key=lambda row: (-row['gain'], row['out_id'], row['in_id']))
    return dict(p0=p0, total_swaps=k*(m-k), swaps=swaps)


def directed_distance(old, new, valid=None):
    """FP64 max_old min_new mean_valid_agent endpoint displacement (meters)."""
    a = _valid_points(old, valid)
    b = _valid_points(new, valid)
    if a.shape[1:] != b.shape[1:] or not len(a) or not len(b):
        raise ValueError('old/new must use the same valid agents and be nonempty')
    pair = _distance(a[:, None], b[None]).mean(axis=-1, dtype=np.float64)
    return float(pair.min(axis=1).max())
