"""Read-only finite-support diagnostics. No model, optimizer or file deletion.

Input candidates must be frozen before truth is passed to this function.
The minimum over all M support points is an M-point diagnostic, NOT a K-point
prediction result and NOT an alternative deployment selector.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def support_gap_diagnostics(
    atoms: np.ndarray,
    selected_indices: np.ndarray,
    truth: np.ndarray,
    valid: np.ndarray | None = None,
    radii: tuple[float, ...] = (0.25, 0.5, 1.0),
) -> dict[str, Any]:
    """Score one scene: atoms [M,N,2], selected_indices [K,N], truth [N,2].

    Returns arrays over valid agents. Aggregate these arrays across scenes
    before computing dataset means/quantiles; do not average scene quantiles.
    Selection indices must have been obtained without consulting truth.
    """
    x = np.asarray(atoms, dtype=np.float64)
    y = np.asarray(truth, dtype=np.float64)
    ids = np.asarray(selected_indices)
    if x.ndim != 3 or x.shape[-1] != 2 or x.shape[0] < 1:
        raise ValueError('atoms must have shape [M,N,2], M>=1')
    m, n, _ = x.shape
    if y.shape != (n, 2):
        raise ValueError('truth must have shape [N,2]')
    if ids.ndim != 2 or ids.shape[1] != n or not 1 <= ids.shape[0] <= m:
        raise ValueError('selected_indices must have shape [K,N], 1<=K<=M')
    if not np.issubdtype(ids.dtype, np.integer):
        raise TypeError('selection indices must be integers, not coordinates')
    mask = np.ones(n, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    if mask.shape != (n,) or not mask.any():
        raise ValueError('valid must select at least one of N agents')
    x, y, ids = x[:, mask], y[mask], ids[:, mask]
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('valid support/truth coordinates must be finite')
    if (ids < 0).any() or (ids >= m).any():
        raise ValueError('selected index is not in this support')
    if any(len(np.unique(ids[:, i])) != len(ids) for i in range(ids.shape[1])):
        raise ValueError('indices must be unique per agent; coordinates may repeat')
    if any(not np.isfinite(r) or r <= 0 for r in radii):
        raise ValueError('radii must be positive and finite')

    # Only evaluation below uses truth. There is no selection based on truth.
    distances = np.linalg.norm(x - y[None], axis=-1)
    pool_error = distances.min(axis=0)
    chosen_error = np.take_along_axis(distances, ids, axis=0).min(axis=0)
    gap = chosen_error - pool_error
    if (gap < -1e-12).any():
        raise AssertionError('subset minimum cannot be below full-pool minimum')
    if not np.isclose(chosen_error.mean(), pool_error.mean() + gap.mean(), atol=1e-12):
        raise AssertionError('mean decomposition failed')

    partition: dict[str, Any] = {}
    for radius in radii:
        selected_hit = chosen_error <= radius
        pool_hit_selection_miss = (pool_error <= radius) & ~selected_hit
        pool_miss = pool_error > radius
        counts = np.array([selected_hit.sum(), pool_hit_selection_miss.sum(), pool_miss.sum()])
        if counts.sum() != len(pool_error) or np.any(selected_hit & pool_miss):
            raise AssertionError('failure classes must be disjoint and exhaustive')
        partition[str(radius)] = {
            'selected_hit_count': int(counts[0]),
            'pool_hit_selection_miss_count': int(counts[1]),
            'pool_miss_count': int(counts[2]),
            'selected_hit_rate': float(selected_hit.mean()),
            'pool_hit_selection_miss_rate': float(pool_hit_selection_miss.mean()),
            'pool_miss_rate': float(pool_miss.mean()),
        }
    return {
        'scope': 'same-support diagnostic only; not a K-point deployable oracle',
        'support_size': m,
        'selected_count': len(ids),
        'valid_occurrences': len(pool_error),
        'pool_error': pool_error,
        'chosen_error': chosen_error,
        'compression_gap': gap,
        'mean_pool_error': float(pool_error.mean()),
        'mean_chosen_error': float(chosen_error.mean()),
        'mean_compression_gap': float(gap.mean()),
        'partition': partition,
    }


def _self_test() -> dict[str, Any]:
    # Three agents: selected hit, finite-support hit missed by selection,
    # and finite-support miss. Truth is never used to construct ids.
    atoms = np.array([
        [[0.1, 0.0], [1.0, 0.0], [1.0, 0.0]],
        [[1.0, 0.0], [0.1, 0.0], [2.0, 0.0]],
        [[2.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
    ])
    ids = np.zeros((1, 3), dtype=np.int64)
    result = support_gap_diagnostics(atoms, ids, np.zeros((3, 2)))
    assert result['partition']['0.5']['selected_hit_count'] == 1
    assert result['partition']['0.5']['pool_hit_selection_miss_count'] == 1
    assert result['partition']['0.5']['pool_miss_count'] == 1

    # All points chosen -> exact zero gap.
    all_ids = np.tile(np.arange(3, dtype=np.int64)[:, None], (1, 3))
    full = support_gap_diagnostics(atoms, all_ids, np.zeros((3, 2)))
    assert np.array_equal(full['compression_gap'], np.zeros(3))

    # Masked-out padding may contain invalid values and is never scored.
    padded = np.concatenate((atoms, np.full((3, 1, 2), np.nan)), axis=1)
    padded_ids = np.concatenate((ids, np.full((1, 1), -1)), axis=1)
    masked = support_gap_diagnostics(padded, padded_ids,
                                    np.concatenate((np.zeros((3, 2)), [[np.nan, np.nan]])),
                                    np.array([True, True, True, False]))
    assert np.array_equal(masked['chosen_error'], result['chosen_error'])

    # Reject an invalid support identity and invalid types, not silently repair.
    invalid_rejected = 0
    for bad in [np.full((1, 3), 3), np.zeros((1, 3), dtype=float)]:
        try:
            support_gap_diagnostics(atoms, bad, np.zeros((3, 2)))
        except (ValueError, TypeError):
            invalid_rejected += 1
    assert invalid_rejected == 2
    return {
        'status': 'PASS',
        'checks': ['three-way partition', 'mean decomposition', 'full support zero gap',
                   'padding exclusion', 'invalid index rejection', 'noninteger rejection'],
        'data': 'synthetic only',
        'model_weights_loaded': False,
        'real_optimizer_updates': 0,
        'remote_server_access': False,
    }


if __name__ == '__main__':
    receipt = _self_test()
    print(json.dumps(receipt, indent=2, ensure_ascii=False))
    Path(__file__).with_name('REFERENCE_SELFTEST.json').write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
