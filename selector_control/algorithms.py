"""Training-free selector controls for finite endpoint supports.

Adapted from the user-supplied selector_reference.py; frozen RSJG Greedy remains untouched.
OneSwap finds the exact minimum numerical swap cost, then applies the fixed acceptance epsilon.
KMeans additionally rejects nonfinite/invalid settings. No algorithm tuning.
Inputs are support coordinates only: no ground truth, density weights, file I/O,
model loading or optimizer. Draws from the model have equal masses here.
K-means returns centers; greedy and one-swap return support-subset indices.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SwapResult:
    indices: np.ndarray
    before: float
    after: float
    accepted: bool
    removed_position: int | None
    added_index: int | None


def _points(value: np.ndarray) -> np.ndarray:
    x = np.asarray(value, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 2 or len(x) == 0:
        raise ValueError('Expected a nonempty [M, 2] endpoint support.')
    if not np.isfinite(x).all():
        raise ValueError('Support contains nonfinite coordinates.')
    return x


def _k(value: int, m: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError('k must be an integer, not bool.')
    k = int(value)
    if not 1 <= k <= m:
        raise ValueError('Require 1 <= k <= M.')
    return k


def pairwise_l2(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x, y = _points(x), _points(y)
    d = np.linalg.norm(x[:, None, :] - y[None, :, :], axis=-1)
    if not np.isfinite(d).all():
        raise ValueError('Distances overflowed; verify units/coordinates.')
    return d


def mean_cover_distance(atoms: np.ndarray, centers: np.ndarray) -> float:
    """Empirical support distortion, not ground-truth minFDE."""
    return float(pairwise_l2(atoms, centers).min(axis=1).mean())


def greedy_reference(atoms: np.ndarray, k: int) -> np.ndarray:
    """Equal-mass greedy medoid selection; stable lowest-index tie breaking.

    Use the project's original frozen implementation for historical regression.
    """
    x = _points(atoms)
    k = _k(k, len(x))
    d = pairwise_l2(x, x)
    best = np.full(len(x), d.max(), dtype=np.float64)
    selected: list[int] = []
    for _ in range(k):
        gain = np.maximum(best[:, None] - d, 0.0).mean(axis=0)
        gain[selected] = -np.inf
        idx = int(np.argmax(gain))
        selected.append(idx)
        best = np.minimum(best, d[:, idx])
    return np.asarray(selected, dtype=np.int64)


def one_swap_polish(atoms: np.ndarray, selected: np.ndarray,
                    tolerance: float = 1e-12) -> SwapResult:
    """At most ONE globally best single exchange of one selected support ID.

    Minimizes the same unsquared mean-distance objective as the greedy method.
    Ties are resolved by outgoing position, then incoming support index.
    This is NOT convergence of PAM/k-medoids and has no GT-improvement guarantee.
    """
    x = _points(atoms)
    ids = np.asarray(selected)
    if ids.ndim != 1 or len(ids) == 0 or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError('selected must be a nonempty 1D integer index array.')
    ids = ids.astype(np.int64, copy=True)
    if len(np.unique(ids)) != len(ids) or np.any(ids < 0) or np.any(ids >= len(x)):
        raise ValueError('selected must contain unique, in-range support IDs.')
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError('tolerance must be finite and nonnegative.')
    d = pairwise_l2(x, x)
    ds = d[:, ids]
    before = float(ds.min(axis=1).mean())
    best_cost, best_move = float("inf"), None
    if len(ids) == len(x):
        return SwapResult(ids, before, before, False, None, None)
    eps = tolerance * (1.0 + abs(before))
    for pos in range(len(ids)):
        remaining = np.delete(ds, pos, axis=1)
        other = remaining.min(axis=1) if remaining.shape[1] else np.full(len(x), np.inf)
        costs = np.minimum(other[:, None], d).mean(axis=0)
        costs[ids] = np.inf
        incoming = int(np.argmin(costs))
        cost = float(costs[incoming])
        if cost < best_cost:  # exact numerical ties retain earliest (position, ID)
            best_cost, best_move = cost, (pos, incoming)
    if best_move is None or not best_cost < before - eps:
        return SwapResult(ids, before, before, False, None, None)
    pos, incoming = best_move
    ids[pos] = incoming
    after = float(d[:, ids].min(axis=1).mean())
    if not after < before - eps:
        raise AssertionError('Accepted swap did not strictly reduce distortion.')
    return SwapResult(ids, before, after, True, pos, incoming)


def kmeans_centers(atoms: np.ndarray, k: int, seed: int,
                   n_init: int = 5, max_iter: int = 100,
                   tolerance_m: float = 1e-6) -> dict:
    """Plain k-means++/Lloyd control, not a reproduction of native GDTS TTST.

    Five fixed restarts are selected by within-support squared distortion only.
    Repeated centers in degenerate supports are retained and reported, not jittered.
    """
    x = _points(atoms)
    k = _k(k, len(x))
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or v < 1 for v in [n_init, max_iter]) or not np.isfinite(tolerance_m) or tolerance_m < 0:
        raise ValueError('Invalid k-means control settings.')
    rng = np.random.default_rng(seed)
    best = None
    for restart in range(n_init):
        chosen = [int(rng.integers(len(x)))]
        nearest_sq = np.sum((x - x[chosen[0]]) ** 2, axis=1)
        for _ in range(1, k):
            mass = nearest_sq.copy()
            mass[chosen] = 0.0
            total = float(mass.sum())
            if total <= 0.0:
                idx = int(np.flatnonzero(~np.isin(np.arange(len(x)), chosen))[0])
            else:
                idx = int(rng.choice(len(x), p=mass / total))
            chosen.append(idx)
            nearest_sq = np.minimum(nearest_sq, np.sum((x - x[idx]) ** 2, axis=1))
        centers = x[chosen].copy()
        iters = 0
        for iters in range(1, max_iter + 1):
            dist_sq = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
            assignment = dist_sq.argmin(axis=1)
            updated = centers.copy()
            empty = []
            for j in range(k):
                cluster = x[assignment == j]
                if len(cluster):
                    updated[j] = cluster.mean(axis=0)
                else:
                    empty.append(j)
            # Deterministic farthest-support repair, no ground truth or jitter.
            residual = dist_sq.min(axis=1).copy()
            for j in empty:
                idx = int(np.argmax(residual))
                updated[j] = x[idx]
                residual = np.minimum(residual, ((x - x[idx]) ** 2).sum(axis=1))
                residual[idx] = -np.inf
            shift = np.linalg.norm(updated - centers, axis=1).max()
            centers = updated
            if shift <= tolerance_m:
                break
        final_sq = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
        inertia = float(final_sq.min(axis=1).mean())
        if best is None or inertia < best['mean_squared_distortion']:
            best = dict(centers=centers.copy(), mean_squared_distortion=inertia,
                        chosen_restart=restart, iterations=iters,
                        exact_unique_centers=int(len(np.unique(centers, axis=0))))
    assert best is not None
    best['mean_cover_distance'] = mean_cover_distance(x, best['centers'])
    best['settings'] = dict(n_init=n_init, max_iter=max_iter, tolerance_m=tolerance_m,
                            seed=int(seed))
    return best
