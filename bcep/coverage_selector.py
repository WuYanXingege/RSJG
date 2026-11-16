"""Reference budgeted medoid selection. No model training or filesystem cleanup.

Finite support x_j with nonnegative normalized masses w_j. Select K support
points greedily to reduce sum_j w_j * min_s ||x_j-s||_2. This is greedy
maximization of the normalized monotone submodular benefit D - distortion,
not a global K-median solver and not a guarantee about test-set minFDE.

Runtime O(K*M^2), memory O(M^2). CPU NumPy reference, not a benchmark of
training/inference speed. Samples drawn from q use equal masses: do not
multiply their masses by q(x_j) a second time.
"""
from __future__ import annotations
from itertools import combinations
import json
import numpy as np


def select_goals(atoms: np.ndarray, k: int,
                 masses: np.ndarray | None = None) -> dict[str, np.ndarray | float]:
    x = np.asarray(atoms, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 2 or len(x) == 0:
        raise ValueError('atoms must be a nonempty [M, 2] array')
    if not np.isfinite(x).all():
        raise ValueError('all coordinates must be finite')
    m = len(x)
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or not 1 <= k <= m:
        raise ValueError('k must be an integer in [1, M]')
    # Unique IDs are selected. Repeated coordinates are not silently jittered.
    w = np.full(m, 1.0 / m) if masses is None else np.asarray(masses, dtype=np.float64)
    if w.shape != (m,) or not np.isfinite(w).all() or np.any(w < 0) or w.sum() <= 0:
        raise ValueError('masses must be finite, nonnegative, length M, and sum to >0')
    w = w / w.sum()
    dist = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1)
    diameter = float(dist.max())
    best = np.full(m, diameter)
    selected: list[int] = []
    used = np.zeros(m, dtype=bool)
    distortions: list[float] = []
    for _ in range(k):
        improvement = np.maximum(best[:, None] - dist, 0.0)
        gain = (w[:, None] * improvement).sum(axis=0)
        gain[used] = -np.inf
        index = int(gain.argmax())  # Stable tie-breaking by the original index.
        used[index] = True
        selected.append(index)
        best = np.minimum(best, dist[:, index])
        distortions.append(float(w @ best))
    ids = np.asarray(selected, dtype=np.int64)
    ownership = dist[:, ids].argmin(axis=1)
    represented_mass = np.bincount(ownership, weights=w, minlength=k)
    return dict(indices=ids, goals=x[ids], represented_mass=represented_mass,
                distortion_by_prefix=np.asarray(distortions), diameter=diameter)


def _self_test() -> dict:
    rng = np.random.default_rng(20261116)
    checked = 0
    worst_ratio = 1.0
    for trial in range(30):
        x = rng.normal(size=(9, 2))
        w = rng.random(9); w /= w.sum()
        k = 3
        r = select_goals(x, k, w)
        d = np.linalg.norm(x[:, None] - x[None], axis=-1)
        optimum = min(float(w @ d[:, ids].min(axis=1)) for ids in combinations(range(9), k))
        f_opt = float(r['diameter']) - optimum
        f_greedy = float(r['diameter']) - r['distortion_by_prefix'][-1]
        ratio = f_greedy / f_opt if f_opt > 1e-12 else 1.0
        guarantee = 1.0 - (1.0 - 1.0 / k) ** k
        assert ratio + 1e-10 >= guarantee
        assert np.all(np.diff(r['distortion_by_prefix']) <= 1e-12)
        assert len(np.unique(r['indices'])) == k
        assert abs(r['represented_mass'].sum() - 1.0) < 1e-12
        assert np.array_equal(select_goals(x, 1, w)['indices'], r['indices'][:1])
        # Independent brute-force check of every greedy step, not only its bound.
        previous = float(r['diameter'])
        chosen = []
        for step, idx in enumerate(r['indices']):
            remaining = [a for a in range(9) if a not in chosen]
            risks = [float(w @ d[:, chosen + [a]].min(axis=1)) for a in remaining]
            assert int(idx) == remaining[int(np.argmin(risks))]
            chosen.append(int(idx))
        worst_ratio = min(worst_ratio, float(ratio)); checked += 1
    collapsed = select_goals(np.zeros((6, 2)), 4)
    assert np.array_equal(collapsed['indices'], np.arange(4))
    assert np.all(collapsed['distortion_by_prefix'] == 0)
    try:
        select_goals(np.ones((2, 2)), 3)
        raise AssertionError('invalid k was not rejected')
    except ValueError:
        pass
    p, K = .9, 20
    q = p ** (1 / (K - 1)) / (p ** (1 / (K - 1)) + (1 - p) ** (1 / (K - 1)))
    return {'status': 'PASS', 'random_finite_support_cases': checked,
            'min_greedy_benefit_ratio': worst_ratio,
            'finite_k_guarantee_for_k3': 1 - (1 - 1/3)**3,
            'two_point_risk_optimum_p0_9_K20': q,
            'scope': 'synthetic selector tests only; no EBJD training, deletion, or replay'}


if __name__ == '__main__':
    print(json.dumps(_self_test(), indent=2))
