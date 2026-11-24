"""Endpoint-only metrics and exact uniform-without-replacement expectations.

Endpoint errors and J follow native CPU FP32 conventions. L is the FP64 mean
of native per-agent minima, matching the linear expectation E[L]. This may
differ slightly from the native rounded FP32 scene_marginal_fde. Formal marginal
minFDE is aggregated using agent_min_sum (FP64), not by averaging window L.
Uniform expectations use FP64 rank probabilities; expected L is the mean of
per-agent expected minima. G is always J-L, without numerical clipping.
"""
from functools import lru_cache
from math import comb
import numpy as np
import torch


@lru_cache(maxsize=None)
def _rank_weights(m, k):
    if not isinstance(m, int) or not isinstance(k, int) or not 1 <= k <= m:
        raise ValueError('require integer 1 <= k <= m')
    denominator = comb(m, k)
    weights = np.zeros(m, dtype=np.float64)
    # Python's exact integer binomial coefficients avoid overflow/underflow of
    # intermediate floating factorials. Rank r is one-based in the formula.
    for r in range(1, m-k+2):
        weights[r-1] = comb(m-r, k-1)/denominator
    if not np.isclose(weights.sum(), 1., rtol=0., atol=8*np.finfo(np.float64).eps):
        raise ArithmeticError('rank probabilities do not sum to one')
    weights.flags.writeable = False
    return weights


def rank_minimum_weights(m, k):
    """Return M probabilities for the minimum rank in a uniform K-subset."""
    return _rank_weights(m, k).copy()


def expected_uniform_min(values, k=20, axis=0):
    """Exact finite-population expected minimum, evaluated in float64."""
    if torch.is_tensor(values):
        values = values.detach().cpu().numpy()
    a = np.asarray(values, dtype=np.float64)
    if a.ndim == 0 or not np.isfinite(a).all():
        raise ValueError('finite non-scalar values required')
    axis = np.core.numeric.normalize_axis_index(axis, a.ndim)
    sorted_values = np.sort(np.moveaxis(a, axis, 0), axis=0)
    w = _rank_weights(len(sorted_values), k)
    return np.tensordot(w, sorted_values, axes=(0, 0))


def _valid(valid, n):
    if valid is None:
        mask = torch.ones(n, dtype=torch.bool)
    else:
        mask = torch.as_tensor(valid, device='cpu', dtype=torch.bool)
    if mask.shape != (n,) or not mask.any():
        raise ValueError('valid mask must select at least one agent')
    return mask


def _error_tensor(errors, valid=None):
    e = torch.as_tensor(errors, device='cpu').detach()
    if e.ndim != 2 or e.shape[0] < 1:
        raise ValueError('errors must be [samples, agents]')
    mask = _valid(valid, e.shape[1])
    e = e[:, mask]
    if not torch.isfinite(e).all() or (e < 0).any():
        raise ValueError('valid errors must be finite and nonnegative')
    return e


def endpoint_errors(endpoints, truth):
    """Native FP32 Euclidean endpoint errors, [K,N], with no ADE/DDIM work."""
    goals = torch.as_tensor(endpoints, device='cpu').detach().float()
    targets = torch.as_tensor(truth, device='cpu').detach().float()
    if goals.ndim != 3 or goals.shape[-1] != 2 or targets.shape != goals.shape[1:]:
        raise ValueError('require endpoints [K,N,2] and truth [N,2]')
    return torch.linalg.vector_norm(goals-targets[None], dim=-1)


def finite_error_statistics(errors, valid=None):
    """Finite K metrics; agent_min contains only valid agents in input order."""
    e = _error_tensor(errors, valid).float()
    # Same contiguous [batch, agent, sample] axes as hard_metric_statistics.
    f = e.T.contiguous()[None]
    n = e.shape[1]
    minima = f.min(2).values[0]
    L = float(minima.double().mean())
    J = float((f.sum(1)/float(n)).min(1).values[0])
    agent = minima.numpy().copy()
    return dict(L=L, J=J, G=J-L, agent_min=agent,
                agent_min_sum=float(minima.double().sum()), n=n)


def endpoint_statistics(endpoints, truth, valid=None):
    return finite_error_statistics(endpoint_errors(endpoints, truth), valid)


def expected_uniform_metrics(errors, k=20, valid=None, *, native_joint_reduction=True):
    """Exact K-subset expectation on a shared set of whole-scene sample IDs.

    Joint error per source scene is computed in native FP32 by default, then
    sorted and averaged with FP64 rank weights. E[L] uses linearity on FP64
    expectations of each agent's minima, matching finite window L in FP64.
    """
    e = _error_tensor(errors, valid)
    values = e.double().numpy()
    agent = np.asarray(expected_uniform_min(values, k=k, axis=0))
    if native_joint_reduction:
        joint = (e.float().T.contiguous()[None].sum(1)/float(e.shape[1]))[0]
    else:
        joint = values.mean(1)
    J = float(expected_uniform_min(joint, k=k, axis=0))
    L = float(agent.mean())
    return dict(L=L, J=J, G=J-L, agent_min=agent,
                agent_min_sum=float(agent.sum()), n=e.shape[1])


def motion_groups(observed, valid=None):
    """Fixed speed definition: total seven-segment length / 2.8 s, low<=0.2."""
    obs = torch.as_tensor(observed, device='cpu').detach()
    if obs.ndim != 3 or obs.shape[1:] != (8, 2):
        raise ValueError('observed must be [N,8,2]')
    mask = _valid(valid, obs.shape[0])
    x = obs[mask].double().numpy()
    if not np.isfinite(x).all():
        raise ValueError('observations must be finite')
    speeds = np.linalg.norm(np.diff(x, axis=1), axis=-1).sum(1)/2.8
    low = speeds <= .2
    n = len(speeds)
    groups = ['all']
    if low.all():
        groups.append('all_low_speed')
    elif low.any():
        groups.append('mixed')
        if 2 <= n <= 8:
            groups.append('mixed_N2_8')
    else:
        groups.append('all_moving')
        if 2 <= n <= 8:
            groups.append('all_moving_N2_8')
    return dict(groups=groups, speeds_m_s=speeds, n=n,
                low_count=int(low.sum()), high_count=int((~low).sum()), threshold_m_s=.2)
