"""Float64 analysis of captured FP32 scores; never alters the native matching scores."""
import numpy as np
from scipy.optimize import linear_sum_assignment


def value(scores, pi):
    return float(np.asarray(scores,dtype=np.float64)[np.arange(len(pi)),pi].sum())


def solve(scores):
    row,col = linear_sum_assignment(-np.asarray(scores))
    pi = np.empty(len(row),dtype=np.int64);pi[row]=col
    return pi


def second_best_gap(scores, chosen):
    """Exact second distinct assignment: forbid each chosen edge, take the best.

    Every distinct permutation excludes at least one chosen edge. Summation is
    float64 over the unchanged FP32 entries. This is NOT a rowwise top-two gap.
    """
    s = np.asarray(scores,dtype=np.float64)
    best = value(s,chosen)
    second = -float('inf'); runner = None
    for row,col in enumerate(chosen):
        blocked = s.copy();blocked[row,col] = -np.inf
        pi = solve(blocked);score = value(s,pi)
        if score > second:
            second,runner = score,pi
    gap = best-second
    if gap < 0:
        raise ValueError(f'Native selected assignment not optimal: gap={gap}')
    return gap,runner


def center(scores):
    s = np.asarray(scores,dtype=np.float64)
    return s-s.mean(-1,keepdims=True)-s.mean(-2,keepdims=True)+s.mean((-1,-2),keepdims=True)


def describe(values):
    x=np.asarray(values,dtype=np.float64).ravel()
    x=x[np.isfinite(x)]
    if not x.size:return dict(count=0)
    return dict(count=int(x.size),mean=float(x.mean()),std=float(x.std()),minimum=float(x.min()),maximum=float(x.max()),
        p01=float(np.quantile(x,.01)),p05=float(np.quantile(x,.05)),median=float(np.median(x)),
        p95=float(np.quantile(x,.95)),p99=float(np.quantile(x,.99)),mean_abs=float(np.abs(x).mean()),
        rms=float(np.sqrt(np.mean(x*x))))


def assignment_change(before,after):
    """[world, person], quotient out one common world relabeling using person 0."""
    a,b=np.asarray(before),np.asarray(after)
    k,n=a.shape
    assert np.array_equal(np.sort(a,axis=0),np.broadcast_to(np.arange(k)[:,None],a.shape))
    assert np.array_equal(np.sort(b,axis=0),np.broadcast_to(np.arange(k)[:,None],b.shape))
    aligned_a=a[np.argsort(a[:,0])];aligned_b=b[np.argsort(b[:,0])]
    raw=int(np.count_nonzero(a!=b));relative=int(np.count_nonzero(aligned_a!=aligned_b))
    return dict(changed_choices=raw,total_choices=k*n,changed_agents=int(np.any(a!=b,axis=0).sum()),
        relative_changed_choices=relative,uniform_world_relabel_only=bool(raw>0 and relative==0))


def gap_bin(gap):
    if gap is None:return 'singleton_bypass'
    if gap==0:return 'strict_tie_0'
    if gap<=1e-6:return 'near_(0,1e-6]'
    if gap<=1e-5:return '(1e-6,1e-5]'
    if gap<=1e-4:return '(1e-5,1e-4]'
    if gap<=1e-3:return '(1e-4,1e-3]'
    return '>1e-3'


def people_bin(n):
    if n==1:return '1'
    if n<=4:return '2-4'
    if n<=8:return '5-8'
    if n<=12:return '9-12'
    if n<=20:return '13-20'
    return '21+'


def proximity_bin(x):
    if not np.isfinite(x):return 'no_neighbor'
    if x<.2:return '<0.2m'
    if x<.5:return '[0.2,0.5)m'
    if x<1:return '[0.5,1)m'
    return '>=1m'
