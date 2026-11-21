"""Original Hungarian, exact FP32-lattice alternating-cycle uniqueness certificate."""
import numpy as np
import math
from ebjd import unified


def integer_scores(scores):
    """Every finite FP32 score is an exact integer times a common power of two."""
    s=np.asarray(scores,dtype=np.float32)
    if s.ndim!=2 or s.shape[0]!=s.shape[1] or not np.isfinite(s).all():
        raise ValueError('finite square FP32 scores required')
    # Decode the IEEE-754 bits in bulk. Unlike float rounding or tolerance
    # tests, this conversion is exact even for subnormals and signed zero.
    bits=np.ascontiguousarray(s).view(np.uint32)
    exponent_bits=((bits>>23)&255).astype(np.int32)
    mantissa=(bits&0x7fffff)|np.where(exponent_bits>0,1<<23,0).astype(np.uint32)
    powers=np.where(exponent_bits>0,exponent_bits-150,-149)
    active=mantissa!=0
    if not active.any():return np.zeros(s.shape,np.int64),0
    minimum=int(powers[active].min());shifts=np.where(active,powers-minimum,0)
    signs=np.where(bits>>31,-1,1)
    if int(shifts.max())+24<=60:
        integers=(mantissa.astype(np.int64)<<shifts)*signs
    else:
        integers=np.array([int(m)*int(sign)*(1<<int(shift)) for m,sign,shift in zip(mantissa.ravel(),signs.ravel(),shifts.ravel())],dtype=object).reshape(s.shape)
    return integers,-minimum


def assignment_certificate(scores,chosen):
    """Minimum nontrivial alternating cycle = exact second-complete-match gap.

    For row u taking chosen[v], edge loss is S[u,chosen[u]]-S[u,chosen[v]].
    A distinct permutation is disjoint cycles; an optimal incumbent has no
    negative cycles, so the cheapest distinct permutation is one minimum cycle.
    Floyd-Warshall computes return paths; self edges are excluded at the end.
    Integer arithmetic resolves strict ties without score perturbations.
    """
    integer,exponent=integer_scores(scores);k=len(integer);chosen=np.asarray(chosen,dtype=np.int64)
    if not np.array_equal(np.sort(chosen),np.arange(k)):raise ValueError('not a bijection')
    selected=integer[np.arange(k),chosen]
    weights=selected[:,None]-integer[:,chosen]
    maximum=int(np.abs(weights).max())
    # Every shortest simple path has <=K-1 edges. The bound also covers each
    # intermediate sum. Fall back to unbounded Python integers for wide lattices.
    use_int64=maximum <= np.iinfo(np.int64).max//(8*max(k,1))
    w=weights.astype(np.int64) if use_int64 else weights.astype(object)
    distance=w.copy();np.fill_diagonal(distance,0)
    for pivot in range(k):
        np.minimum(distance,distance[:,pivot,None]+distance[None,pivot,:],out=distance)
        if distance.diagonal().min()<0:break
    cycles=w+distance.T
    numerator=int(cycles[~np.eye(k,dtype=bool)].min()) if k>1 else 0
    optimal=numerator>=0 and distance.diagonal().min()>=0
    return dict(unique=bool(optimal and numerator>0),native_solution_exactly_optimal=bool(optimal),
        gap=math.ldexp(float(numerator),-exponent),gap_integer=str(numerator),lattice_exponent=exponent,
        arithmetic='int64' if use_int64 else 'python_integer',hungarian_calls_for_uniqueness=0)


def solve(scores):
    s=np.asarray(scores,np.float32)
    row,col=unified.linear_sum_assignment(-s)  # SAME function/algorithm as native.
    chosen=np.empty(len(s),np.int64);chosen[row]=col
    return chosen,assignment_certificate(s,chosen)


def reference_gap(scores,chosen):
    """Original K forbidden-edge solves, FP64 values; audit reference only."""
    s=np.asarray(scores,np.float32).astype(np.float64);k=len(s)
    best=float(s[np.arange(k),chosen].sum());second=-np.inf
    for row,col in enumerate(chosen):
        blocked=s.copy();blocked[row,col]=-np.inf
        _,q=unified.linear_sum_assignment(-blocked)
        second=max(second,float(s[np.arange(k),q].sum()))
    return best-second
