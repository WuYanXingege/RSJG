"""One fixed endpoint Hungarian rule; entire A trajectories move as byte vectors."""
import time
import numpy as np
from scipy.optimize import linear_sum_assignment


def repair_trajectories(a_paths, b_end, valid):
    a_paths=np.asarray(a_paths);b_end=np.asarray(b_end);valid=np.asarray(valid,dtype=bool)
    assert a_paths.dtype==np.float32 and b_end.dtype==np.float32
    k,n,t,xy=a_paths.shape
    assert (k,t,xy)==(20,12,2) and b_end.shape==(k,n,2) and valid.shape==(n,)
    assert np.isfinite(a_paths).all() and np.isfinite(b_end).all() and valid.any()
    start=time.perf_counter();out=a_paths.copy();perm=np.tile(np.arange(k,dtype=np.int64)[:,None],(1,n))
    inverse_failures=0;hungarian_seconds=0.;cost_sum=0.;changed_agents=0
    for i in np.flatnonzero(valid):
        a=a_paths[:,i,-1,:].astype(np.float64)
        b=b_end[:,i,:].astype(np.float64)
        cost=np.linalg.norm(b[:,None,:]-a[None,:,:],axis=-1)
        tick=time.perf_counter();rows,cols=linear_sum_assignment(cost)
        hungarian_seconds+=time.perf_counter()-tick
        assert np.array_equal(rows,np.arange(k))
        perm[rows,i]=cols;p=perm[:,i]
        assert np.array_equal(np.sort(p),np.arange(k))
        out[:,i,:,:]=a_paths[p,i,:,:]
        inverse_failures+=int(out[np.argsort(p),i].tobytes()!=a_paths[:,i].tobytes())
        changed_agents+=int(not np.array_equal(p,np.arange(k)))
        cost_sum+=float(cost[rows,cols].sum())
    assert inverse_failures==0
    assert out[:,~valid].tobytes()==a_paths[:,~valid].tobytes()
    return out,perm,dict(inverse_restore_failures=inverse_failures,changed_agents=changed_agents,
        changed_window=not np.array_equal(out,a_paths),matching_seconds=time.perf_counter()-start,
        hungarian_seconds=hungarian_seconds,matching_cost_sum_m=cost_sum)


def joint_criterion(a,b,r):
    """Unrounded native scores; preserve negative and >100% retention rates."""
    positive=b<a
    threshold=b+.10*(a-b) if positive else a
    return dict(A=a,B=b,R=r,B_positive_gain=positive,
        retention_rate=(a-r)/(a-b) if positive else None,
        retention_percent=100*(a-r)/(a-b) if positive else None,
        limit_m=threshold,R_minus_limit_m=r-threshold,R_minus_limit_cm=100*(r-threshold),
        passed=bool(r<=threshold))
