"""Exact deterministic best whole-scene one-swap search under the initial J cap."""
import time
import numpy as np


def distances(raw):
    x=np.asarray(raw,dtype=np.float64)
    assert x.ndim==3 and x.shape[-1]==2 and np.isfinite(x).all()
    return np.linalg.norm(x[:,None]-x[None,:],axis=-1)


def objectives(distance,chosen):
    # Fixed summation order for direct and batched candidate evaluations.
    joint=distance.mean(axis=2)
    return float(joint[:,chosen].min(axis=1).mean()),float(distance[:,chosen].min(axis=1).mean())


def select_swaps(raw,initial):
    start=time.perf_counter();d=distances(raw);joint=d.mean(axis=2)
    selected=np.asarray(initial,dtype=np.int64).copy();m,n=d.shape[0],d.shape[2];k=len(selected)
    assert k>=2 and k<=m and len(set(selected))==k
    J0,P0=objectives(d,selected);J,P=J0,P0;history=[];evaluated=0;feasible=0
    while True:
        incoming=np.array(sorted(set(range(m))-set(selected)),dtype=np.int64)
        if not len(incoming):break
        jd=joint[:,selected];pd=d[:,selected,:]
        jarg=jd.argmin(axis=1);parg=pd.argmin(axis=1)
        jtwo=np.partition(jd,1,axis=1)[:,:2]
        ptwo=np.partition(pd,1,axis=1)[:,:2,:]
        best=None
        for slot,outgoing in enumerate(selected):
            jexcept=np.where(jarg==slot,jtwo[:,1],jtwo[:,0])
            jc=np.ascontiguousarray(np.minimum(jexcept[:,None],joint[:,incoming]).T).mean(axis=1)
            eligible=incoming[jc<=J0];evaluated+=len(incoming);feasible+=len(eligible)
            if not len(eligible):continue
            pexcept=np.where(parg==slot,ptwo[:,1,:],ptwo[:,0,:])
            costs=np.minimum(pexcept[:,None,:],d[:,eligible,:])
            pc=np.ascontiguousarray(costs.transpose(1,0,2)).reshape(len(eligible),m*n).mean(axis=1)
            valid=np.flatnonzero(P-pc>1e-12)
            for idx in valid:
                key=(float(pc[idx]),int(outgoing),int(eligible[idx]))
                if best is None or key<best[0]:best=(key,slot)
        if best is None:break
        key,slot=best;trial=selected.copy();trial[slot]=key[2]
        newJ,newP=objectives(d,trial)
        # Direct recomputation after each acceptance; no tolerance relaxes J0.
        assert newJ<=J0 and P-newP>1e-12
        assert newP==key[0], 'batched P reduction must match direct P exactly'
        history.append(dict(out=key[1],incoming=key[2],slot=int(slot),J_before=J,P_before=P,J_after=newJ,P_after=newP))
        selected,J,P=trial,newJ,newP
    assert len(set(selected))==k and J<=J0
    return selected,dict(J0=J0,P0=P0,J=J,P=P,swaps=len(history),history=history,
        constraint_violations=sum(h['J_after']>J0 for h in history),evaluated_exchanges=evaluated,
        feasible_exchanges=feasible,seconds=time.perf_counter()-start)
