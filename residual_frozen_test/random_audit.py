"""Fingerprints of actual keyed inputs plus repeat-generation (no extra DDIM)."""
import hashlib
import torch
from joint_flow_dependency.model import undirected_edges
from joint_flow_dependency.sampling import base_noise,bridge_noise


def fingerprint(t):
    a=t.detach().cpu().contiguous().numpy()
    return dict(shape=list(a.shape),dtype=str(a.dtype),sha256=hashlib.sha256(a.tobytes()).hexdigest())


def regenerate(batch,raw_ids,seed):
    edges=undirected_edges(batch.observed[0])
    epsilon,xi=base_noise(batch.scene_ids[0],batch.agent_ids[0],edges,seed=seed)
    noise=bridge_noise(batch.scene_ids[0],batch.agent_ids[0],raw_ids.cpu().tolist(),seed=seed)
    return dict(personal=epsilon,shared_edge=xi,bridge=noise)


def audit_actual(batch,endpoints,seed):
    actual=dict(personal=endpoints['epsilon'],shared_edge=endpoints['xi'],bridge=endpoints['noise'])
    regenerated=regenerate(batch,endpoints['raw_indices'],seed)
    for k,t in actual.items():assert torch.equal(t.cpu(),regenerated[k]), k+' seed failed to reach actual inference'
    return dict(seed=seed,actual={k:fingerprint(t) for k,t in actual.items()},
        regenerated_identical=True,raw_candidates=fingerprint(endpoints['raw']),
        raw_ids=fingerprint(endpoints['raw_indices']),additional_DDIM_calls=0)


def validate_resume(batch,raw_ids,seed,proof):
    assert proof['seed']==seed and proof['regenerated_identical']
    assert {k:fingerprint(t) for k,t in regenerate(batch,raw_ids,seed).items()}==proof['actual']


def validate_cross_seed(rows):
    from .common import NEW_SEEDS
    grouped={}
    for row in rows:
        if row.get('random_inputs'):
            grouped.setdefault((row['fold'],row['window']),{})[row['seed']]=row['random_inputs']
    results=[]
    for (fold,window),seeds in sorted(grouped.items()):
        assert set(seeds)==set(NEW_SEEDS)
        for stream in ('personal','shared_edge','bridge'):
            values=[seeds[s]['actual'][stream] for s in NEW_SEEDS]
            nonempty=all(all(n>0 for n in v['shape']) for v in values)
            distinct=len({v['sha256'] for v in values})
            assert not nonempty or distinct==4,'seed failed to alter a nonempty stream'
            results.append(dict(fold=fold,window=window,stream=stream,nonempty=nonempty,
                                distinct_fingerprints=distinct,seeds=list(NEW_SEEDS),passed=True))
    assert grouped and all(any(r['fold']==f and r['stream']=='shared_edge' and r['nonempty'] for r in results)
                           for f in ('zara1','zara2','hotel'))
    return dict(passed=True,windows=len(grouped),checks=results)
