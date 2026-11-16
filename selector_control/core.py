"""Truth-free selection and explicit identity-keyed risk probing."""
import hashlib
import json
import time
import numpy as np
from bcep.coverage_selector import select_goals
from .algorithms import kmeans_centers, one_swap_polish, pairwise_l2

METHODS = ('IID20', 'Greedy20', 'KMeans20', 'OneSwap20')


def local_seed(model_kind, scene_id, agent_id, inference_seed):
    key = json.dumps([str(model_kind),str(scene_id),str(agent_id),int(inference_seed),'kmeans20_v1'],separators=(',',':')).encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8],'little')


def select_one(atoms, historical_iid, historical_greedy, historical_indices, seed):
    """No truth/label inputs. Historical output is authoritative for old methods."""
    x=np.asarray(atoms)
    if x.shape!=(256,2):raise ValueError('fixed M256 2D support required')
    np.testing.assert_array_equal(historical_iid,x[:20])
    np.testing.assert_array_equal(historical_greedy,x[historical_indices])
    start=time.perf_counter();iid=x[:20].copy();iid_s=time.perf_counter()-start
    start=time.perf_counter();reference=select_goals(x,20);greedy_s=time.perf_counter()-start
    np.testing.assert_array_equal(reference['indices'],historical_indices)
    np.testing.assert_array_equal(reference['goals'],historical_greedy)
    start=time.perf_counter();km=kmeans_centers(x,20,seed,n_init=5,max_iter=100,tolerance_m=1e-6);km_s=time.perf_counter()-start
    start=time.perf_counter();swap=one_swap_polish(x,historical_indices,tolerance=1e-12);swap_s=time.perf_counter()-start
    assert swap.after<=swap.before+1e-12*(1+abs(swap.before))
    return dict(outputs={'IID20':iid.astype('float64'),'Greedy20':historical_greedy.astype('float64'),
                         'KMeans20':km['centers'],'OneSwap20':x[swap.indices].astype('float64')},
                swap_indices=swap.indices,swap_accepted=swap.accepted,swap_before=swap.before,swap_after=swap.after,
                removed_position=swap.removed_position,added_index=swap.added_index,
                kmeans_restart=km['chosen_restart'],kmeans_iterations=km['iterations'],kmeans_seed=seed,
                seconds={'IID20':iid_s,'Greedy20':greedy_s,'KMeans20':km_s,'OneSwap20':greedy_s+swap_s},
                swap_additional_seconds=swap_s)


def identity_index(metadata,valid):
    if metadata['unit']!='m':raise ValueError('probe requires metre coordinates')
    result={}
    for row,member in enumerate(metadata['members']):
        ids=member['agent_ids'];active=np.flatnonzero(valid[row]).tolist()
        if active!=list(range(len(ids))):raise ValueError('valid/membership mismatch')
        if metadata['scene_ids'][row]!=member['scene_id'] or metadata['agent_ids'][row]!=ids:
            raise ValueError('metadata identity columns disagree')
        for agent,agent_id in enumerate(ids):
            key=(str(member['scene_id']),str(agent_id))
            if key in result:raise ValueError('duplicate occurrence identity')
            result[key]=(row,agent)
    return result


def probe_correspondence(source_meta,source_valid,target_meta,target_valid):
    if source_meta['model_kind']!=target_meta['model_kind'] or source_meta['unit']!=target_meta['unit']:
        raise ValueError('cross-density or cross-unit probe forbidden')
    a=identity_index(source_meta,source_valid);b=identity_index(target_meta,target_valid)
    if set(a)!=set(b):raise ValueError('probe membership mismatch')
    return [(key,a[key],b[key]) for key in a]


def risks(atoms,selection):
    distance=pairwise_l2(atoms,selection).min(axis=1)
    return float(distance.mean()),float(np.square(distance).mean())
