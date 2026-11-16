"""Focused pure-NumPy synthetic checks, run once inside the task's CPU ledger."""
import itertools
import random
import numpy as np
from .algorithms import one_swap_polish,kmeans_centers,mean_cover_distance
from .core import select_one,local_seed,probe_correspondence
from bcep.coverage_selector import select_goals


def run_checks():
    rng=np.random.default_rng(20261116);accepted=0
    for case in range(40):
        x=rng.normal(size=(8,2));ids=rng.choice(8,3,replace=False);before=x.copy()
        result=one_swap_polish(x,ids);options=[]
        for pos in range(3):
            for incoming in range(8):
                if incoming in ids:continue
                trial=ids.copy();trial[pos]=incoming
                # Independent full enumeration, with the same left-to-right
                # support-axis accumulation as the vectorized matrix reduction.
                # np.mean on a contiguous 1D array may use pairwise summation,
                # changing last-bit ordering of mathematically tied swaps.
                nearest=np.linalg.norm(x[:,None,:]-x[trial][None,:,:],axis=-1).min(axis=1)
                cost=sum(float(v) for v in nearest)/len(x)
                options.append((cost,pos,incoming,trial))
        best=min(options,key=lambda t:t[:3]);old=mean_cover_distance(x,x[ids]);eps=1e-12*(1+abs(old))
        expected=best[3] if best[0]<old-eps else ids
        np.testing.assert_array_equal(result.indices,expected)
        np.testing.assert_allclose(result.after,min(old,best[0]),rtol=0,atol=1e-12)
        assert result.after<=result.before+eps and len(set(result.indices))==3
        np.testing.assert_array_equal(x,before);accepted+=result.accepted
        a=kmeans_centers(x,3,case);b=kmeans_centers(x,3,case)
        np.testing.assert_array_equal(a['centers'],b['centers'])
    for m,k in [(1,1),(6,1),(6,6),(6,3)]:
        x=np.zeros((m,2));ids=np.arange(k);swap=one_swap_polish(x,ids)
        assert not swap.accepted and swap.after==0
        km=kmeans_centers(x,k,4);assert km['mean_squared_distortion']==0
        assert km['chosen_restart']==0
    duplicate=np.array([[0.,0.],[0.,0.],[1.,1.],[1.,1.]])
    assert kmeans_centers(duplicate,4,5)['exact_unique_centers']==2
    rejected=0
    invalid=[lambda:one_swap_polish([[np.nan,0]],np.array([0])),lambda:one_swap_polish([[0.,0.]],np.array([1])),
             lambda:one_swap_polish([[0.,0.],[1.,1.]],np.array([0,0])),lambda:one_swap_polish([[0.,0.]],np.array([0.])),
             lambda:kmeans_centers([[0.,0.]],0,0),lambda:kmeans_centers([[np.inf,0]],1,0),
             lambda:kmeans_centers([[0.,0.]],1,0,tolerance_m=np.nan),lambda:kmeans_centers([[0.,0.]],True,0)]
    for fn in invalid:
        try:fn()
        except (ValueError,TypeError):rejected+=1
    assert rejected==len(invalid)
    points=np.array([[-1.,0.],[1.,0.]])
    center=kmeans_centers(points,1,7)['centers'];np.testing.assert_array_equal(center,[[0.,0.]])
    assert np.linalg.norm(center[0])<np.linalg.norm(points,axis=1).min()
    # Generation projection contains only support data; truth is inaccessible.
    x=rng.normal(size=(256,2)).astype('float32');g=select_goals(x,20);seed=local_seed('flow','scene','agent',2035)
    def generate(record):return select_one(record['atoms'],record['iid'],record['greedy'],record['ids'],seed)
    class Record(dict):
        def __getitem__(self,k):
            if k=='truth':raise AssertionError('truth access during generation')
            return super().__getitem__(k)
    record=Record(atoms=x,iid=x[:20],greedy=x[g['indices']],ids=g['indices'],truth=np.zeros(2))
    global_np=np.random.get_state();global_py=random.getstate()
    a=generate(record);record['truth']=rng.normal(size=2)*1000;b=generate(record)
    for method in a['outputs']:np.testing.assert_array_equal(a['outputs'][method],b['outputs'][method])
    np.testing.assert_array_equal(global_np[1],np.random.get_state()[1]);assert global_np[2:]==np.random.get_state()[2:]
    assert global_py==random.getstate()
    valid=np.ones((2,1),bool)
    def metadata(order):
        return dict(model_kind='flow',unit='m',members=[dict(scene_id=i,agent_ids=['p']) for i in order],scene_ids=order,agent_ids=[['p'],['p']])
    pairs=probe_correspondence(metadata(['s1','s2']),valid,metadata(['s2','s1']),valid)
    assert pairs==[(('s1','p'),(0,0),(1,0)),(('s2','p'),(1,0),(0,0))]
    for field,value in [('model_kind','mdn8'),('unit','pixel')]:
        wrong=metadata(['s1','s2']);wrong[field]=value
        try:probe_correspondence(metadata(['s1','s2']),valid,wrong,valid)
        except ValueError:pass
        else:raise AssertionError('invalid probe accepted')
    return dict(status='PASS',random_exhaustive_swap_cases=40,accepted_random_swaps=accepted,invalid_cases_rejected=rejected,
                k1_km_degeneracy=True,repeated_coordinates_retained=True,kmeans_local_rng=True,centroid_counterexample=True,
                truth_replacement_outputs_exact=True,truth_access_guard=True,probe_identity_permutation=True,
                no_input_mutation=True,optimizer_updates=0,gpu_tasks=0)
