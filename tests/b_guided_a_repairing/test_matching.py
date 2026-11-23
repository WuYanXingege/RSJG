import numpy as np
import torch
from b_guided_a_repairing.matching import repair_trajectories,joint_criterion
from ebjd.metrics import hard_metric_statistics


def test_full_trajectory_inverse_and_native_marginals():
    rng=np.random.default_rng(2035)
    a=rng.normal(size=(20,4,12,2)).astype(np.float32)
    expected=np.stack([rng.permutation(20) for _ in range(4)],1)
    b=np.stack([a[expected[:,i],i,-1] for i in range(4)],1)
    valid=np.array([True,True,True,False]);a0=a.copy();b0=b.copy()
    r,p,proof=repair_trajectories(a,b,valid)
    assert np.array_equal(p[:,:3],expected[:,:3]) and np.array_equal(p[:,3],np.arange(20))
    assert np.array_equal(a,a0) and np.array_equal(b,b0) and proof['inverse_restore_failures']==0
    truth=torch.from_numpy(rng.normal(size=(1,4,12,2)).astype(np.float32))
    x=hard_metric_statistics(torch.from_numpy(a)[None],truth,torch.from_numpy(valid)[None])
    y=hard_metric_statistics(torch.from_numpy(r)[None],truth,torch.from_numpy(valid)[None])
    assert torch.equal(x.agent_minade,y.agent_minade)
    assert torch.equal(x.agent_minfde,y.agent_minfde)


def test_singleton_all_four_native_metrics():
    rng=np.random.default_rng(8);a=rng.normal(size=(20,1,12,2)).astype(np.float32)
    b=a[::-1,:,-1].copy();valid=np.ones(1,dtype=bool)
    r,p,_=repair_trajectories(a,b,valid)
    truth=torch.from_numpy(rng.normal(size=(1,1,12,2)).astype(np.float32))
    x=hard_metric_statistics(torch.from_numpy(a)[None],truth,torch.from_numpy(valid)[None])
    y=hard_metric_statistics(torch.from_numpy(r)[None],truth,torch.from_numpy(valid)[None])
    for key in ('agent_minade','agent_minfde','scene_jade','scene_jfde'):
        assert torch.equal(getattr(x,key),getattr(y,key))


def test_equal_costs_fixed_order_and_zero_change():
    a=np.zeros((20,2,12,2),dtype=np.float32);b=np.zeros((20,2,2),dtype=np.float32)
    r,p,proof=repair_trajectories(a,b,np.ones(2,dtype=bool))
    r2,p2,_=repair_trajectories(a,b,np.ones(2,dtype=bool))
    assert np.array_equal(r,a) and np.array_equal(p,p2)
    assert not proof['changed_window']


def test_decision_exact_boundary_and_no_positive_gain():
    limit=1.+.1*(2.-1.)
    assert joint_criterion(2.,1.,limit)['passed']
    assert not joint_criterion(2.,1.,np.nextafter(limit,np.inf))['passed']
    assert joint_criterion(2.,1.,.5)['retention_rate']==1.5
    assert joint_criterion(2.,1.,3.)['retention_rate']==-1.
    assert joint_criterion(1.,2.,1.)['passed']
    assert joint_criterion(1.,2.,1.)['retention_rate'] is None
    assert not joint_criterion(1.,1.,1.00001)['passed']
