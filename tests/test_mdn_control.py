import importlib.util
import inspect
import random
from pathlib import Path
import numpy as np
import torch
from torch.distributions import Categorical, MultivariateNormal, MixtureSameFamily
from bcep.provider import BCEP
from bcep.motion_frame import frame,to_local
from bcep_control.mdn import MixtureHead,MDNProvider
from bcep_control.pilot_runner import run_pilot
from bcep_control.support_gap import _self_test,support_gap_diagnostics
from bcep_control.contracts import state_digest,validate_checkpoint
import pytest


def fixtures():
    p=Path(__file__).with_name('test_bcep_prepilot.py')
    spec=importlib.util.spec_from_file_location('control_fixture',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def test_mixture_matches_distribution():
    torch.manual_seed(12);head=MixtureHead().double();c=torch.randn(7,128,dtype=torch.float64);r=torch.randn(7,2,dtype=torch.float64)
    logits,mean,lower=head.parameters_at(c,4.8)
    ref=MixtureSameFamily(Categorical(logits=logits),MultivariateNormal(mean,scale_tril=lower)).log_prob(r)
    torch.testing.assert_close(head.log_prob(r,c,4.8),ref,atol=1e-10,rtol=1e-10)


def test_covariance_and_asymmetric_components():
    torch.manual_seed(3101);m=MixtureHead();l,mu,L=m.parameters_at(torch.zeros(4,128),4.8)
    assert (torch.linalg.eigvalsh(L@L.transpose(-1,-2))>0).all()
    assert (L.diagonal(dim1=-2,dim2=-1)>.001).all()
    assert not torch.equal(mu[:,0],mu[:,1])


def test_identical_random_encoder_not_trained_copy():
    torch.manual_seed(3101);a=BCEP()
    torch.manual_seed(3101);b=MDNProvider()
    assert state_digest(a.encoder.state_dict())==state_digest(b.encoder.state_dict())
    assert not hasattr(b,'flow')


def test_world_jacobian_and_mask():
    torch.manual_seed(4);m=MDNProvider();m.scales.fill_(2.3)
    o=torch.randn(3,8,2);future=torch.randn(3,12,2);valid=torch.tensor([True,True,False]);o[-1]=float('nan');future[-1]=float('nan')
    context=m.encoder(o,None,valid,.4);b,r,s=frame(o[valid],12,.4,m.scales[-1])
    expected=m.mdn.log_prob(to_local(future[valid,-1],b,r,s),context[valid],4.8)-s.log().sum()
    torch.testing.assert_close(m.log_prob_world(future[valid,-1],o[valid],context[valid],12),expected)
    c=m.loss_components(o,None,valid,future);assert c['valid_agent_count']==2 and torch.isfinite(c['loss_total'])
    c['loss_total'].backward();assert m.encoder.edge_value.weight.grad is not None and m.mdn.net[-1].weight.grad.abs().sum()>0


def test_local_rng_and_repeatable_sampling():
    torch.manual_seed(4);m=MDNProvider().eval();o=torch.randn(3,8,2);v=torch.ones(3,dtype=torch.bool)
    cpu=torch.get_rng_state();py=random.getstate();npstate=np.random.get_state()
    a,_=m.sample_support(o,None,v,'scene',['a','b','c'],2035)
    b,_=m.sample_support(o,None,v,'scene',['a','b','c'],2035)
    assert torch.equal(a,b) and torch.equal(cpu,torch.get_rng_state()) and random.getstate()==py
    assert np.array_equal(npstate[1],np.random.get_state()[1])
    c,_=m.sample_support(o,None,v,'scene',['a','b','c'],2036);assert not torch.equal(a,c)


def test_shared_support_padding_and_singleton():
    torch.manual_seed(5);m=MDNProvider().eval();o=torch.randn(3,8,2);v=torch.tensor([True,False,True]);o[1]=float('nan')
    atoms,_=m.sample_support(o,None,v,'s',['a','pad','b'],2035)
    chosen,meta=m.select_support(atoms,v,['a','pad','b'])
    assert torch.equal(atoms[:,1],torch.zeros_like(atoms[:,1]))
    for j,i in enumerate([0,2]):assert torch.equal(chosen[:,i],atoms[meta['agents'][j]['indices'],i])
    one,_=m.sample_support(o[:1],None,v[:1],'s',['a'],2035);assert one.shape==(256,1,2)


def test_observed_only_sampling_signature():
    sig=inspect.signature(MDNProvider.sample_support)
    assert 'future' not in sig.parameters and 'truth' not in sig.parameters


def test_control_full_resume_and_final_validation(tmp_path):
    f=fixtures();ds,split,cfg,identity=f.runner_fixture();cfg.update(max_epochs=2,max_updates=8)
    continuous=[];resumed=[]
    a=run_pilot(ds,split,cfg,identity,tmp_path/'a',on_update=lambda i,u:continuous.append((i,u)))
    run_pilot(ds,split,cfg,identity,tmp_path/'b',stop_after_updates=3,on_update=lambda i,u:resumed.append((i,u)))
    b=run_pilot(ds,split,cfg,identity,tmp_path/'b',resume=True,on_update=lambda i,u:resumed.append((i,u)))
    for k in ['model','optimizer','scheduler','rng','epoch','completed_epoch','update','stale','selected_epoch','best']:
        f.assert_tree_equal(a[k],b[k])
    assert continuous==resumed and a['completed_epoch']==2 and a['update']==8 and a['phase']=='complete'
    assert a['schema']=='mdn-control-pilot-v1'


def test_wrong_kind_rejected():
    with pytest.raises(ValueError,match='not an MDN'):validate_checkpoint({'schema':'bcep-pilot-v2'},'unused','unused')


def test_support_gap_reference_and_nonadditive_quantiles():
    assert _self_test()['status']=='PASS'
    x=np.array([[[0.,0.],[2.,0.]],[[1.,0.],[0.,0.]]]);ids=np.array([[0,0]])
    d=support_gap_diagnostics(x,ids,np.zeros((2,2)))
    assert d['mean_chosen_error']==d['mean_pool_error']+d['mean_compression_gap']
