import itertools, math, random
import numpy as np
import pytest
import torch
from bcep.provider import BCEP
from bcep.conditional_flow import ConditionalFlow
from bcep.motion_frame import frame,to_local,to_world
from bcep.coverage_selector import select_goals,_self_test

torch.set_num_threads(2)

def test_flow_inverse_and_jacobian():
 torch.manual_seed(1);m=ConditionalFlow(dim=3).double()
 for net in m.nets:
  with torch.no_grad():net[-1].weight.normal_(0,.03);net[-1].bias.normal_(0,.1)
 c=torch.randn(4,3,dtype=torch.float64);x=torch.tensor([[.3,-.8],[6.,-7.],[-4.99,4.99],[0.,0.]],dtype=torch.float64,requires_grad=True)
 y,ld=m.transform(x,c,4.8);z,il=m.transform(y,c,4.8,inverse=True)
 torch.testing.assert_close(z,x,atol=1e-9,rtol=1e-9);torch.testing.assert_close(ld+il,torch.zeros_like(ld),atol=1e-9,rtol=1e-9)
 jac=torch.autograd.functional.jacobian(lambda v:m.transform(v[None],c[:1],4.8)[0][0],x[0])
 torch.testing.assert_close(torch.linalg.slogdet(jac)[1],ld[0],atol=1e-9,rtol=1e-9)

def test_motion_world_jacobian():
 o=torch.randn(3,8,2,dtype=torch.float64);o[0]=0;s=torch.tensor([.4,2.],dtype=torch.float64)
 b,r,s=frame(o,12,.4,s);x=torch.randn(3,2,dtype=torch.float64)
 torch.testing.assert_close(to_local(to_world(x,b,r,s),b,r,s),x)
 assert torch.equal(r[0],torch.eye(2,dtype=r.dtype))
 model=BCEP().double();model.scales[-1].copy_(s);c=torch.zeros(3,128,dtype=torch.float64)
 world=to_world(x,b,r,s);torch.testing.assert_close(model.log_prob_world(world,o,c,12),model.flow.log_prob(x,c,4.8)-s.log().sum())

def test_selector_enumeration_submodularity_and_degeneracy():
 assert _self_test()['status']=='PASS'
 rng=np.random.default_rng(9);x=rng.normal(size=(7,2));d=np.linalg.norm(x[:,None]-x[None],axis=-1);D=d.max()
 def f(ids):return 0. if not ids else D-d[:,ids].min(1).mean()
 for small in [[],[0],[0,1]]:
  big=list(set(small+[2,3]));element=6
  assert f(small+[element])-f(small)>=f(big+[element])-f(big)-1e-12
 for bad in [np.array([[float('nan'),0.]]),np.empty((0,2))]:
  with pytest.raises(ValueError):select_goals(bad,1)
 with pytest.raises(ValueError):select_goals(x,2,np.zeros(7))
 assert np.array_equal(select_goals(np.zeros((5,2)),3)['indices'],[0,1,2])

def test_rng_future_padding_and_agent_order():
 torch.manual_seed(4);m=BCEP().eval();o=torch.randn(3,8,2);valid=torch.tensor([True,True,False]);maps=torch.randn(3,6,32,48);ids=['1','2','PAD']
 state=torch.get_rng_state().clone();npstate=np.random.get_state();pystate=random.getstate()
 g,meta=m.predict(o,maps,valid,'s',ids,seed=2035)
 assert torch.equal(state,torch.get_rng_state());assert np.array_equal(npstate[1],np.random.get_state()[1]);assert pystate==random.getstate()
 g2,_=m.predict(o,maps,valid,'s',ids,seed=2035);assert torch.equal(g,g2)
 other,_=m.predict(o,maps,valid,'s',ids,seed=2036);assert not torch.equal(g,other)
 o[2]=float('nan');maps[2]=float('nan');g3,_=m.predict(o,maps,valid,'s',ids,seed=2035);assert torch.equal(g,g3)
 perm=[1,0,2];gp,_=m.predict(o[perm],maps[perm],valid[perm],'s',[ids[i] for i in perm],seed=2035)
 torch.testing.assert_close(gp,g[:,perm],atol=1e-5,rtol=1e-5)
 assert torch.equal(g[:,2],torch.zeros_like(g[:,2]));assert meta['support_mass']=='uniform'
 # No future argument exists in inference; loss targets cannot modify eval model.
 m.loss(o[:2],maps[:2],valid[:2],torch.randn(2,12,2));g4,_=m.predict(o,maps,valid,'s',ids,seed=2035);assert torch.equal(g,g4)
 # A different scene call cannot advance the local support stream for s.
 m.predict(o[:1],None,valid[:1],'other',['1']);assert torch.equal(g,m.predict(o,maps,valid,'s',ids,seed=2035)[0])

def test_synthetic_optimizer_and_strict_resume(tmp_path):
 torch.manual_seed(3);m=BCEP();o=torch.randn(2,8,2);v=torch.ones(2,dtype=torch.bool);f=o[:,-1,None]+torch.randn(2,12,2);maps=torch.rand(2,6,32,32)
 opt=torch.optim.Adam(m.parameters(),lr=1e-4);before=m.flow.nets[0][-1].weight.clone()
 for _ in range(3):
  opt.zero_grad();loss=m.loss(o,maps,v,f);assert torch.isfinite(loss);loss.backward();assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None);opt.step()
 assert not torch.equal(before,m.flow.nets[0][-1].weight)
 path=tmp_path/'synthetic.pt';torch.save({'model':m.state_dict(),'optimizer':opt.state_dict()},path);c=torch.load(path,weights_only=False)
 n=BCEP();n.load_state_dict(c['model'],strict=True);opt2=torch.optim.Adam(n.parameters());opt2.load_state_dict(c['optimizer'])
 torch.testing.assert_close(n.loss(o,maps,v,f),m.loss(o,maps,v,f),rtol=0,atol=0)

def test_short_horizon_padding_loss_and_no_teacher_imports():
 import ast
 from pathlib import Path
 for p in (Path(__file__).resolve().parents[1]/'bcep').rglob('*.py'):
  tree=ast.parse(p.read_text())
  for n in ast.walk(tree):
   if isinstance(n,(ast.Import,ast.ImportFrom)):
    module=getattr(n,'module','') or ''
    assert not any(x in module.lower() for x in ['gdts','goal_provider','spatial_density_goal','learned_goal'])
 m=BCEP(pred_length=1);o=torch.zeros(2,5,2);o[1]=float('nan');f=torch.zeros(2,1,2);f[1]=float('nan');valid=torch.tensor([True,False])
 loss=m.loss(o,None,valid,f);assert torch.isfinite(loss);loss.backward()
 assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)

def test_candidate_gather_multiset():
 from ebjd.unified import gather_candidates
 m=BCEP().eval();o=torch.randn(2,8,2);v=torch.ones(2,dtype=torch.bool)
 g,_=m.predict(o,None,v,'scene',['a','b'])
 indices=torch.stack([torch.arange(19,-1,-1),torch.roll(torch.arange(20),3)],1)[None]
 h=gather_candidates(g[None],indices)
 for i in range(2):assert torch.equal(h[0,:,i],g[indices[0,:,i],i])
 assert torch.equal(g.sort(0).values,h[0].sort(0).values)

def test_endpoint_and_iid_energy_definitions():
 from bcep.evaluation import endpoint_metrics,energy_score_iid
 pred=np.array([[[[0.,0.]],[[1.,0.]]]])
 truth=np.zeros((1,1,2));valid=np.ones((1,1),bool)
 assert endpoint_metrics(pred,truth,valid)['minFDE']==0
 assert energy_score_iid(pred[0],truth[0])[0]==0
