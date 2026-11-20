import copy
from types import SimpleNamespace
import pytest
import torch
from bcep_v2.joint import FeedbackCoupling,assignment,neighbors
from bcep_v3.coupling import HCRACoupling
from bcep_v3.gate import GeometryReliabilityGate
from bcep_v3.geometry import reference_paths,pair_features
from bcep_v3.model import build
from bcep_v3.fixtures import joint_config,SyntheticData,SyntheticBank
from bcep_v3.contracts import validate_batch
from ebjd.representation import cv_baseline
from ebjd.unified import gather_candidates


def context(n=3,b=1):
    torch.manual_seed(101+n)
    obs=torch.randn(b,n,8,2)*.2+torch.arange(n)[None,:,None,None]*.4
    base,velocity=cv_baseline(obs)
    return SimpleNamespace(observed=obs,baseline=base,mean_velocity=velocity,
                           valid=torch.ones(b,n,dtype=torch.bool),social_h=torch.randn(b,n,192))


def coupling():
    torch.manual_seed(70)
    old=FeedbackCoupling()
    with torch.no_grad():old.compatibility[-1].weight.normal_(std=.05)
    new=HCRACoupling();new.load_state_dict(old.state_dict(),strict=True);new.enable_v3()
    return old,new


@pytest.mark.parametrize('n',[1,2,9,57])
def test_zero_residual_old_feedback_and_bijection(n):
    old,new=coupling();c=context(n);g=torch.randn(1,20,n,2)
    a=old.assign(c,g,torch.tensor(1.),straight_through=False)
    b=new.assign(c,g,torch.tensor(1.),straight_through=False,return_details=True)
    for x,y in zip(a,b[:3]):assert torch.equal(x,y)
    s,f=old.forward_features(c,g,torch.tensor(1.));pi0,p0,_=assignment(s,c.valid,False)
    assert torch.equal(pi0,b[3]['initial_pi']) and not b[3]['score_delta'].any()
    assert torch.equal(b[1].sort(1).values,torch.arange(20)[None,:,None].expand_as(b[1]))
    assert torch.equal(g.sort(1).values,b[0].sort(1).values)
    if n>1:assert not torch.equal(a[2],s),'old nonzero feedback was lost'


def test_permutations_active_residual_padding_and_ties():
    _,m=coupling();c=context(9,2);c.valid[1,6:]=False
    g=torch.randn(2,20,9,2)
    with torch.no_grad():
        m.csee.out.weight.normal_(std=.01);m.hcra.delta[-1].weight.normal_(std=.02)
        m.world_queries.mul_(10)  # Avoid inherited near-tied untrained score optima; tested separately.
    a=m.assign(c,g,torch.tensor(1.),straight_through=False,return_details=True)
    # Independent candidate permutations for EVERY person, not a common world renaming.
    order=torch.stack([torch.stack([torch.randperm(20) for _ in range(9)]) for _ in range(2)])
    gp=g.transpose(1,2).gather(2,order[...,None].expand(-1,-1,-1,2)).transpose(1,2)
    p=m.assign(c,gp,torch.tensor(1.),straight_through=False,return_details=True)
    expected=a[2].gather(3,order[:,:,None].expand(-1,-1,20,-1))
    torch.testing.assert_close(p[2],expected,atol=1e-5,rtol=1e-5)
    torch.testing.assert_close(p[3]['features'],a[3]['features'].gather(2,order[...,None].expand(-1,-1,-1,192)),atol=2e-6,rtol=2e-6)
    ix=torch.randperm(9);cc=SimpleNamespace(**{k:(v[:,ix] if torch.is_tensor(v) else v) for k,v in vars(c).items()})
    aa=m.assign(cc,g[:,:,ix],torch.tensor(1.),straight_through=False)
    torch.testing.assert_close(aa[2],a[2][:,ix],atol=1e-5,rtol=1e-5)
    cb=SimpleNamespace(**{k:v.flip(0) for k,v in vars(c).items()})
    bb=m.assign(cb,g.flip(0),torch.tensor(1.),straight_through=False)
    torch.testing.assert_close(bb[2],a[2].flip(0),atol=1e-5,rtol=1e-5)
    assert not a[2][1,6:].any()
    tied=context(11);tied.observed.zero_();assert len(neighbors(tied.observed,tied.valid)[0])==110
    one=context(1);h=m.assign(one,g[:1,:,:1],torch.tensor(1.),straight_through=False)
    assert torch.equal(h[1].flatten(),torch.arange(20))


def test_actual_pair_geometry_then_st_mixing_and_world_independence():
    _,m=coupling();c=context(3);g=torch.randn(1,20,3,2)
    paths=reference_paths(c,g);torch.testing.assert_close(paths[:,:,:,-1],g.transpose(1,2),atol=1e-6,rtol=1e-6)
    p=torch.eye(20).expand(1,3,20,20).clone().requires_grad_()
    rel,query,legal=m.hcra.relations(c,g,p,torch.tensor(1.))
    raw=pair_features(paths,c.observed,torch.tensor([0]),torch.tensor([0]),torch.tensor([1]),torch.tensor(1.))
    encoded=m.hcra.pair_encoder(raw)[0]
    torch.testing.assert_close(rel[0,:,0,1],encoded[torch.arange(20),torch.arange(20)])
    rel.square().sum().backward();assert p.grad.isfinite().all() and p.grad.abs().sum()>0
    perm=torch.randperm(20)
    rr,qq,_=m.hcra.relations(c,g,p.detach()[:,:,perm],torch.tensor(1.))
    torch.testing.assert_close(rr,rel[:,perm]);torch.testing.assert_close(qq,query[:,:,perm])
    # E[geometry(actual pairs)] differs from geometry(E[endpoints]).
    mixed=p.detach().clone();mixed[:,:,:2]=.5*(mixed[:,:,:1]+mixed[:,:,1:2])
    r,_,_=m.hcra.relations(c,g,mixed,torch.tensor(1.))
    bary=g.clone();bary[:,:2]=g[:,:2].mean(1,keepdim=True)
    naive,_,_=m.hcra.relations(c,bary,p.detach(),torch.tensor(1.))
    assert (r-naive).abs().max()>1e-4


def test_gate_bound_symmetric_world_independent_actual_condition():
    c=context(3,2);g=torch.randn(2,4,3,2);y=torch.randn(2,4,3,12,2);t=torch.tensor([.3,.7]);base=torch.tensor([.7,.2])
    m=GeometryReliabilityGate();a,p=m(c,y,g,t,base,torch.tensor(1.))
    assert torch.equal(a,base[:,None,None].expand_as(a));assert torch.equal(p,p.transpose(-1,-2))
    with torch.no_grad():m.net[-1].weight.normal_(std=.1)
    a,p=m(c,y,g,t,base,torch.tensor(1.));assert (a>.5*base[:,None,None]).all() and (a<1.5*base[:,None,None]).all()
    aa,pp=m(c,y.flip(1),g.flip(1),t,base,torch.tensor(1.));torch.testing.assert_close(aa,a.flip(1));torch.testing.assert_close(pp,p.flip(1))
    other,_=m(c,y,g+1,t,base,torch.tensor(1.));assert not torch.equal(a,other)
    c.valid[1,2]=False;a,p=m(c,y,g,t,base,torch.tensor(1.));assert not a[1,:,2].any() and not p[1,:,2].any()


def test_strict_migration_and_rejection():
    c=joint_config();old,_=build(c,variant='v2_compatible');new,origin=build(c,v2_state=old.state_dict())
    assert all(torch.equal(v,new.state_dict()[k]) for k,v in old.state_dict().items())
    assert origin['new_tensor_count']>0
    broken=dict(old.state_dict());broken.pop(next(iter(broken)))
    with pytest.raises(RuntimeError):build(c,v2_state=broken)
    with pytest.raises(ValueError):build(dict(c,pair_objective='shared_world_bounded'))
    _,m=coupling();ctx=context(2);g=torch.randn(1,20,2,2);g[0,0,0,0]=float('nan')
    with pytest.raises(ValueError):m.assign(ctx,g,torch.tensor(1.),straight_through=False)
    with pytest.raises(ValueError):m.csee(torch.randn(1,2,20,192),ctx,torch.zeros_like(g),torch.tensor(1.),{'confidence':1})
    d=SyntheticData();x=d.batch([0],'train');g=SyntheticBank().batch([0],x)
    x.metadata['unit']='cm'
    with pytest.raises(ValueError):validate_batch(x,g)
    x.metadata['unit']='m';x.frame_ids[0][1]=x.frame_ids[0][0]
    with pytest.raises(ValueError):validate_batch(x,g)


def test_audit_entry_rejects_real_training_and_wrong_adapter():
    from bcep_v3.stage2 import callbacks
    c=joint_config();d=SyntheticData();b=SyntheticBank()
    with pytest.raises(PermissionError):callbacks(None,d,'cpu',dict(c,allow_real_training=True),b,b)
    with pytest.raises(PermissionError):callbacks(None,object(),'cpu',c,b,b)


def test_zero_and_near_tie_boundary_is_inherited_not_claimed_equivariant():
    # A truly tied Hungarian objective has no unique permutation. Both solutions
    # remain legal but their coordinate correspondence need not be invariant.
    c=context(2);score=torch.zeros(1,2,20,20);order=torch.arange(19,-1,-1)
    pi,_,_=assignment(score,c.valid,False)
    after,_,_=assignment(score[...,order],c.valid,False)
    assert torch.equal(pi.sort(1).values,after.sort(1).values)
    assert not torch.equal(order[after],pi)


def test_cached_neighbors_keep_forward_and_backward_exact():
    _,m=coupling();c=context(9);g=torch.randn(1,20,9,2)
    p=torch.eye(20).expand(1,9,20,20).clone().requires_grad_()
    edges=neighbors(c.observed,c.valid)
    a=m.hcra.relations(c,g,p,torch.tensor(1.));b=m.hcra.relations(c,g,p,torch.tensor(1.),edges)
    for aa,bb in zip(a,b):assert torch.equal(aa,bb)
    ga=torch.autograd.grad(a[0].square().sum()+a[1].square().sum(),p,retain_graph=True)[0]
    gb=torch.autograd.grad(b[0].square().sum()+b[1].square().sum(),p)[0]
    assert torch.equal(ga,gb)
    f=torch.randn(1,9,20,192)
    assert torch.equal(m.messages(f,c.observed,g,c.valid,p),m.messages(f,c.observed,g,c.valid,p,edge_indices=edges))
