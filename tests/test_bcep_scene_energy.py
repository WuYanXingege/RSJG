import copy,itertools
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from bcep_scene_energy.energy import SceneEnergy,ConditionalReplacementScorer,EnergyState
from bcep_scene_energy.coupling import SceneEnergyCoupling
from bcep_scene_energy.matching import solve,reference_gap,integer_scores,assignment_certificate
from bcep_scene_energy.model import build
from bcep_v2.joint import pair_loss
from bcep_v3.fixtures import joint_config
from ebjd.representation import cv_baseline


def fixture(n=3,b=1,seed=71):
    gen=torch.Generator().manual_seed(seed)
    obs=torch.randn(b,n,8,2,generator=gen)*.2+torch.arange(n)[None,:,None,None]*.4
    baseline,velocity=cv_baseline(obs)
    ctx=SimpleNamespace(observed=obs,baseline=baseline,mean_velocity=velocity,
        valid=torch.ones(b,n,dtype=torch.bool),social_h=torch.randn(b,n,192,generator=gen))
    goals=obs[:,None,:,-1]+torch.randn(b,20,n,2,generator=gen)
    return ctx,goals


def coupling(policy='proposal_st'):
    torch.manual_seed(3201);m=SceneEnergyCoupling();m.enable_energy(rejection_gradient=policy);return m


def state64(state):
    return EnergyState(state.z.double(),state.tables.double(),state.edges,state.weights.double(),state.incidence)


@pytest.mark.parametrize('k',[2,3,4,5,6,7])
def test_integer_cycle_gap_matches_complete_enumeration(k):
    rng=np.random.default_rng(70+k)
    for s in [rng.normal(size=(k,k)).astype('f'),rng.integers(-2,3,(k,k)).astype('f'),np.zeros((k,k),'f')]:
        p,c=solve(s);integer,q=integer_scores(s)
        totals=sorted((sum(int(integer[i,p[i]]) for i in range(k)) for p in itertools.permutations(range(k))),reverse=True)
        gap=totals[0]-totals[1]
        assert int(c['gap_integer'])==gap and c['unique']==(gap>0)
        assert c['gap']==reference_gap(s,p)


def test_unique_check_random_ties_near_ties_and_wide_lattice():
    rng=np.random.default_rng(2035)
    for _ in range(100):
        s=rng.normal(size=(20,20)).astype('f');p,c=solve(s)
        assert c['gap']==reference_gap(s,p)
    for value in [np.float32(0),np.nextafter(np.float32(0),np.float32(1)),np.float32(1e-8),np.float32(1e-6)]:
        s=np.eye(20,dtype='f')*value;p,c=solve(s)
        assert c['unique']==(value>0) and c['gap']==2*float(value)
    s=np.array([[1e30,0,0],[0,1e-20,0],[0,0,1e-20]],dtype='f')
    p,c=solve(s);ints,q=integer_scores(s)
    totals=sorted((sum(int(ints[i,x[i]]) for i in range(3)) for x in itertools.permutations(range(3))),reverse=True)
    assert int(c['gap_integer'])==totals[0]-totals[1] and c['arithmetic']=='python_integer'
    bad=np.array([1,0]);cert=assignment_certificate(np.eye(2,dtype='f'),bad)
    assert not cert['unique'] and not cert['native_solution_exactly_optimal']


@pytest.mark.parametrize('n',[1,2,9,20,42,57])
def test_neural_energy_fast_formula_symmetry_bijection_and_monotone(n):
    m=coupling();ctx,g=fixture(n)
    with torch.no_grad():
        h,pi,s,detail=m.assign(ctx,g,torch.tensor(1.),straight_through=False,return_details=True)
        st=state64(detail['scenes'][0]['state']);p=torch.nn.functional.one_hot(pi[0].T,20).double()
        torch.testing.assert_close(SceneEnergy.forward(st,p),SceneEnergy.complete_pair_formula(st,p),rtol=2e-12,atol=2e-12)
        # Soft second moments use candidate-pair expectations, not squared barycentres.
        soft=torch.softmax(torch.arange(n*400,dtype=torch.float64).reshape(n,20,20).sin(),-1)
        torch.testing.assert_close(SceneEnergy.forward(st,soft),SceneEnergy.complete_pair_formula(st,soft),rtol=2e-12,atol=2e-12)
        if n>1:
            rows=detail['scenes'][0]['trace']
            assert all(r['energy_after_fp64']<=r['energy_before_fp64'] for r in rows)
            assert all(not r['accepted'] or (-r['energy_delta_fp64']>r['arithmetic_error_bound'] and r['unique']) for r in rows)
        assert torch.equal(pi.sort(1).values,torch.arange(20)[None,:,None].expand_as(pi))
        assert torch.equal(h,g.gather(1,pi[...,None].expand_as(g)))
        truth=torch.randn(1,n,2)
        assert torch.equal((g-truth[:,None]).norm(dim=-1).min(1).values,(h-truth[:,None]).norm(dim=-1).min(1).values)


def test_exact_conditional_replacement_neural_tables_and_shared_rho():
    m=coupling();ctx,g=fixture(3);s0,u=m.forward_features(ctx,g,torch.tensor(1.))
    states,_=m.energy.prepare(u,ctx,g,torch.tensor(1.));state=state64(states[0]);p=torch.eye(20,dtype=torch.float64).expand(3,20,20).clone()
    base=SceneEnergy.forward(state,p);scorer=ConditionalReplacementScorer()
    for i in range(3):
        score=scorer(state,p,i)
        for a in range(20):
            q=p.clone();q[i]=torch.nn.functional.one_hot(torch.full((20,),a),20).double()
            torch.testing.assert_close(score[:,a],-(SceneEnergy.forward(state,q)-base),rtol=2e-11,atol=2e-11)
    from bcep_scene_energy.energy import motion_summary
    from bcep_v3.geometry import reference_paths
    args=(u,motion_summary(ctx.observed,torch.tensor(1.)),reference_paths(ctx,g),ctx.observed,torch.tensor(1.))
    a=m.energy.rho.chunk(*args,torch.tensor([0]),torch.tensor([0]),torch.tensor([1]))
    b=m.energy.rho.chunk(*args,torch.tensor([0]),torch.tensor([1]),torch.tensor([0]))
    torch.testing.assert_close(a,b.transpose(1,2),atol=1e-7,rtol=1e-7)
    assert len([name for name,_ in m.named_modules() if name=='energy.rho.net'])==1


def test_fast_and_complete_neural_energy_gradients_match_on_stochastic_manifold():
    m=coupling().double();ctx,g=fixture(3)
    u=torch.randn(1,3,20,192,dtype=torch.float64,requires_grad=True)
    state=m.energy.prepare(u,ctx,g,torch.tensor(1.))[0][0]
    logits=torch.randn(3,20,20,dtype=torch.float64,requires_grad=True);p=logits.softmax(-1)
    params=list(m.energy.parameters())+[u,logits]
    a=SceneEnergy.forward(state,p).sum();b=SceneEnergy.complete_pair_formula(state,p).sum()
    ga=torch.autograd.grad(a,params,retain_graph=True);gb=torch.autograd.grad(b,params)
    for x,y in zip(ga,gb):torch.testing.assert_close(x,y,rtol=2e-10,atol=2e-11)


def test_candidate_and_world_relabel_carried_incumbent_and_order_dependence():
    m=coupling();ctx,g=fixture(9);identity=torch.arange(20)[None,:,None].expand(1,20,9)
    with torch.no_grad():
        a=m.assign(ctx,g,torch.tensor(1.),straight_through=False,incumbent=identity,return_details=True)
        world=torch.arange(19,-1,-1)
        b=m.assign(ctx,g,torch.tensor(1.),straight_through=False,incumbent=identity[:,world])
        assert torch.equal(a[1][:,world],b[1])
        order=torch.stack([torch.randperm(20) for _ in range(9)])
        renamed=g.transpose(1,2).gather(2,order[None,:,:,None].expand(1,9,20,2)).transpose(1,2)
        ip=order.argsort(1).T[None]
        c=m.assign(ctx,renamed,torch.tensor(1.),straight_through=False,incumbent=ip)
        back=order.gather(1,c[1][0].T).T[None]
        assert torch.equal(a[1],back)
        ix=torch.arange(8,-1,-1)
        cc=SimpleNamespace(**{k:v[:,ix] for k,v in vars(ctx).items()})
        d=m.assign(cc,g[:,:,ix],torch.tensor(1.),straight_through=False,incumbent=identity[:,:,ix],visit_orders=[list(range(8,-1,-1))])
        assert torch.equal(a[1][:,:,ix],d[1])
        e=m.assign(cc,g[:,:,ix],torch.tensor(1.),straight_through=False,incumbent=identity[:,:,ix])
        # Changing physical visitation order is a DIFFERENT coordinate algorithm.
        assert not torch.equal(a[1][:,:,ix],e[1])


def test_rejection_proxy_prevents_fixed_point_starvation_without_moving_goals():
    ctx,g=fixture(2);m=coupling()
    with torch.no_grad():fixed=m.assign(ctx,g,torch.tensor(1.),straight_through=False)[1]
    g=g.requires_grad_();h,pi,score,d=m.assign(ctx,g,torch.tensor(1.),straight_through=True,incumbent=fixed,return_details=True)
    assert not any(r['accepted'] for r in d['scenes'][0]['trace'])
    assert torch.equal(pi,fixed)
    loss,_=pair_loss(h,ctx.baseline[:,:,-1]+.7,ctx.valid)
    phi=list(m.energy.phi.parameters());rho=list(m.energy.rho.parameters())
    gg=torch.autograd.grad(loss,phi+rho+[g],allow_unused=True)
    assert all(x is not None and x.isfinite().all() for x in gg[:-1]) and gg[-1] is None
    assert sum(x.abs().sum() for x in gg[:len(phi)])>0 and sum(x.abs().sum() for x in gg[len(phi):-1])>0
    m.coordinator.rejection_gradient='accepted_only'
    h,pi,score=m.assign(ctx,g,torch.tensor(1.),straight_through=True,incumbent=fixed)
    loss,_=pair_loss(h,ctx.baseline[:,:,-1]+.7,ctx.valid)
    dead=torch.autograd.grad(loss,phi+rho,allow_unused=True)
    assert all(x is None or not x.any() for x in dead)


def test_strict_tie_abstains_and_padding_keeps_identity():
    m=coupling();ctx,g=fixture(3,2);ctx.valid[1,1:]=False
    # Controlled SYNTHETIC degeneracy, never a loaded historical checkpoint.
    with torch.no_grad():
        m.energy.phi.net[-1].weight.zero_();m.energy.phi.net[-1].bias.zero_()
        m.energy.rho.net[-1].weight.zero_();m.energy.rho.net[-1].bias.zero_()
        h,pi,score,d=m.assign(ctx,g,torch.tensor(1.),straight_through=False,return_details=True)
    assert not any(r['accepted'] for s in d['scenes'] for r in s['trace'])
    assert all(r['reason']=='strict_tie' for r in d['scenes'][0]['trace'])
    assert torch.equal(pi,d['initial_pi']) and torch.equal(pi[1],torch.arange(20)[:,None].expand(20,3))


def test_build_uses_v2_random_parameters_and_no_historical_initialization():
    from bcep_v2.joint import build_joint
    cfg=dict(joint_config(),allow_real_training=False)
    torch.manual_seed(3201);v2=build_joint(cfg)
    torch.manual_seed(3201);energy,origin=build(cfg)
    for k,v in v2.state_dict().items():
        if k not in origin['removed_old_feedback_keys']:assert torch.equal(v,energy.state_dict()[k])
    assert not hasattr(energy.coupling,'hcra') and energy.coupling.csee is None
    assert len(energy.bridge.denoiser.blocks)==6
    with pytest.raises(TypeError):build(cfg,v3_state={})
    with pytest.raises(PermissionError):build(dict(cfg,allow_real_training=True))
    from bcep_scene_energy.interface import train
    with pytest.raises(PermissionError):train()
