"""Default-entry identity regressions. No optimizer construction or updates."""
from types import SimpleNamespace
import pytest
import torch
from test_bcep_scene_energy import fixture,coupling
from bcep_scene_energy.identity import canonical_order,CandidateIdentityAmbiguity
from ebjd.unified import gather_candidates
from bcep_v2.joint import pair_loss


def permutations(n,b=1):
    gen=torch.Generator().manual_seed(2035)
    return torch.stack([torch.randperm(20,generator=gen) for _ in range(b*n)]).reshape(b,n,20)

def relabel(g,r):return gather_candidates(g,r.transpose(1,2))

def assert_relabel(n,ids=False,duplicate=False):
    m=coupling();ctx,g=fixture(n);r=permutations(n)
    if duplicate:g[:,1]=g[:,0]
    identity=torch.arange(20).expand(1,n,20)
    with torch.no_grad():
        a=m.assign(ctx,g,torch.tensor(1.),straight_through=False,return_details=True,candidate_ids=identity if ids else None)
        b=m.assign(ctx,relabel(g,r),torch.tensor(1.),straight_through=False,return_details=True,candidate_ids=r if ids else None)
    back=r.gather(2,b[1].transpose(1,2)).transpose(1,2)
    assert torch.equal(a[1],back) and torch.equal(a[0],b[0])
    assert torch.equal(a[3]['canonical_initial_scores'],b[3]['canonical_initial_scores'])
    assert torch.equal(a[3]['canonical_initial_pi'],b[3]['canonical_initial_pi'])
    assert torch.equal(a[3]['canonical_final_pi'],b[3]['canonical_final_pi'])


@pytest.mark.parametrize('n',[2,9,20,42,57])
def test_default_entry_recomputes_identical_S0_and_final(n):assert_relabel(n)


def test_duplicate_occurrences_need_transported_identity():
    c,g=fixture(2);g[:,1]=g[:,0]
    with pytest.raises(CandidateIdentityAmbiguity):canonical_order(g,c.valid)
    assert_relabel(2,ids=True,duplicate=True)


def test_exact_FP32_order_one_ulp_signed_zero_and_inverse():
    c,g=fixture(2);g[:,0]=torch.tensor([1.,2.]);g[:,1]=g[:,0]
    g[:,1,:,0]=torch.nextafter(g[:,0,:,0],torch.full_like(g[:,0,:,0],torch.inf))
    r=permutations(2);a=canonical_order(g,c.valid);b=canonical_order(relabel(g,r),c.valid)
    assert torch.equal(a.gather(g),b.gather(relabel(g,r)))
    assert torch.equal(relabel(a.gather(g),a.input_to_canonical),g)
    g[:,0]=0.;g[:,1]=0.;g[:,1,:,0]=-0.
    with pytest.raises(CandidateIdentityAmbiguity):canonical_order(g,c.valid)
    ids=torch.arange(20).expand(1,2,20)
    a=canonical_order(g,c.valid,ids);b=canonical_order(relabel(g,r),c.valid,r)
    assert torch.equal(a.gather(g).view(torch.int32),b.gather(relabel(g,r)).view(torch.int32))


def test_augmentation_transport_and_duplicate_after_FP32_rounding():
    c,g=fixture(9);r=permutations(9);ids=torch.arange(20).expand(1,9,20)
    angle=torch.tensor(.731);rotation=torch.stack([torch.stack([angle.cos(),-angle.sin()]),torch.stack([angle.sin(),angle.cos()])])
    ag=g@rotation.T+torch.tensor([7.,-3.]);rg=relabel(g,r)@rotation.T+torch.tensor([7.,-3.])
    c.observed=c.observed@rotation.T+torch.tensor([7.,-3.])
    a=canonical_order(ag,c.valid,ids);b=canonical_order(rg,c.valid,r)
    assert torch.equal(a.gather(ag),b.gather(rg))
    # Finite-precision augmentation can coalesce distinct candidates; occurrence
    # metadata remains valid even then. Never infer an ID from rounded geometry.
    g[:,1]=g[:,0];g[:,1,:,0]=torch.nextafter(g[:,0,:,0],torch.full_like(g[:,0,:,0],torch.inf))
    rounded=g+2**24
    with pytest.raises(CandidateIdentityAmbiguity):canonical_order(rounded,c.valid)
    a=canonical_order(rounded,c.valid,ids);b=canonical_order(relabel(rounded,r),c.valid,r)
    assert torch.equal(a.gather(rounded),b.gather(relabel(rounded,r)))


def test_single_person_and_padding_preserve_native_identity():
    m=coupling();c,g=fixture(3,2);c.valid[1,1:]=False;g[1,:,1:]=float('nan')
    r=permutations(3,2)
    with torch.no_grad():
        a=m.assign(c,g,torch.tensor(1.),straight_through=False)
        b=m.assign(c,relabel(g,r),torch.tensor(1.),straight_through=False)
    identity=torch.arange(20)[:,None].expand(20,3)
    assert torch.equal(a[1][1],identity) and torch.equal(b[1][1],identity)
    assert torch.equal(a[0][1,:,0],b[0][1,r[1,0].argsort(),0])


def test_default_relabel_gradients_equal_and_goals_frozen():
    m=coupling();c,g=fixture(9);g.requires_grad_();r=permutations(9)
    params=list(m.energy.parameters());grads=[]
    for goals in [g,relabel(g,r)]:
        h,pi,s=m.assign(c,goals,torch.tensor(1.),straight_through=True)
        loss,_=pair_loss(h,c.baseline[:,:,-1]+.7,c.valid)
        gg=torch.autograd.grad(loss,params+[g],allow_unused=True)
        assert gg[-1] is None and all(x is not None and torch.isfinite(x).all() for x in gg[:-1])
        assert sum(x.square().sum() for x in gg[:4])>0 and sum(x.square().sum() for x in gg[4:-1])>0
        grads.append(gg[:-1])
    assert all(torch.equal(a,b) for a,b in zip(*grads))


def test_invalid_occurrence_ids_fail_closed():
    c,g=fixture(2)
    for ids in [torch.zeros(1,2,20,dtype=torch.int64),torch.arange(20.).expand(1,2,20)]:
        with pytest.raises(ValueError):canonical_order(g,c.valid,ids)


def test_strict_S0_ties_stay_identical_without_score_perturbation():
    m=coupling();c,g=fixture(9);r=permutations(9)
    # Fresh synthetic fixture only. Make the INITIAL scorer exactly indifferent.
    with torch.no_grad():
        m.score[-1].weight.zero_();m.score[-1].bias.zero_()
        a=m.assign(c,g,torch.tensor(1.),straight_through=False,return_details=True)
        b=m.assign(c,relabel(g,r),torch.tensor(1.),straight_through=False,return_details=True)
    assert not a[3]['canonical_initial_scores'].any() and not b[3]['canonical_initial_scores'].any()
    assert torch.equal(a[3]['canonical_initial_pi'],b[3]['canonical_initial_pi'])
    assert torch.equal(a[1],r.gather(2,b[1].transpose(1,2)).transpose(1,2))


def test_optional_CSEE_keeps_same_candidate_identity_contract():
    torch.manual_seed(3201)
    from bcep_scene_energy.coupling import SceneEnergyCoupling
    m=SceneEnergyCoupling();m.enable_energy(csee=True);c,g=fixture(9);r=permutations(9)
    with torch.no_grad():
        a=m.assign(c,g,torch.tensor(1.),straight_through=False)
        b=m.assign(c,relabel(g,r),torch.tensor(1.),straight_through=False)
    assert torch.equal(a[0],b[0]) and torch.equal(a[1],r.gather(2,b[1].transpose(1,2)).transpose(1,2))
