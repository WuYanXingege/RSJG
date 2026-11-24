"""Fixed-source FP32 equivalence tests; these are not research measurements."""
import copy
from pathlib import Path
import subprocess
import types
import numpy as np
import pytest
import torch

from ebjd.denoiser import GeometryConditionedJointDenoiser
from ebjd.encoders import EBJDContext
from ebjd.representation import EndpointBridgeDecomposition, cv_baseline
from ebjd.sampling import differentiable_sample, _ddim_schedule
from ebjd.objectives import relative_motion_loss, relative_motion_targets
from ebjd.unified import hard_assignment_indices
from bcep_v2.stages import _add_log, _finish_log, _FrozenValidationContexts

BASE = '4969c12adb5675aa654b2e748dec8da0f6e32adf'
ROOT = Path(__file__).resolve().parents[2]
# Locked before executing tests; no test failure changes this envelope.
GRAD_ATOL = 1e-6
GRAD_RTOL = 1e-5


def original(relative):
    text = subprocess.check_output(['git','show',BASE+':'+relative],cwd=ROOT,text=True)
    module = types.ModuleType('reference_'+Path(relative).stem)
    module.__package__ = relative.split('/')[0]
    exec(compile(text,BASE+':'+relative,'exec'),module.__dict__)
    return module


def fixture_model():
    torch.set_num_threads(1);torch.manual_seed(2035)
    rep = EndpointBridgeDecomposition()
    net = GeometryConditionedJointDenoiser(rep,dim=16,ff_dim=64,activation_checkpointing=False,proposal_conditioned=True)
    with torch.no_grad():
        net.proposal_embedding[-1].weight.normal_(0,.02)
        for block in net.blocks:block.modulation[-1].weight.normal_(0,.02)
    observed = torch.randn(1,3,8,2)
    baseline, velocity = cv_baseline(observed)
    c = EBJDContext(memory=torch.randn(1,3,72,16),social_h=torch.randn(1,3,16),
        observed_bias=torch.randn(1,4,3,3),baseline=baseline,mean_velocity=velocity,
        map_features=torch.randn(1,3,32,8,8),map_logits=torch.randn(1,3,12,8,8),
        map_center=observed[:,:,-1],valid=torch.tensor([[True,True,False]]),observed=observed,crop_width_m=32.)
    noise=torch.randn(1,2,3,12,2)
    goals=torch.randn(1,2,3,2)
    old=original('ebjd/denoiser.py').GeometryConditionedJointDenoiser(copy.deepcopy(rep),dim=16,ff_dim=64,
        activation_checkpointing=False,proposal_conditioned=True)
    old.load_state_dict(net.state_dict(),strict=True)
    return net,old,c,noise,goals


@pytest.mark.parametrize('checkpoint_steps',[False,True])
def test_rollout_output_gradients_and_single_adamw_update(checkpoint_steps):
    net,old,c,noise,goals=fixture_model()
    base_sample=original('ebjd/sampling.py').differentiable_sample
    new_model=types.SimpleNamespace(denoiser=net,representation=net.representation)
    old_model=types.SimpleNamespace(denoiser=old,representation=old.representation)
    # Intrinsic same-device baseline replay is bitwise exact for this fixture.
    with torch.no_grad():
        reference=base_sample(old_model,c,noise,steps=3,proposal_goals=goals,proposal_eta=0.)
        repeated=base_sample(old_model,c,noise,steps=3,proposal_goals=goals,proposal_eta=0.)
        assert all(torch.equal(a,b) for a,b in zip(reference,repeated))
    optimizers=[torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=1e-4) for m in (net,old)]
    got=differentiable_sample(new_model,c,noise,steps=3,checkpoint_steps=checkpoint_steps,proposal_goals=goals,proposal_eta=0.)
    expected=base_sample(old_model,c,noise,steps=3,checkpoint_steps=checkpoint_steps,proposal_goals=goals,proposal_eta=0.)
    assert all(torch.equal(a,b) for a,b in zip(got,expected))
    got[0].square().mean().backward();expected[0].square().mean().backward()
    for (name,p),(old_name,q) in zip(net.named_parameters(),old.named_parameters()):
        assert name==old_name and (p.grad is None)==(q.grad is None)
        if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,atol=GRAD_ATOL,rtol=GRAD_RTOL)
    for opt in optimizers:opt.step()
    for p,q in zip(net.parameters(),old.parameters()):
        torch.testing.assert_close(p,q,atol=GRAD_ATOL,rtol=GRAD_RTOL)


def test_modulation_and_proposal_call_counts_and_no_cross_rollout_graph_cache():
    net,old,c,noise,goals=fixture_model();counts=dict(modulation=0,proposal=0)
    def increment(key):
        def hook(*args):counts[key]+=1
        return hook
    handles=[b.modulation.register_forward_hook(increment('modulation')) for b in net.blocks]
    handles.append(net.proposal_embedding.register_forward_hook(increment('proposal')))
    model=types.SimpleNamespace(denoiser=net,representation=net.representation)
    for _ in range(2):
        value=differentiable_sample(model,c,noise,steps=3,proposal_goals=goals,proposal_eta=0.)[0]
        value.square().mean().backward();net.zero_grad(set_to_none=True)
    assert counts==dict(modulation=36,proposal=2)
    for h in handles:h.remove()
    assert set(net.state_dict())==set(old.state_dict())
    s1=_ddim_schedule(3,1,torch.device('cpu'),torch.float32)
    assert _ddim_schedule(3,1,torch.device('cpu'),torch.float32) is s1
    assert _ddim_schedule(4,1,torch.device('cpu'),torch.float32) is not s1


def test_hungarian_exact_with_ties_singletons_padding_and_permutations():
    torch.manual_seed(43)
    scores=torch.randn(3,5,20,20);scores[0]=0
    valid=torch.tensor([[True]*5,[True,False,False,False,False],[True,True,True,False,False]])
    reference=original('ebjd/unified.py').hard_assignment_indices(scores,valid)
    indices=hard_assignment_indices(scores,valid)
    assert torch.equal(indices,reference)
    assert torch.equal(indices.sort(1).values,torch.arange(20)[None,:,None].expand_as(indices))


def test_shared_geometry_targets_preserve_loss_and_gradients_exactly():
    torch.manual_seed(64)
    observed=torch.randn(3,5,8,2);valid=torch.tensor([[True]*5,[True,False,False,False,False],[True,True,False,False,False]])
    c=types.SimpleNamespace(observed=observed,mean_velocity=torch.randn(3,5,2),valid=valid)
    target=torch.randn(3,5,12,2);p=torch.randn(3,2,5,12,2,requires_grad=True);q=p.detach().clone().requires_grad_()
    old=original('ebjd/objectives.py').relative_motion_loss
    cached=relative_motion_targets(target,c)
    got=relative_motion_loss(p,target,c,target_state=cached)+relative_motion_loss(p*.8,target,c,target_state=cached)
    ref=old(q,target,c)+old(q*.8,target,c)
    assert torch.equal(got,ref)
    got.sum().backward();ref.sum().backward();assert torch.equal(p.grad,q.grad)


def test_deferred_log_scalars_match_original_python_double_order():
    torch.manual_seed(70);values={};reference={}
    for i in range(33):
        for key in ('loss','pair','nll'):
            x=torch.randn((),dtype=torch.float32,requires_grad=True);weight=(i+1)/33
            _add_log(values,key,x,weight)
            reference[key]=reference.get(key,0.)+float(x.detach())*weight
    assert all(not value.requires_grad for value in values.values())
    assert _finish_log(values)==reference


def test_head_validation_cache_is_bounded_and_invalidates_changed_base():
    base=torch.nn.Linear(2,3).eval().requires_grad_(False)
    cache=_FrozenValidationContexts(base,48);cache.refresh()
    a=torch.arange(8,dtype=torch.float32).requires_grad_();b=torch.arange(4,dtype=torch.float32)
    cache.put(17,(a,b));cache.put(18,(a,b))
    assert cache.info()['bytes']==48 and cache.info()['entries']==1
    assert all(not value.requires_grad for value in cache.get(17,'cpu'))
    assert cache.get(18,'cpu') is None
    cache.refresh();assert torch.equal(cache.get(17,'cpu')[0],a)
    with torch.no_grad():base.weight.add_(.01)
    cache.refresh();assert cache.get(17,'cpu') is None and cache.used==0


def test_named_disable_switches_recover_original_rollout(monkeypatch):
    import ebjd.execution as execution
    monkeypatch.setattr(execution,'DISABLED',execution.NAMES)
    net,old,c,noise,goals=fixture_model()
    new_model=types.SimpleNamespace(denoiser=net,representation=net.representation)
    old_model=types.SimpleNamespace(denoiser=old,representation=old.representation)
    with torch.no_grad():
        got=differentiable_sample(new_model,c,noise,steps=3,proposal_goals=goals,proposal_eta=0.)
        ref=original('ebjd/sampling.py').differentiable_sample(old_model,c,noise,steps=3,proposal_goals=goals,proposal_eta=0.)
    assert all(torch.equal(a,b) for a,b in zip(got,ref))
    scores=torch.randn(2,3,20,20);valid=torch.ones(2,3,dtype=torch.bool)
    assert torch.equal(hard_assignment_indices(scores,valid),original('ebjd/unified.py').hard_assignment_indices(scores,valid))
