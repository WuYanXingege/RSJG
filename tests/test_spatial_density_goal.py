import copy
import dataclasses
import json
from pathlib import Path
import sys

import pytest
import torch

from ebjd.spatial_density_goal import SpatialGoalDensity, SpatialDensityCouplingBridge, SPATIAL_SCHEMA
from ebjd.model import EBJDModel
from ebjd.unified import LearnableJointCouplingHead, gather_candidates, joint_pair_loss
from test_unified_coupling import context
from conftest import make_batch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from train_learned_goal_bridge import objective, optimizer_for, save_checkpoint, restore_rng

torch.set_num_threads(2)


def test_continuous_nll_matches_explicit_normalized_density():
    torch.manual_seed(11)
    ctx=context();head=SpatialGoalDensity(32,grid_size=4);d=head(ctx);scale=torch.tensor(1.7)
    truth=ctx.map_center+torch.tensor([1.3,-2.4]);truth[0,1]=ctx.map_center[0,1]+torch.tensor([40.,-20.])
    expected=[]
    for b,n in ctx.valid.nonzero().tolist():
        y=truth[b,n];center=ctx.map_center[b,n];offset=y-center;h=ctx.crop_width_m/2
        mass=d.tail_logit[b,n].sigmoid()
        if offset.abs().max()<=h:
            normals=torch.distributions.Normal(head.world_centres(ctx)[b,n],head.kernel_sigma_m(ctx))
            z=(normals.cdf(center+h)-normals.cdf(center-h)).prod(-1)
            pdf=(d.spatial_logits[b,n].softmax(-1)*normals.log_prob(y).sum(-1).exp()/z).sum()
            pdf=(1-mass)*pdf
        else:
            # Point (40,-20) is in +x sector: x=r, y=r*u, dy dx=r du dr.
            assert offset[0]>offset[1].abs()
            radius=offset[0];s=d.tail_scale[b,n,0]*scale
            pdf=mass*d.side_logits[b,n].softmax(-1)[0]*torch.exp(-(radius-h)/s)/(s*2*radius)
        expected.append(-torch.log(pdf)-2*scale.log())
    torch.testing.assert_close(head.nll(ctx,truth,scale,distribution=d),torch.stack(expected).mean())
    torch.testing.assert_close(head.log_masses(d).exp().sum(-1),torch.ones(2,3))


def test_outside_crop_labels_are_finite_supervised_and_not_clipped():
    torch.manual_seed(12); ctx = context(); head = SpatialGoalDensity(32, grid_size=8)
    ctx.map_features.requires_grad_(); ctx.social_h.requires_grad_()
    truth = ctx.map_center + torch.tensor([50., -40.])
    truth[~ctx.valid] = float('nan')
    loss = head.nll(ctx, truth, torch.tensor(2.)); loss.backward()
    assert torch.isfinite(loss)
    assert head.tail[-1].weight.grad.abs().sum() > 0
    assert ctx.social_h.grad.abs().sum() > 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in head.parameters())
    clipped = ctx.map_center + torch.tensor([16., -16.])
    assert not torch.isclose(loss, head.nll(ctx, clipped, torch.tensor(2.)))
    # Force global component to verify output is not clamped to a crop.
    d = head(ctx); d.tail_logit = torch.full_like(d.tail_logit, 100.)
    sampled = head.sample(ctx, torch.tensor(2.), noise=torch.zeros(2,3,20,3), distribution=d)
    assert ((sampled[:,0]-ctx.map_center)[ctx.valid].abs().amax(-1) > 16).all()


def test_sampling_replay_padding_world_order_and_coordinate_equivariance():
    torch.manual_seed(13); ctx = context(batch=1); head = SpatialGoalDensity(32, grid_size=8)
    noise = torch.randn(1,3,20,3)
    a = head.sample(ctx, torch.tensor(1.3), noise=noise)
    assert torch.equal(a, head.sample(ctx, torch.tensor(1.3), noise=noise))
    assert not a.requires_grad
    assert torch.count_nonzero(a.permute(0,2,1,3)[~ctx.valid]) == 0
    shift = torch.tensor([12., -3.])
    moved = dataclasses.replace(ctx, map_center=ctx.map_center+shift, baseline=ctx.baseline+shift)
    b = head.sample(moved, torch.tensor(1.3), noise=noise)
    torch.testing.assert_close((b-a).permute(0,2,1,3)[ctx.valid], shift.expand(2,20,2))
    order = torch.tensor([2,0,1])
    perm = dataclasses.replace(ctx, map_features=ctx.map_features[:,order], social_h=ctx.social_h[:,order],
        map_center=ctx.map_center[:,order], baseline=ctx.baseline[:,order], valid=ctx.valid[:,order])
    torch.testing.assert_close(head.sample(perm,torch.tensor(1.3),noise=noise[:,order]),a[:,:,order])
    worlds = torch.randperm(20)
    torch.testing.assert_close(head.sample(ctx,torch.tensor(1.3),noise=noise[:,:,worlds]),a[:,worlds])
    assert torch.equal(head.sample(ctx,torch.tensor(1.),generator=torch.Generator().manual_seed(42)),
                       head.sample(ctx,torch.tensor(1.),generator=torch.Generator().manual_seed(42)))


def test_spatial_axes_and_local_kernel_sampling_match_world_map():
    ctx = context(batch=1); head = SpatialGoalDensity(32, grid_size=4)
    # Non-square incoming feature maps use the same normalized x/y pixel convention.
    ctx.map_features = torch.randn(1,3,32,7,13)
    d = head(ctx)
    assert d.spatial_logits.shape == (1,3,16)
    d.spatial_logits = torch.full_like(d.spatial_logits, -100.)
    d.spatial_logits[..., 3] = 100.  # row 0, column 3: x=+16, y=-16
    d.tail_logit = torch.full_like(d.tail_logit, -100.)
    noise = torch.zeros(1,3,20,3); noise[...,1]=1.; noise[...,2]=-2.
    actual = head.sample(ctx,torch.tensor(1.),noise=noise,distribution=d)
    standard=torch.distributions.Normal(0.,1.)
    bounds_low=standard.cdf(torch.tensor([-32.,0.])/head.kernel_sigma_m(ctx))
    bounds_high=standard.cdf(torch.tensor([0.,32.])/head.kernel_sigma_m(ctx))
    q=bounds_low+(bounds_high-bounds_low)*standard.cdf(torch.tensor([1.,-2.]))
    expected=ctx.map_center+torch.tensor([16.,-16.])+head.kernel_sigma_m(ctx)*standard.icdf(q)
    torch.testing.assert_close(actual[:,0][ctx.valid],expected[ctx.valid])


def test_density_and_coupling_gradient_responsibilities_and_exact_sets():
    torch.manual_seed(14); ctx=context(); head=SpatialGoalDensity(32,grid_size=8)
    ctx.map_features.requires_grad_();ctx.social_h.requires_grad_()
    pair=LearnableJointCouplingHead(32,singleton_backward='identity')
    g=head.sample(ctx,torch.tensor(1.))
    h,indices,_=pair.assign(ctx,g,torch.tensor(1.),straight_through=True)
    torch.testing.assert_close(h,gather_candidates(g,indices))
    assert torch.equal(indices.sort(1).values,torch.arange(20).view(1,20,1).expand_as(indices))
    joint_pair_loss(h,torch.randn(2,3,2),ctx.valid,torch.tensor(1.)).backward()
    assert all(p.grad is None for p in head.parameters())
    assert sum(p.grad.abs().sum() for p in pair.parameters() if p.grad is not None)>0
    assert ctx.social_h.grad.abs().sum()>0
    head.nll(ctx,ctx.map_center+torch.randn(2,3,2),torch.tensor(1.)).backward()
    for name in ['spatial_features','social_film','spatial_readout','tail']:
        assert sum(p.grad.abs().sum() for p in getattr(head,name).parameters() if p.grad is not None)>0
    assert ctx.map_features.grad.abs().sum()>0


def test_complete_eight_layer_update_and_strict_restore(tmp_path):
    torch.manual_seed(15)
    model=SpatialDensityCouplingBridge(EBJDModel(hidden_dim=192,attention_heads=4,ff_dim=768,
        proposal_conditioned=True,block_type='endpoint_gated_dit_v1',coarse_blocks=4,refinement_blocks=4))
    model.bridge.denoiser.uncheckpointed_max_agent_slots=16
    batch=make_batch((2,),pixels=32)
    root=Path(__file__).resolve().parents[1]
    contract=json.loads((root/'configs/research/spatial_density_bridge_v1.json').read_text())
    config={'training':dict(epochs=100,lr_warmup_epochs=5,lr_final_fraction=.1,new_module_lr=1e-4,unet_lr=1e-4,
        weight_decay=1e-4,optimizer_betas=[.9,.999],optimizer_eps='1e-8')}
    opt=optimizer_for(model,config);gen=torch.Generator().manual_seed(2)
    before=model.goal.spatial_readout.weight.detach().clone()
    agent,scene,_=objective(model,batch,contract);(agent+scene).backward()
    for prefix in ['goal.','coupling.','bridge.encoder.map_unet.','bridge.denoiser.']:
        grads=[p.grad for n,p in model.named_parameters() if n.startswith(prefix) and p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(g.abs().sum() for g in grads)>0,prefix
    opt.step();opt.zero_grad(set_to_none=True)
    assert not torch.equal(before,model.goal.spatial_readout.weight)
    path=tmp_path/'last.pt';save_checkpoint(path,model,opt,1,1,gen,{}, {'schema':SPATIAL_SCHEMA},config)
    ck=torch.load(path,weights_only=False); clone=copy.deepcopy(model); clone_opt=optimizer_for(clone,config)
    clone.load_state_dict(ck['model'],strict=True);clone_opt.load_state_dict(ck['optimizer'])
    assert ck['schema']==SPATIAL_SCHEMA and len(clone_opt.state)==len(opt.state)>0
    assert not any(k.endswith('.grid') for k in ck['model'])
    restore_rng(ck['rng'],gen);a=torch.randn(5)
    restore_rng(ck['rng'],gen);assert torch.equal(a,torch.randn(5))
    # Exact replay of the NEXT update after restoring model/optimizer/all RNG.
    restore_rng(ck['rng'],gen); l1=sum(objective(model,batch,contract)[:2]);l1.backward();opt.step()
    restore_rng(ck['rng'],gen); l2=sum(objective(clone,batch,contract)[:2]);l2.backward();clone_opt.step()
    torch.testing.assert_close(l1,l2,rtol=0,atol=0)
    for a,b in zip(model.parameters(),clone.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_density_rejects_legacy_noise_and_diagnostics_do_not_consume_rng():
    # The strict full config is checked by real-data preflight; density cannot
    # accidentally receive the previous two-number-per-sample noise contract.
    head=SpatialGoalDensity(32,grid_size=8)
    with pytest.raises(ValueError,match='endpoint noise'):
        head.sample(context(),torch.tensor(1.),noise=torch.randn(2,3,20,2))
    ctx=context();d=head(ctx);truth=ctx.map_center+1.
    rng=torch.get_rng_state().clone()
    stats=head.diagnostics(ctx,truth,torch.tensor(1.),distribution=d)
    assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.isfinite(x) and not x.requires_grad for x in stats.values())
    assert 0<=stats['spatial_tail_responsibility']<=1
    assert stats['spatial_outside_crop_fraction']==0


def test_domain_gate_has_no_tail_density_competition_for_in_crop_labels():
    ctx=context();head=SpatialGoalDensity(32,grid_size=8);d=head(ctx)
    inside=ctx.map_center+torch.tensor([1.,2.])
    outside=ctx.map_center+torch.tensor([20.,1.])
    head.nll(ctx,inside,torch.tensor(1.),distribution=d).backward()
    # Branch gate and local spatial density learn; exterior shape cannot steal
    # in-crop supervision. The exterior has gradients for actual out-crop GT.
    assert head.tail[-1].weight.grad[0].abs().sum()>0
    assert head.tail[-1].weight.grad[1:].abs().sum()==0
    assert head.spatial_readout.weight.grad.abs().sum()>0
    head.zero_grad();head.nll(ctx,outside,torch.tensor(1.)).backward()
    assert head.tail[-1].weight.grad[1:].abs().sum()>0
    assert head.spatial_readout.weight.grad.abs().sum()==0


def test_exterior_inverse_sampler_matches_side_and_radius_distribution():
    from ebjd.spatial_density_goal import SpatialDensity
    base=context(batch=1,agents=1)
    # 1000 independent scenes x 20 draws, no network inference needed here.
    ctx=dataclasses.replace(base,valid=torch.ones(1000,1,dtype=torch.bool),
        social_h=base.social_h.expand(1000,-1,-1),map_center=base.map_center.expand(1000,-1,-1))
    head=SpatialGoalDensity(32,grid_size=4)
    d=SpatialDensity(torch.zeros(1000,1,16),torch.full((1000,1),100.),
        torch.tensor([.1,.2,.3,.4]).log().view(1,1,4).expand(1000,1,4),torch.full((1000,1,4),2.))
    y=head.sample(ctx,torch.tensor(1.),distribution=d,generator=torch.Generator().manual_seed(121))
    relative=(y-ctx.map_center[:,None]).reshape(-1,2)
    side=head.exterior_side(relative)
    frequencies=torch.bincount(side,minlength=4)/len(side)
    torch.testing.assert_close(frequencies,torch.tensor([.1,.2,.3,.4]),rtol=0,atol=.015)
    radius=relative.abs().amax(-1);assert (radius>16).all()
    assert abs(float((radius-16).mean())-2.)<.05


def test_legacy_checkpoint_schema_is_not_changed_by_trial_identity(tmp_path):
    model=torch.nn.Linear(2,2);optimizer=torch.optim.AdamW(model.parameters())
    config={'training':dict(epochs=100,lr_warmup_epochs=5,lr_final_fraction=.1)}
    path=tmp_path/'legacy.pt'
    save_checkpoint(path,model,optimizer,0,0,torch.Generator(),{},
        {'schema':'unrelated-research-trial-identity'},config)
    assert torch.load(path,weights_only=False)['schema']=='ebjd-learned-goal-scratch-v1'
