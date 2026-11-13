import copy
import dataclasses
import json
import sys
from pathlib import Path
from unittest.mock import patch
import pytest
import torch
from ebjd.learned_goal import LearnedGoalMixture, LearnedGoalCouplingBridge
from ebjd.model import EBJDModel
from ebjd.unified import LearnableJointCouplingHead, joint_pair_loss, gather_candidates
from test_unified_coupling import context
from conftest import make_batch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from train_learned_goal_bridge import objective, optimizer_for, set_epoch_lr, save_checkpoint, restore_rng, capture_rng


def test_nll_matches_explicit_uniform_gaussian_mixture():
    torch.manual_seed(3); ctx=context(); head=LearnedGoalMixture(32)
    mu,std=head(ctx);truth=torch.randn(2,3,2);scale=torch.tensor(1.3)
    dist=torch.distributions.MixtureSameFamily(torch.distributions.Categorical(probs=torch.ones(2,3,20)/20),
        torch.distributions.Independent(torch.distributions.Normal(mu,std),1))
    target=(truth-ctx.baseline[...,-1,:])/scale
    expected=-dist.log_prob(target)[ctx.valid].mean()
    torch.testing.assert_close(head.nll(ctx,truth,scale),expected)


def test_joint_loss_updates_goal_and_preserves_bijection():
    torch.manual_seed(5);ctx=context();ctx.memory.requires_grad_();ctx.social_h.requires_grad_()
    head=LearnedGoalMixture(32);pair=LearnableJointCouplingHead(32,singleton_backward='identity')
    eps=torch.randn(2,3,20,2);goal=head.sample(ctx,torch.tensor(1.),noise=eps)
    h,permutation,_=pair.assign(ctx,goal,torch.tensor(1.),straight_through=True)
    torch.testing.assert_close(h,gather_candidates(goal,permutation))
    loss=head.nll(ctx,torch.randn(2,3,2),torch.tensor(1.))+joint_pair_loss(h,torch.randn(2,3,2),ctx.valid,torch.tensor(1.))
    loss.backward()
    for module in [head,pair]:
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in module.parameters())
        assert sum(p.grad.abs().sum() for p in module.parameters() if p.grad is not None)>0
    assert ctx.memory.grad.abs().sum()>0 and ctx.social_h.grad.abs().sum()>0
    old=goal.detach().clone();torch.optim.Adam(head.parameters(),lr=1e-3).step()
    new=head.sample(ctx,torch.tensor(1.),noise=eps)
    assert not torch.equal(old,new)  # endpoints no longer fixed across training
    assert torch.count_nonzero(new.permute(0,2,1,3)[~ctx.valid])==0


def test_agent_permutation_and_translation_equivariance_fixed_noise():
    torch.manual_seed(6);ctx=context(batch=1);head=LearnedGoalMixture(32)
    eps=torch.randn(1,3,20,2);original=head.sample(ctx,torch.tensor(1.),noise=eps)
    order=torch.tensor([2,0,1]);permuted=dataclasses.replace(ctx,memory=ctx.memory[:,order],
        social_h=ctx.social_h[:,order],baseline=ctx.baseline[:,order],valid=ctx.valid[:,order])
    torch.testing.assert_close(head.sample(permuted,torch.tensor(1.),noise=eps[:,order]),original[:,:,order])
    translated=dataclasses.replace(ctx,baseline=ctx.baseline+torch.tensor([3.,-2.]))
    actual=head.sample(translated,torch.tensor(1.),noise=eps)
    torch.testing.assert_close((actual-original).permute(0,2,1,3)[ctx.valid],torch.tensor([3.,-2.]).expand(2,20,2))


def test_scratch_config_never_loads_pretrained_weights():
    from conftest import load_paper_config
    from ebjd.config import build_model
    c=load_paper_config();c['paths'].update(initialization_role='all_random_scratch',
        accepted_fold_matched_gdts_unet=None,accepted_fold_matched_gdts_unet_sha256=None)
    with patch('torch.load',side_effect=AssertionError('no checkpoint may be read')):
        m=build_model(c,proposal_conditioned=True)
    c['paths']['accepted_fold_matched_gdts_unet']='forbidden.pt'
    with pytest.raises(ValueError,match='forbids'):build_model(c)


def test_full_objective_and_complete_restore(tmp_path):
    torch.set_num_threads(2);torch.manual_seed(7)
    model=LearnedGoalCouplingBridge(EBJDModel(hidden_dim=192,attention_heads=4,ff_dim=768,
        proposal_conditioned=True,block_type='endpoint_gated_dit_v1',coarse_blocks=4,refinement_blocks=4))
    model.bridge.denoiser.uncheckpointed_max_agent_slots=16
    batch=make_batch((2,),pixels=32)
    contract=json.loads((Path(__file__).resolve().parents[1]/'configs/research/learned_goal_scratch100_v1.json').read_text())
    config={'training':dict(epochs=100,lr_warmup_epochs=5,lr_final_fraction=.1,new_module_lr=1e-4,unet_lr=1e-4,
        weight_decay=1e-4,optimizer_betas=[.9,.999],optimizer_eps='1e-8')}
    opt=optimizer_for(model,config);gen=torch.Generator().manual_seed(2)
    agent,scene,_=objective(model,batch,contract);(agent+scene).backward()
    for prefix in ['goal.','coupling.','bridge.encoder.map_unet.','bridge.denoiser.']:
        grads=[p.grad for n,p in model.named_parameters() if n.startswith(prefix) and p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(g.abs().sum() for g in grads)>0,prefix
    opt.step();opt.zero_grad(set_to_none=True)
    set_epoch_lr(opt,31,config)
    assert opt.param_groups[0]['lr']>1e-5  # not legacy 30-epoch cosine
    path=tmp_path/'last.pt';save_checkpoint(path,model,opt,31,7,gen,{}, {'test':True},config)
    ck=torch.load(path,weights_only=False);copy_model=copy.deepcopy(model);copy_opt=optimizer_for(copy_model,config)
    copy_model.load_state_dict(ck['model'],strict=True);copy_opt.load_state_dict(ck['optimizer'])
    assert ck['scheduler_state']['total_epochs']==100 and ck['early_stop_state']=={'enabled':False}
    restore_rng(ck['rng'],gen);a=torch.randn(4);ga=torch.rand(4,generator=gen)
    restore_rng(ck['rng'],gen);assert torch.equal(a,torch.randn(4));assert torch.equal(ga,torch.rand(4,generator=gen))
    assert len(copy_opt.state)==len(opt.state)>0
    set_epoch_lr(copy_opt,100,config);assert copy_opt.param_groups[0]['lr']==pytest.approx(1e-5)
