import copy
import dataclasses
import torch
from ebjd.learned_goal import LearnedGoalMixture
from ebjd.spatial_goal import SpatialGoalMixture, install_spatial_goal
from test_unified_coupling import context


def test_exact_initial_migration_and_optimizer_moments():
    torch.manual_seed(1)
    model = torch.nn.Module(); model.goal = LearnedGoalMixture(32)
    ctx = context(); truth = torch.randn(2,3,2); scale = torch.tensor(1.)
    factory = lambda m: torch.optim.AdamW(m.parameters(),lr=1e-4)
    opt=factory(model)
    model.goal.nll(ctx,truth,scale).backward();opt.step();opt.zero_grad(set_to_none=True)
    before=copy.deepcopy(model); old_states={n:copy.deepcopy(opt.state[p]) for n,p in model.named_parameters()}
    opt=install_spatial_goal(model,opt,factory)
    for a,b in zip(before.goal(ctx),model.goal(ctx)): assert torch.equal(a,b)
    for n,p in model.named_parameters():
        if n in old_states:
            for k,v in old_states[n].items():assert torch.equal(v,opt.state[p][k])
        else:assert p not in opt.state
    before.goal.nll(ctx,truth,scale).backward();model.goal.nll(ctx,truth,scale).backward()
    old=dict(before.named_parameters())
    for n,p in model.named_parameters():
        if n in old:torch.testing.assert_close(p.grad,old[n].grad,atol=0,rtol=0)
    opt.step();opt.zero_grad(set_to_none=True)
    model.goal.nll(ctx,truth,scale).backward()
    for n,p in model.goal.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(),n
        if n.startswith('spatial_'): assert p.grad.abs().sum()>0,n
    restored=copy.deepcopy(model); restored.load_state_dict(model.state_dict(),strict=True)


def test_spatial_readout_is_not_only_map_average_and_preserves_agent_identity():
    torch.manual_seed(2);head=SpatialGoalMixture(32);ctx=context(batch=1)
    torch.nn.init.normal_(head.spatial_output.weight,std=.02)
    eps=torch.randn(1,3,20,2);scale=torch.tensor(1.)
    actual=head.sample(ctx,scale,noise=eps)
    order=torch.tensor([2,0,1])
    moved=dataclasses.replace(ctx,memory=ctx.memory[:,order],social_h=ctx.social_h[:,order],
        baseline=ctx.baseline[:,order],valid=ctx.valid[:,order])
    torch.testing.assert_close(head.sample(moved,scale,noise=eps[:,order]),actual[:,:,order])
    assert torch.count_nonzero(actual.permute(0,2,1,3)[~ctx.valid])==0
    # Perturb two spatial contents with opposite offsets: identical pooled mean,
    # different token contents. Token position is already encoded by MSCE.
    changed=ctx.memory.clone();delta=torch.randn_like(changed[:,:,8]);changed[:,:,8]+=delta;changed[:,:,9]-=delta
    alt=dataclasses.replace(ctx,memory=changed)
    torch.testing.assert_close(ctx.memory[:,:,8:].mean(-2),alt.memory[:,:,8:].mean(-2))
    assert not torch.allclose(head(ctx)[0],head(alt)[0])
    shifted=dataclasses.replace(ctx,baseline=ctx.baseline+torch.tensor([5.,-3.]))
    torch.testing.assert_close((head.sample(shifted,scale,noise=eps)-actual).permute(0,2,1,3)[ctx.valid],
        torch.tensor([5.,-3.]).expand(2,20,2))


def test_spatial_readout_gradient_reaches_shared_encoder_memory():
    torch.manual_seed(3);ctx=context();ctx.memory.requires_grad_();ctx.social_h.requires_grad_()
    head=SpatialGoalMixture(32);torch.nn.init.normal_(head.spatial_output.weight,std=.02)
    head.nll(ctx,torch.randn(2,3,2),torch.tensor(1.)).backward()
    assert ctx.memory.grad[:,:,8:].abs().sum()>0
    assert ctx.social_h.grad.abs().sum()>0
