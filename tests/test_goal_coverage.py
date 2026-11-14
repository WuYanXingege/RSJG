import dataclasses
import torch
from ebjd.goal_coverage import marginal_goal_coverage
from ebjd.learned_goal import LearnedGoalMixture
from ebjd.unified import LearnableJointCouplingHead, joint_pair_loss
from test_unified_coupling import context


def test_singleton_equals_existing_joint_softmin_and_joint_differs_for_multiple_agents():
    torch.manual_seed(12);scale=torch.tensor(1.)
    p=torch.randn(2,20,1,2);gt=torch.randn(2,1,2);valid=torch.ones(2,1,dtype=torch.bool)
    torch.testing.assert_close(marginal_goal_coverage(p,gt,valid,scale),joint_pair_loss(p,gt,valid,scale))
    # Half the worlds serve person 0, half serve person 1: good marginal
    # coverage does not imply a good common world.
    p=torch.zeros(1,20,2,2);p[:,:10,0,0]=2.;p[:,10:,1,0]=2.
    gt=torch.zeros(1,2,2);valid=torch.ones(1,2,dtype=torch.bool)
    assert marginal_goal_coverage(p,gt,valid,scale)<.1
    assert joint_pair_loss(p,gt,valid,scale)>.99


def test_coverage_padding_agent_world_permutation_and_gradient():
    torch.manual_seed(13);p=torch.randn(2,20,3,2,requires_grad=True);gt=torch.randn(2,3,2)
    valid=torch.tensor([[True,True,False],[True,False,False]]);scale=torch.tensor(1.2)
    loss=marginal_goal_coverage(p,gt,valid,scale)
    a=torch.tensor([2,0,1]);w=torch.randperm(20)
    torch.testing.assert_close(loss,marginal_goal_coverage(p[:,w][:,:,a],gt[:,a],valid[:,a],scale))
    loss.backward();assert torch.isfinite(p.grad).all() and p.grad.abs().sum()>0
    assert torch.count_nonzero(p.grad.permute(0,2,1,3)[~valid])==0
    altered=p.detach().clone().permute(0,2,1,3);altered[~valid]=10000
    torch.testing.assert_close(loss,marginal_goal_coverage(altered.permute(0,2,1,3),gt,valid,scale))


def test_detach_only_blocks_goal_coordinates_not_coupling_or_shared_context():
    torch.manual_seed(14);ctx=context();ctx.social_h.requires_grad_();ctx.memory.requires_grad_()
    goal=LearnedGoalMixture(32);coupling=LearnableJointCouplingHead(32,singleton_backward='identity')
    proposals=goal.sample(ctx,torch.tensor(1.));gt=torch.randn(2,3,2)
    h,_,_=coupling.assign(ctx,proposals.detach(),torch.tensor(1.),straight_through=True)
    pair=joint_pair_loss(h,gt,ctx.valid,torch.tensor(1.));pair.backward(retain_graph=True)
    assert all(p.grad is None for p in goal.parameters())
    grads=[p.grad for p in coupling.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads) and sum(g.abs().sum() for g in grads)>0
    assert ctx.social_h.grad is not None and ctx.social_h.grad.abs().sum()>0
    coverage=marginal_goal_coverage(proposals,gt,ctx.valid,torch.tensor(1.))
    nll=goal.nll(ctx,gt,torch.tensor(1.));(nll+coverage).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in goal.parameters())
    assert sum(p.grad.abs().sum() for p in goal.parameters())>0
    assert ctx.memory.grad is not None and ctx.memory.grad.abs().sum()>0
