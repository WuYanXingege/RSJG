import torch

from ebjd.encoders import EBJDContext
from ebjd.unified import (
    LearnableJointCouplingHead, gather_candidates, hard_assignment_indices,
    joint_pair_loss, log_sinkhorn,
)


def context(batch=2, agents=3, dim=32):
    valid = torch.tensor([[True, True, False], [True, False, False]])[:batch, :agents]
    observed = torch.randn(batch, agents, 8, 2)
    return EBJDContext(
        memory=torch.randn(batch, agents, 72, dim),
        social_h=torch.randn(batch, agents, dim),
        observed_bias=torch.zeros(batch, 4, agents, agents),
        baseline=torch.randn(batch, agents, 12, 2),
        mean_velocity=torch.randn(batch, agents, 2),
        map_features=torch.randn(batch, agents, 32, 8, 8),
        map_logits=torch.randn(batch, agents, 12, 8, 8),
        map_center=torch.randn(batch, agents, 2), valid=valid,
        observed=observed, crop_width_m=32.0)


def test_sinkhorn_and_hard_assignments_are_bijective_and_padding_safe():
    torch.manual_seed(1)
    scores = torch.randn(2, 3, 20, 20)
    ctx = context()
    soft = log_sinkhorn(scores)
    torch.testing.assert_close(soft.sum(-1), torch.ones_like(soft.sum(-1)), atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(soft.sum(-2), torch.ones_like(soft.sum(-2)), atol=2e-5, rtol=2e-5)
    indices = hard_assignment_indices(scores, ctx.valid)
    for scene in range(2):
        for agent in range(3):
            assert torch.equal(indices[scene, :, agent].sort().values, torch.arange(20))
    assert torch.equal(indices[1, :, 0], torch.arange(20))  # single-person identity


def test_coupling_preserves_candidate_sets_and_pair_loss_has_gradient():
    torch.manual_seed(2)
    ctx = context()
    proposals = torch.randn(2, 20, 3, 2)
    head = LearnableJointCouplingHead(32, 4, 20)
    coupled, indices, _ = head.assign(ctx, proposals, torch.tensor(1.3), straight_through=True)
    hard = gather_candidates(proposals, indices)
    torch.testing.assert_close(coupled, hard)
    for scene in range(2):
        for agent in range(3):
            torch.testing.assert_close(
                coupled[scene, :, agent].sort(0).values,
                proposals[scene, :, agent].sort(0).values)
    loss = joint_pair_loss(coupled, torch.randn(2, 3, 2), ctx.valid, torch.tensor(1.3))
    loss.backward()
    gradients = [p.grad for p in head.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert sum(float(g.abs().sum()) for g in gradients) > 0


def test_candidate_reordering_reorders_score_columns_without_index_embedding():
    torch.manual_seed(3)
    ctx = context(batch=1)
    proposals = torch.randn(1, 20, 3, 2)
    order = torch.randperm(20)
    head = LearnableJointCouplingHead(32, 4, 20).eval()
    left = head(ctx, proposals, torch.tensor(1.0))
    right = head(ctx, proposals[:, order], torch.tensor(1.0))
    torch.testing.assert_close(right, left[..., order], atol=2e-5, rtol=2e-5)
