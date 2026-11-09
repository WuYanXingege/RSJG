import pytest
import torch

from ebjd.encoders import EBJDContext
from ebjd.unified import (
    LearnableJointCouplingHead, gather_candidates, hard_assignment_indices,
    joint_pair_loss, log_sinkhorn, permutation_matrix, UnifiedCouplingBridge,
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


def test_singleton_identity_surrogate_has_no_coupling_or_context_gradient():
    torch.manual_seed(41)
    ctx = context(batch=1, agents=1)
    ctx.social_h.requires_grad_(True)
    head = LearnableJointCouplingHead(32, 4, 20, singleton_backward="identity")
    proposals = torch.randn(1, 20, 1, 2)
    coupled, indices, scores = head.assign(
        ctx, proposals, torch.tensor(1.0), straight_through=True)
    scores.retain_grad()
    torch.testing.assert_close(coupled, proposals, rtol=0, atol=0)
    assert torch.equal(indices[0, :, 0], torch.arange(20))
    joint_pair_loss(coupled, torch.randn(1, 1, 2), ctx.valid,
                    torch.tensor(1.0)).backward()
    assert torch.count_nonzero(scores.grad) == 0
    assert torch.count_nonzero(ctx.social_h.grad) == 0
    assert all(p.grad is None or torch.count_nonzero(p.grad) == 0
               for p in head.parameters())


def test_singleton_identity_surrogate_keeps_multiagent_gradients_and_masks_padding():
    torch.manual_seed(42)
    ctx = context()
    head = LearnableJointCouplingHead(32, 4, 20, singleton_backward="identity")
    proposals = torch.randn(2, 20, 3, 2)
    coupled, indices, scores = head.assign(
        ctx, proposals, torch.tensor(1.0), straight_through=True)
    scores.retain_grad()
    joint_pair_loss(coupled, torch.randn(2, 3, 2), ctx.valid,
                    torch.tensor(1.0)).backward()
    assert torch.count_nonzero(scores.grad[0, :2]) > 0
    assert torch.count_nonzero(scores.grad[0, 2]) == 0
    assert torch.count_nonzero(scores.grad[1]) == 0
    assert sum(float(p.grad.abs().sum()) for p in head.parameters()
               if p.grad is not None) > 0
    torch.testing.assert_close(coupled[1], proposals[1], rtol=0, atol=0)
    torch.testing.assert_close(coupled, gather_candidates(proposals, indices))


def test_default_surrogate_matches_historical_legacy_forward_and_backward():
    torch.manual_seed(43)
    ctx = context()
    proposals = torch.randn(2, 20, 3, 2)
    truth = torch.randn(2, 3, 2)
    head = LearnableJointCouplingHead(32, 4, 20)
    old = LearnableJointCouplingHead(32, 4, 20, singleton_backward="legacy")
    old.load_state_dict(head.state_dict(), strict=True)
    actual, indices, scores = head.assign(
        ctx, proposals, torch.tensor(1.0), straight_through=True)
    scores.retain_grad()
    old_scores = old(ctx, proposals, torch.tensor(1.0))
    old_indices = hard_assignment_indices(old_scores, ctx.valid)
    hard = permutation_matrix(old_indices, 20).to(old_scores)
    soft = log_sinkhorn(old_scores, iterations=20)
    soft = torch.where(ctx.valid[:, :, None, None], soft,
                       torch.eye(20)[None, None])
    expected = torch.einsum("bnkj,bnjd->bknd", hard + soft - soft.detach(),
                            proposals.permute(0, 2, 1, 3))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(scores, old_scores, rtol=0, atol=0)
    assert torch.equal(indices, old_indices)
    joint_pair_loss(actual, truth, ctx.valid, torch.tensor(1.0)).backward()
    joint_pair_loss(expected, truth, ctx.valid, torch.tensor(1.0)).backward()
    for parameter, old_parameter in zip(head.parameters(), old.parameters()):
        torch.testing.assert_close(parameter.grad, old_parameter.grad, rtol=0, atol=0)
    # The old surrogate really sends a gradient through singleton scenes.
    assert torch.count_nonzero(scores.grad[1, 0]) > 0


def test_singleton_surrogate_option_is_validated_and_forwarded():
    with pytest.raises(ValueError, match="singleton_backward"):
        LearnableJointCouplingHead(32, 4, 20, singleton_backward="unknown")

    class BridgeStub(torch.nn.Module):
        def __init__(self):
            super().__init__()
            from types import SimpleNamespace
            self.denoiser = SimpleNamespace(proposal_conditioned=True)
            self.encoder = SimpleNamespace(
                history_stem=[torch.nn.Linear(2, 32)],
                history_social=[SimpleNamespace(attn=SimpleNamespace(heads=4))])

    wrapped = UnifiedCouplingBridge(BridgeStub(), singleton_backward="identity")
    assert wrapped.coupling.singleton_backward == "identity"
