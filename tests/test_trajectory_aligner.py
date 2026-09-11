import torch

from src.models.interaction_graph import EDGE_FEATURE_DIM
from src.models.trajectory_aligner import RelationAwareTrajectoryAligner


def _inputs(num_agents=3, num_samples=4, num_steps=5):
    torch.manual_seed(71)
    trajectories = torch.randn(num_agents, num_samples, num_steps, 2)
    last_pos = torch.randn(num_agents, 2)
    agent_feat = torch.randn(num_agents, 8)
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    edge_feat = torch.randn(1, EDGE_FEATURE_DIM)
    edge_weight = torch.ones(1)
    relation_prob = torch.softmax(torch.randn(1, 3), dim=-1)
    scene_index = torch.zeros(num_agents, dtype=torch.long)
    return (trajectories, last_pos, agent_feat, edge_index, edge_feat,
            edge_weight, relation_prob, scene_index)


def test_hard_alignment_preserves_every_agent_sample_set_exactly():
    model = RelationAwareTrajectoryAligner(
        agent_dim=8, relation_dim=3, hidden_dim=16,
        keep_threshold=0.0).eval()
    inputs = _inputs()
    trajectories = inputs[0]

    result = model.align(*inputs, hard=True)
    aligned = result['aligned_trajectories']

    # Sorting unique flattened sample signatures avoids assuming which hard
    # permutation the randomly initialized scorer happens to choose.
    original_signature = trajectories.flatten(2).sum(dim=-1).sort(dim=1).values
    aligned_signature = aligned.flatten(2).sum(dim=-1).sort(dim=1).values
    torch.testing.assert_close(aligned_signature, original_signature)
    for permutation in result['permutation']:
        assert torch.equal(
            permutation.sort().values,
            torch.arange(trajectories.shape[1]))

    # Agent 2 is an isolated graph component and must be an exact bypass.
    torch.testing.assert_close(aligned[2], trajectories[2], rtol=0, atol=0)


def test_fresh_aligner_uses_conservative_identity_path():
    model = RelationAwareTrajectoryAligner(
        agent_dim=8, relation_dim=3, hidden_dim=16,
        keep_threshold=0.6).eval()
    inputs = _inputs()
    result = model.align(*inputs, hard=True)
    expected = torch.arange(inputs[0].shape[1])[None].expand(
        inputs[0].shape[0], -1)
    assert torch.equal(result['permutation'], expected)
    torch.testing.assert_close(
        result['aligned_trajectories'], inputs[0], rtol=0, atol=0)


def test_actual_trajectory_alignment_loss_is_finite_and_differentiable():
    model = RelationAwareTrajectoryAligner(
        agent_dim=8, relation_dim=3, hidden_dim=16,
        sinkhorn_iterations=5, alignment_iterations=2)
    inputs = _inputs()
    trajectories = inputs[0]
    ground_truth = torch.randn(
        trajectories.shape[0], trajectories.shape[2], 2)
    future_mask = torch.ones(
        trajectories.shape[0], trajectories.shape[2], dtype=torch.bool)

    losses = model.loss(
        trajectories, ground_truth, future_mask, *inputs[1:])
    optimized = (
        losses['trajectory_alignment_loss'] +
        losses['trajectory_pair_loss'] +
        losses['alignment_no_harm_loss'] +
        0.02 * losses['alignment_entropy_loss'])
    assert torch.isfinite(optimized)
    optimized.backward()
    gradients = [
        parameter.grad for parameter in model.parameters()
        if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(gradient.abs().sum() > 0 for gradient in gradients)


def test_empty_graph_is_exact_identity_and_has_finite_zero_pair_loss():
    model = RelationAwareTrajectoryAligner(
        agent_dim=8, relation_dim=0, hidden_dim=16)
    trajectories = torch.randn(1, 3, 5, 2)
    last_pos = torch.randn(1, 2)
    agent_feat = torch.randn(1, 8)
    edge_index = torch.empty(2, 0, dtype=torch.long)
    edge_feat = torch.empty(0, EDGE_FEATURE_DIM)
    edge_weight = torch.empty(0)
    ground_truth = torch.randn(1, 5, 2)
    mask = torch.ones(1, 5, dtype=torch.bool)

    result = model.align(
        trajectories, last_pos, agent_feat, edge_index, edge_feat,
        edge_weight, scene_index=torch.tensor([4]), hard=True)
    torch.testing.assert_close(
        result['aligned_trajectories'], trajectories, rtol=0, atol=0)
    losses = model.loss(
        trajectories, ground_truth, mask, last_pos, agent_feat, edge_index,
        edge_feat, edge_weight, scene_index=torch.tensor([4]))
    assert torch.isfinite(losses['trajectory_pair_loss'])
