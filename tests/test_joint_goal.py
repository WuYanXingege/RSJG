import torch

from src.joint_goal_loss import (
    JointGoalLoss,
    best_joint_continuous_refinement_loss,
    build_soft_goal_target,
    low_rank_social_mode_loss,
    scene_joint_ranking_loss,
)
from src.models.continuous_goal_refiner import ContinuousJointGoalRefiner
from src.models.joint_goal import LowRankJointGoal
from src.models.model_utils.sampling_2D_map import (
    TTST_test_time_sampling_trick,
    generate_goal_candidates,
)


def test_joint_goal_probabilities_scene_index_prior_and_mask():
    torch.manual_seed(3)
    num_agents, num_candidates, agent_dim, num_modes = 5, 4, 6, 3
    model = LowRankJointGoal(
        agent_dim=agent_dim,
        num_social_modes=num_modes,
        hidden_dim=16,
        dropout=0.0,
    ).eval()
    agent_feat = torch.randn(num_agents, agent_dim)
    candidates = torch.randn(num_agents, num_candidates, 2)
    last_pos = torch.randn(num_agents, 2)
    scene_index = torch.tensor([10, 10, 42, 42, 42])
    candidate_mask = torch.ones(num_agents, num_candidates, dtype=torch.bool)
    candidate_mask[0, 3] = False
    candidate_mask[4] = False  # finite-safe fallback enables candidate zero
    prior = torch.zeros(num_agents, num_candidates)
    prior[:, 2] = 20.0
    prior[4] = float("-inf")

    output = model(
        agent_feat,
        candidates,
        scene_index=scene_index,
        last_pos=last_pos,
        candidate_mask=candidate_mask,
        candidate_log_prior=prior,
    )

    assert output["mode_prob"].shape == (2, num_modes)
    assert output["conditional_goal_prob"].shape == (
        num_agents, num_modes, num_candidates)
    torch.testing.assert_close(
        output["mode_prob"].sum(-1), torch.ones(2))
    torch.testing.assert_close(
        output["conditional_goal_prob"].sum(-1),
        torch.ones(num_agents, num_modes))
    assert torch.all(output["conditional_goal_prob"][0, :, 3] == 0)
    assert torch.all(output["conditional_goal_prob"][4, :, 0] == 1)
    assert torch.isfinite(output["conditional_goal_prob"]).all()
    assert output["candidate_mask"][4].tolist() == [True, False, False, False]
    # The very strong heatmap prior remains visible after the learned residual.
    assert torch.all(output["conditional_goal_prob"][1:4, :, 2] > 0.99)


def test_joint_goal_is_agent_permutation_equivariant():
    torch.manual_seed(8)
    model = LowRankJointGoal(
        agent_dim=5, num_social_modes=2, hidden_dim=12).eval()
    features = torch.randn(6, 5)
    candidates = torch.randn(6, 3, 2)
    last_pos = torch.randn(6, 2)
    scene_index = torch.tensor([9, 1, 9, 1, 9, 1])
    permutation = torch.tensor([4, 0, 5, 2, 1, 3])

    original = model(
        features, candidates, scene_index=scene_index, last_pos=last_pos)
    permuted = model(
        features[permutation], candidates[permutation],
        scene_index=scene_index[permutation], last_pos=last_pos[permutation])

    torch.testing.assert_close(
        original["mode_prob"], permuted["mode_prob"], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        original["conditional_goal_prob"][permutation],
        permuted["conditional_goal_prob"], atol=1e-6, rtol=1e-6)


def test_soft_goal_target_and_low_rank_loss_are_finite_and_differentiable():
    candidates = torch.tensor([
        [[0.0, 0.0], [1.0, 0.0], [4.0, 0.0]],
        [[0.0, 1.0], [2.0, 1.0], [5.0, 1.0]],
        [[1.0, 2.0], [3.0, 2.0], [6.0, 2.0]],
    ])
    ground_truth = torch.tensor([[0.2, 0.0], [1.8, 1.0], [3.2, 2.0]])
    target = build_soft_goal_target(candidates, ground_truth, sigma_goal=0.5)
    torch.testing.assert_close(target.sum(-1), torch.ones(3))
    assert target.argmax(-1).tolist() == [0, 1, 1]

    scene_index = torch.tensor([4, 4, 9])
    mode_logits = torch.tensor(
        [[0.3, -0.4], [-0.2, 0.6]], requires_grad=True)
    conditional_logits = torch.randn(3, 2, 3, requires_grad=True)
    loss = low_rank_social_mode_loss(
        mode_logits, conditional_logits, target, scene_index=scene_index)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert mode_logits.grad is not None
    assert conditional_logits.grad is not None
    assert torch.isfinite(mode_logits.grad).all()
    assert torch.isfinite(conditional_logits.grad).all()


def test_joint_goal_loss_supports_no_edge_and_single_agent():
    candidates = torch.tensor([[[0.0, 0.0], [2.0, 0.0], [5.0, 0.0]]])
    ground_truth = torch.tensor([[1.8, 0.0]])
    mode_logits = torch.tensor([[0.0, 1.0]], requires_grad=True)
    conditional_logits = torch.tensor(
        [[[0.1, 1.1, -0.5], [-0.2, 1.5, 0.0]]], requires_grad=True)
    criterion = JointGoalLoss(sigma_goal=0.4, energy_weight=1.0)

    losses = criterion(
        candidates,
        ground_truth,
        mode_logits,
        conditional_logits,
        scene_index=torch.tensor([12]),
    )
    assert set(losses) == {"mode_loss", "pseudo_likelihood_loss"}
    assert all(torch.isfinite(value) for value in losses.values())
    (losses["mode_loss"] + losses["pseudo_likelihood_loss"]).backward()
    assert mode_logits.grad is not None
    assert conditional_logits.grad is not None


def test_ttst_single_cluster_and_zero_mass_rows_have_finite_fallback():
    probability = torch.zeros(2, 1, 4, 4)

    legacy = TTST_test_time_sampling_trick(
        probability, num_goals=1, device=torch.device('cpu'))
    assert legacy.shape == (2, 2, 1, 2)  # one cluster plus legacy argmax
    assert torch.isfinite(legacy).all()

    candidates, candidate_prob = generate_goal_candidates(
        probability, num_candidates=2, device=torch.device('cpu'),
        use_ttst=True)
    assert candidates.shape == (2, 2, 2)
    assert candidate_prob.shape == (2, 2)
    assert torch.isfinite(candidates).all()
    assert torch.isfinite(candidate_prob).all()
    torch.testing.assert_close(candidate_prob.sum(-1), torch.ones(2))


def test_scene_likelihood_can_be_normalized_by_agent_count():
    mode_log_prob = torch.zeros(1, 1)
    conditional_log_prob = torch.tensor([
        [[0.8, 0.2]],
        [[0.8, 0.2]],
    ]).log()
    target = torch.tensor([[1.0, 0.0], [1.0, 0.0]])

    unnormalized = low_rank_social_mode_loss(
        mode_log_prob, conditional_log_prob, target)
    normalized = low_rank_social_mode_loss(
        mode_log_prob, conditional_log_prob, target,
        normalize_by_num_agents=True)

    torch.testing.assert_close(unnormalized, 2.0 * normalized)


def test_scene_joint_ranking_pushes_probability_toward_better_joint_mode():
    candidates = torch.tensor([
        [[0.0, 0.0], [4.0, 0.0]],
        [[0.0, 1.0], [4.0, 1.0]],
    ])
    ground_truth = torch.tensor([[0.0, 0.0], [0.0, 1.0]])
    # Mode 0 selects the correct candidate; mode 1 selects the distant one.
    conditional_logits = torch.tensor([
        [[4.0, -4.0], [-4.0, 4.0]],
        [[4.0, -4.0], [-4.0, 4.0]],
    ], requires_grad=True)
    mode_logits = torch.tensor([[-2.0, 2.0]], requires_grad=True)

    loss, details = scene_joint_ranking_loss(
        mode_logits,
        conditional_logits,
        candidates,
        ground_truth,
        target_temperature=0.5,
        return_details=True,
    )
    loss.backward()

    assert details['scene_endpoint_error'][0, 0] < \
        details['scene_endpoint_error'][0, 1]
    assert mode_logits.grad[0, 0] < 0
    assert mode_logits.grad[0, 1] > 0
    assert torch.isfinite(conditional_logits.grad).all()


def test_continuous_refiner_is_identity_at_initialization_and_bounded():
    torch.manual_seed(41)
    refiner = ContinuousJointGoalRefiner(
        agent_dim=5,
        relation_dim=2,
        hidden_dim=12,
        num_steps=2,
        max_delta=0.4,
    )
    goals = torch.randn(3, 4, 2)
    last_pos = torch.randn(3, 2)
    features = torch.randn(3, 5)
    edge_index = torch.tensor([[0, 1], [1, 2]])
    edge_feat = torch.randn(2, 14)
    edge_weight = torch.tensor([0.7, 0.4])
    relation_prob = torch.softmax(torch.randn(2, 2), dim=-1)

    initial = refiner(
        goals, last_pos, features, edge_index, edge_feat, edge_weight,
        relation_prob)
    torch.testing.assert_close(initial, goals)

    with torch.no_grad():
        refiner.delta_head[-1].bias.copy_(torch.tensor([5.0, 5.0]))
    refined = refiner(
        goals, last_pos, features, edge_index, edge_feat, edge_weight,
        relation_prob)
    displacement = torch.linalg.vector_norm(refined - goals, dim=-1)
    assert displacement.max() <= 0.400001

    objectives = best_joint_continuous_refinement_loss(
        goals, refined, torch.zeros(3, 2), scene_index=torch.tensor([0, 0, 0]))
    assert torch.isfinite(objectives['continuous_refinement_loss'])
    assert torch.isfinite(objectives['refinement_delta_loss'])
