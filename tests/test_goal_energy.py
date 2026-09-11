import torch

from src.joint_goal_loss import pseudo_likelihood_loss
from src.models.goal_energy import (
    LowRankGoalEnergy,
    group_relative_displacement_feature,
    relation_energy_from_factors,
    relation_mixture_effective_energy,
    selected_effective_energy,
)
from src.models.interaction_graph import (
    EDGE_FEATURE_DIM,
    build_interaction_graph,
)
from src.models.joint_sampler import JointGoalSampler


def _energy_inputs():
    torch.manual_seed(11)
    num_agents, num_candidates = 4, 3
    agent_feat = torch.randn(num_agents, 5)
    goal_candidates = torch.randn(num_agents, num_candidates, 2)
    last_pos = torch.randn(num_agents, 2)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
    edge_feat = torch.randn(3, 4)
    relation_prob = torch.softmax(torch.randn(3, 3), dim=-1)
    return (agent_feat, goal_candidates, last_pos, edge_index,
            edge_feat, relation_prob)


def test_low_rank_energy_shapes_finite_gradients_and_sparse_storage():
    inputs = _energy_inputs()
    model = LowRankGoalEnergy(
        agent_dim=5, edge_dim=4, num_relation_modes=3,
        rank=2, hidden_dim=16)
    output = model(
        inputs[0], inputs[1], inputs[2], inputs[3],
        relation_prob=inputs[5], edge_feat=inputs[4],
        scene_index=torch.tensor([0, 0, 0, 0]),
        return_full_matrix=True,
    )

    assert output["left_factor"].shape == (3, 3, 3, 2)
    assert output["right_factor"].shape == (3, 3, 3, 2)
    assert output["relation_energy"].shape == (3, 3, 3, 3)
    assert output["effective_energy"].shape == (3, 3, 3)
    assert torch.isfinite(output["effective_energy"]).all()
    # The first dimension is sparse edges E, never a dense [N,N,...] pair.
    assert output["effective_energy"].shape[0] == inputs[3].shape[1]

    output["effective_energy"].mean().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters() if parameter.grad is not None)


def test_relation_mixture_uses_logsumexp_not_expected_energy():
    relation_energy = torch.tensor([[[[0.0]], [[4.0]]]])  # [E=1,M=2,K=1,K=1]
    relation_prob = torch.tensor([[0.5, 0.5]])
    effective = relation_mixture_effective_energy(
        relation_energy, relation_prob)
    expected_formula = -torch.log(
        0.5 * torch.exp(torch.tensor(0.0)) +
        0.5 * torch.exp(torch.tensor(-4.0)))
    torch.testing.assert_close(effective.squeeze(), expected_formula)
    assert not torch.isclose(effective.squeeze(), torch.tensor(2.0))


def test_selected_neighbor_energy_matches_full_matrix_in_both_directions():
    inputs = _energy_inputs()
    model = LowRankGoalEnergy(
        agent_dim=5, edge_dim=4, num_relation_modes=3,
        rank=4, hidden_dim=12)
    output = model(
        inputs[0], inputs[1], inputs[2], inputs[3],
        relation_prob=inputs[5], edge_feat=inputs[4],
        return_full_matrix=True,
    )
    full = output["effective_energy"]
    destination_choice = torch.tensor([0, 2, 1])
    source_choice = torch.tensor([2, 0, 1])

    source_conditional = selected_effective_energy(
        output["left_factor"], output["right_factor"],
        output["relation_prob"], destination_choice,
        conditioned_side="source", edge_weight=output["edge_weight"])
    destination_conditional = selected_effective_energy(
        output["left_factor"], output["right_factor"],
        output["relation_prob"], source_choice,
        conditioned_side="destination", edge_weight=output["edge_weight"])
    edge_rows = torch.arange(full.shape[0])
    torch.testing.assert_close(
        source_conditional, full[edge_rows, :, destination_choice])
    torch.testing.assert_close(
        destination_conditional, full[edge_rows, source_choice, :])


def test_group_relative_feature_represents_formation_preservation():
    # Two pedestrians walking together with a one-metre lateral offset.
    last_pos = torch.tensor([[2.0, 0.0], [2.0, 1.0]])
    candidates = torch.tensor([
        [[3.0, 0.0], [3.0, 3.0]],
        [[3.0, 1.0], [3.0, -3.0]],
    ])
    edge_index = torch.tensor([[0], [1]])
    feature = group_relative_displacement_feature(
        candidates, last_pos, edge_index)
    assert feature.shape == (1, 2, 2, 2)
    torch.testing.assert_close(feature[0, 0, 0], torch.zeros(2))
    assert torch.linalg.vector_norm(feature[0, 1, 1]) > 0


def test_group_relative_ablation_preserves_shapes_but_changes_factor_input():
    inputs = _energy_inputs()
    with_feature = LowRankGoalEnergy(
        agent_dim=5, edge_dim=4, num_relation_modes=3, rank=3,
        hidden_dim=10, use_group_relative_feature=True)
    without_feature = LowRankGoalEnergy(
        agent_dim=5, edge_dim=4, num_relation_modes=3, rank=3,
        hidden_dim=10, use_group_relative_feature=False)
    without_feature.load_state_dict(with_feature.state_dict())
    output_on = with_feature(
        inputs[0], inputs[1], inputs[2], inputs[3],
        relation_prob=inputs[5], edge_feat=inputs[4])
    output_off = without_feature(
        inputs[0], inputs[1], inputs[2], inputs[3],
        relation_prob=inputs[5], edge_feat=inputs[4])
    assert output_on["left_factor"].shape == output_off["left_factor"].shape
    assert not torch.allclose(
        output_on["left_factor"], output_off["left_factor"])


def test_energy_and_pseudo_likelihood_handle_no_edges():
    num_agents, num_candidates = 1, 3
    model = LowRankGoalEnergy(
        agent_dim=4, edge_dim=2, num_relation_modes=2, rank=2,
        hidden_dim=8)
    output = model(
        torch.randn(num_agents, 4),
        torch.randn(num_agents, num_candidates, 2),
        torch.randn(num_agents, 2),
        torch.empty(2, 0, dtype=torch.long),
        relation_prob=torch.empty(0, 2),
        edge_feat=torch.empty(0, 2),
        scene_index=torch.tensor([5]),
        return_full_matrix=True,
    )
    assert output["effective_energy"].shape == (0, 3, 3)

    unary = torch.randn(1, 3, requires_grad=True)
    target = torch.tensor([[0.1, 0.8, 0.1]])
    loss = pseudo_likelihood_loss(
        unary, target, torch.empty(2, 0, dtype=torch.long),
        torch.empty(0, 3, 3), scene_index=torch.tensor([5]))
    assert torch.isfinite(loss)
    loss.backward()
    assert unary.grad is not None and torch.isfinite(unary.grad).all()


def test_pair_energy_is_equivariant_to_agent_reordering():
    """Reindexing a physical pair only transposes its K-by-K energy."""
    torch.manual_seed(31)
    observations = torch.tensor([
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
        [[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]],
    ])
    agent_feat = torch.randn(2, 5)
    candidates = torch.tensor([
        [[3.0, 0.0], [4.0, 2.0]],
        [[3.0, 1.0], [4.0, -1.0]],
    ])
    relation_prob = torch.tensor([[0.2, 0.3, 0.5]])
    model = LowRankGoalEnergy(
        agent_dim=5, edge_dim=EDGE_FEATURE_DIM,
        num_relation_modes=3, rank=2, hidden_dim=12).eval()

    graph = build_interaction_graph(observations, graph_type='full')
    original = model(
        agent_feat, candidates, observations[:, -1], graph[0],
        relation_prob=relation_prob, edge_feat=graph[1],
        edge_weight=graph[2], return_full_matrix=True)

    permutation = torch.tensor([1, 0])
    permuted_graph = build_interaction_graph(
        observations[permutation], graph_type='full')
    permuted = model(
        agent_feat[permutation], candidates[permutation],
        observations[permutation, -1], permuted_graph[0],
        relation_prob=relation_prob, edge_feat=permuted_graph[1],
        edge_weight=permuted_graph[2], return_full_matrix=True)
    torch.testing.assert_close(
        original['effective_energy'][0],
        permuted['effective_energy'][0].transpose(0, 1))


def test_crossing_energy_can_prefer_coordinated_over_conflicting_choices():
    """Low-rank factors can express a yielding/off-diagonal compatibility."""
    # Candidate index 0 for both agents denotes simultaneous crossing. The
    # two off-diagonal combinations denote one agent yielding. Rank two is
    # already sufficient to make either coordinated choice lower energy.
    left_factor = torch.tensor([[[[3.0, 0.0], [0.0, 3.0]]]])
    right_factor = torch.tensor([[[[0.0, 3.0], [3.0, 0.0]]]])
    energy = relation_energy_from_factors(left_factor, right_factor)[0, 0]

    assert energy[0, 1] < energy[0, 0]
    assert energy[1, 0] < energy[1, 1]


def test_relation_mode_switches_diagonal_vs_yielding_joint_choice():
    """The same pair factors express coordination or off-diagonal yielding."""
    # Relation 0 rewards matching choices; relation 1 rewards opposite choices.
    left_factor = torch.tensor([[
        [[3.0, 0.0], [0.0, 3.0]],
        [[3.0, 0.0], [0.0, 3.0]],
    ]])
    right_factor = torch.tensor([[
        [[3.0, 0.0], [0.0, 3.0]],
        [[0.0, 3.0], [3.0, 0.0]],
    ]])
    candidates = torch.tensor([
        [[0.0, 0.0], [1.0, 0.0]],
        [[0.0, 1.0], [1.0, 1.0]],
    ])
    unary = torch.tensor([[[100.0, -100.0]], [[0.0, 0.0]]])
    sampler = JointGoalSampler(
        num_samples=1, num_refinement_steps=1,
        energy_weight=1.0, sampling_mode='map')

    def select(relation_prob):
        details = sampler(
            candidates,
            mode_log_prob=torch.zeros(1, 1),
            conditional_goal_log_prob=unary,
            scene_index=torch.zeros(2, dtype=torch.long),
            energy_output={
                'edge_index': torch.tensor([[0], [1]]),
                'left_factor': left_factor,
                'right_factor': right_factor,
                'relation_prob': relation_prob,
                'edge_weight': torch.ones(1),
            },
            return_details=True)
        return details['candidate_index'][:, 0].tolist()

    assert select(torch.tensor([[1.0, 0.0]])) == [0, 0]
    assert select(torch.tensor([[0.0, 1.0]])) == [0, 1]
