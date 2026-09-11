"""Unit tests for V4 trajectory-pair relation and energy modules."""

import pytest
import torch

from src.models.interaction_graph import (
    EDGE_FEATURE_DIM,
    build_interaction_graph,
)
from src.models.trajectory_pair_energy import (
    RelationConditionedTrajectoryPairEnergy,
    trajectory_relation_mixture_compatibility,
)
from src.models.trajectory_pair_relation import (
    CandidateConditionedRelation,
    TRAJECTORY_PAIR_GEOMETRY_DIM,
    TrajectoryEncoder,
    build_trajectory_pair_geometry,
    reverse_trajectory_pair_geometry,
)


def _bank_inputs(requires_grad=False):
    torch.manual_seed(404)
    num_agents, num_candidates, num_steps = 3, 4, 6
    last_pos = torch.tensor([
        [0.0, 0.0], [0.0, 1.0], [3.0, 0.5],
    ])
    velocity = torch.tensor([
        [0.30, 0.02], [0.28, -0.01], [-0.15, 0.04],
    ])
    time = torch.arange(1, num_steps + 1).float()[None, None, :, None]
    offsets = 0.08 * torch.randn(num_agents, num_candidates, 1, 2)
    curvature = 0.01 * torch.randn(
        num_agents, num_candidates, 1, 2) * time.square()
    trajectories = (
        last_pos[:, None, None, :] + offsets +
        velocity[:, None, None, :] * time + curvature)
    trajectories.requires_grad_(requires_grad)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    edge_feat = torch.randn(2, EDGE_FEATURE_DIM)
    edge_weight = torch.tensor([1.0, 0.7])
    relation_prior = torch.softmax(torch.randn(2, 4), dim=-1)
    scene_index = torch.tensor([5, 5, 5])
    return (trajectories, last_pos, edge_index, edge_feat, edge_weight,
            relation_prior, scene_index)


def test_complete_trajectory_encoder_and_geometry_shapes_and_gradients():
    inputs = _bank_inputs(requires_grad=True)
    trajectories, last_pos, edge_index = inputs[:3]
    encoder = TrajectoryEncoder(output_dim=16, hidden_dim=20, dt=0.4)

    trajectory_feat = encoder(trajectories, last_pos)
    geometry = build_trajectory_pair_geometry(
        trajectories, last_pos, edge_index, scene_index=inputs[-1], dt=0.4)

    assert trajectory_feat.shape == (3, 4, 16)
    assert geometry.shape == (2, 4, 4, TRAJECTORY_PAIR_GEOMETRY_DIM)
    assert torch.isfinite(trajectory_feat).all()
    assert torch.isfinite(geometry).all()
    # Endpoint/minimum/mean distances cannot be negative.
    assert (geometry[..., :3] >= 0).all()
    # Complete-path encoding and differentiable geometric channels both reach
    # the original trajectory bank.
    loss = trajectory_feat.square().mean() + geometry[..., :8].mean()
    loss.backward()
    assert trajectories.grad is not None
    assert torch.isfinite(trajectories.grad).all()
    assert trajectories.grad.abs().sum() > 0
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in encoder.parameters())


def test_pair_geometry_has_correct_reverse_orientation_and_scene_checks():
    trajectories, last_pos, _, _, _, _, _ = _bank_inputs()
    forward_edge = torch.tensor([[0], [1]], dtype=torch.long)
    reverse_edge = torch.tensor([[1], [0]], dtype=torch.long)
    forward = build_trajectory_pair_geometry(
        trajectories, last_pos, forward_edge)
    reverse = build_trajectory_pair_geometry(
        trajectories, last_pos, reverse_edge)
    expected_reverse = reverse_trajectory_pair_geometry(
        forward).transpose(1, 2)
    torch.testing.assert_close(reverse, expected_reverse)

    with pytest.raises(ValueError, match="cross scene"):
        build_trajectory_pair_geometry(
            trajectories, last_pos, forward_edge,
            scene_index=torch.tensor([0, 1, 1]))
    with pytest.raises(ValueError, match="self edges"):
        build_trajectory_pair_geometry(
            trajectories, last_pos,
            torch.tensor([[0], [0]], dtype=torch.long))


def test_candidate_relation_shapes_prior_ablation_and_gradients():
    inputs = _bank_inputs()
    trajectories, last_pos, edge_index, edge_feat, _, prior, scene = inputs
    encoder = TrajectoryEncoder(output_dim=12)
    trajectory_feat = encoder(trajectories, last_pos)
    geometry = build_trajectory_pair_geometry(
        trajectories, last_pos, edge_index, scene)
    relation = CandidateConditionedRelation(
        trajectory_dim=12, edge_dim=EDGE_FEATURE_DIM,
        num_relation_modes=4, hidden_dim=18, temperature=0.7)

    output = relation(
        trajectory_feat, edge_index, edge_feat, prior, geometry, scene)
    posterior = output["relation_posterior"]
    assert posterior.shape == (2, 4, 4, 4)
    assert output["residual_logits"].shape == posterior.shape
    assert output["relation_prior_kl_pointwise"].shape == (2, 4, 4)
    assert output["relation_prior_kl"].ndim == 0
    torch.testing.assert_close(
        posterior.sum(dim=-1), torch.ones_like(posterior[..., 0]))
    assert torch.isfinite(output["relation_prior_kl"])

    off = relation(
        trajectory_feat, edge_index, edge_feat, prior, geometry, scene,
        trajectory_conditioned=False)
    expected_prior = prior[:, None, None, :].expand_as(off["relation_posterior"])
    torch.testing.assert_close(off["relation_posterior"], expected_prior)
    torch.testing.assert_close(
        off["relation_prior_kl"], torch.zeros_like(off["relation_prior_kl"]),
        atol=1e-6, rtol=0)

    (posterior[..., 0].square().mean() +
     output["relation_prior_kl"]).backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in relation.residual_mlp.parameters())
    assert any(parameter.grad is not None for parameter in encoder.parameters())


@pytest.mark.parametrize("gate_type", ["pair", "edge", "none"])
def test_low_rank_energy_shapes_mixture_and_gate_ablation(gate_type):
    inputs = _bank_inputs()
    trajectories, last_pos, edge_index, edge_feat, weight, prior, scene = inputs
    trajectory_feat = TrajectoryEncoder(output_dim=10)(trajectories, last_pos)
    geometry = build_trajectory_pair_geometry(
        trajectories, last_pos, edge_index, scene)
    posterior = prior[:, None, None, :].expand(-1, 4, 4, -1)
    energy = RelationConditionedTrajectoryPairEnergy(
        trajectory_dim=10, edge_dim=EDGE_FEATURE_DIM,
        num_relation_modes=4, rank=3, hidden_dim=16,
        gate_type=gate_type)
    output = energy(
        trajectory_feat, edge_index, edge_feat, posterior, geometry,
        edge_weight=weight, scene_index=scene)

    assert output["left_factor"].shape == (2, 4, 4, 3)
    assert output["right_factor"].shape == (2, 4, 4, 3)
    assert output["relation_energy"].shape == (2, 4, 4, 4)
    assert output["effective_energy"].shape == (2, 4, 4)
    assert output["pair_score"].shape == (2, 4, 4)
    assert output["pair_gate"].shape == (2, 4, 4)
    assert output["edge_gate"].shape == (2,)
    assert all(torch.isfinite(value).all() for value in output.values())
    if gate_type == "none":
        torch.testing.assert_close(
            output["pair_gate"], torch.ones_like(output["pair_gate"]))
    elif gate_type == "edge":
        torch.testing.assert_close(
            output["pair_gate"],
            output["edge_gate"][:, None, None].expand_as(output["pair_gate"]))


def test_compatibility_mixture_is_logsumexp_not_energy_expectation():
    relation_energy = torch.tensor([[[[0.0, 4.0]]]])
    relation_prob = torch.tensor([[[[0.5, 0.5]]]])
    compatibility = trajectory_relation_mixture_compatibility(
        relation_energy, relation_prob)
    expected = torch.log(
        0.5 * torch.exp(torch.tensor(0.0)) +
        0.5 * torch.exp(torch.tensor(-4.0)))
    torch.testing.assert_close(compatibility.squeeze(), expected)
    assert not torch.isclose(compatibility.squeeze(), torch.tensor(-2.0))


def test_direct_mlp_energy_ablation_has_same_contract_and_gradients():
    inputs = _bank_inputs()
    trajectories, last_pos, edge_index, edge_feat, weight, prior, scene = inputs
    trajectory_feat = TrajectoryEncoder(output_dim=9)(trajectories, last_pos)
    geometry = build_trajectory_pair_geometry(
        trajectories, last_pos, edge_index, scene)
    posterior = prior[:, None, None, :].expand(-1, 4, 4, -1)
    energy = RelationConditionedTrajectoryPairEnergy(
        trajectory_dim=9, edge_dim=EDGE_FEATURE_DIM,
        num_relation_modes=4, rank=3, hidden_dim=14,
        energy_type="mlp", gate_type="pair")
    output = energy(
        trajectory_feat, edge_index, edge_feat, posterior, geometry,
        weight, scene)

    assert output["left_factor"].shape == (2, 4, 4, 3)
    assert output["right_factor"].shape == (2, 4, 4, 3)
    assert output["relation_energy"].shape == (2, 4, 4, 4)
    assert output["pair_score"].shape == (2, 4, 4)
    output["pair_score"].square().mean().backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in energy.direct_compatibility_head.parameters())
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in energy.pair_gate_head.parameters())


@pytest.mark.parametrize("energy_type", ["lowrank", "mlp"])
def test_pair_relation_and_energy_are_equivariant_to_agent_reindexing(
        energy_type):
    torch.manual_seed(77)
    observations = torch.tensor([
        [[0.0, 0.0], [0.4, 0.0], [0.8, 0.0]],
        [[0.0, 1.0], [0.4, 1.0], [0.8, 1.0]],
    ])
    num_candidates, num_steps = 3, 5
    time = torch.arange(1, num_steps + 1).float()[None, None, :, None]
    candidate_velocity = torch.tensor([
        [[0.20, 0.00], [0.15, 0.05], [0.10, -0.08]],
        [[0.18, 0.00], [0.11, -0.07], [0.16, 0.06]],
    ])
    trajectories = (
        observations[:, -1, None, None, :] +
        candidate_velocity[:, :, None, :] * time)
    encoder = TrajectoryEncoder(output_dim=11).eval()
    relation = CandidateConditionedRelation(
        trajectory_dim=11, edge_dim=EDGE_FEATURE_DIM,
        num_relation_modes=3, hidden_dim=15).eval()
    energy = RelationConditionedTrajectoryPairEnergy(
        trajectory_dim=11, edge_dim=EDGE_FEATURE_DIM,
        num_relation_modes=3, rank=4, hidden_dim=15,
        gate_type="pair", energy_type=energy_type).eval()
    relation_prior = torch.tensor([[0.2, 0.5, 0.3]])

    def run(obs, bank):
        edge_index, edge_feat, edge_weight = build_interaction_graph(
            obs, graph_type="full")
        trajectory_feat = encoder(bank, obs[:, -1])
        geometry = build_trajectory_pair_geometry(
            bank, obs[:, -1], edge_index)
        relation_output = relation(
            trajectory_feat, edge_index, edge_feat, relation_prior, geometry)
        energy_output = energy(
            trajectory_feat, edge_index, edge_feat,
            relation_output["relation_posterior"], geometry, edge_weight)
        return relation_output, energy_output

    original_relation, original_energy = run(observations, trajectories)
    reorder = torch.tensor([1, 0])
    reversed_relation, reversed_energy = run(
        observations[reorder], trajectories[reorder])
    torch.testing.assert_close(
        original_relation["relation_posterior"][0],
        reversed_relation["relation_posterior"][0].transpose(0, 1))
    torch.testing.assert_close(
        original_energy["pair_score"][0],
        reversed_energy["pair_score"][0].transpose(0, 1))
    torch.testing.assert_close(
        original_energy["pair_gate"][0],
        reversed_energy["pair_gate"][0].transpose(0, 1))


def test_end_to_end_task_gradient_reaches_encoder_relation_energy_and_gate():
    inputs = _bank_inputs(requires_grad=True)
    trajectories, last_pos, edge_index, edge_feat, weight, prior, scene = inputs
    encoder = TrajectoryEncoder(output_dim=13, hidden_dim=17)
    relation = CandidateConditionedRelation(
        trajectory_dim=13, edge_dim=EDGE_FEATURE_DIM,
        num_relation_modes=4, hidden_dim=19)
    energy = RelationConditionedTrajectoryPairEnergy(
        trajectory_dim=13, edge_dim=EDGE_FEATURE_DIM,
        num_relation_modes=4, rank=5, hidden_dim=19,
        gate_type="pair")
    trajectory_feat = encoder(trajectories, last_pos)
    geometry = build_trajectory_pair_geometry(
        trajectories, last_pos, edge_index, scene)
    relation_output = relation(
        trajectory_feat, edge_index, edge_feat, prior, geometry, scene)
    energy_output = energy(
        trajectory_feat, edge_index, edge_feat,
        relation_output["relation_posterior"], geometry, weight, scene)

    loss = (
        energy_output["pair_score"].square().mean() +
        0.1 * relation_output["relation_prior_kl"])
    loss.backward()
    modules = (
        encoder.step_projection,
        encoder.temporal_encoder,
        relation.residual_mlp,
        energy.factor_head,
        energy.pair_gate_head,
    )
    for module in modules:
        gradients = [
            parameter.grad for parameter in module.parameters()
            if parameter.requires_grad]
        assert any(
            gradient is not None and torch.isfinite(gradient).all() and
            gradient.abs().sum() > 0 for gradient in gradients)
    assert trajectories.grad is not None
    assert torch.isfinite(trajectories.grad).all()


def test_empty_edges_and_nonfinite_input_are_safe():
    torch.manual_seed(9)
    trajectories = torch.randn(1, 3, 5, 2)
    last_pos = torch.randn(1, 2)
    edge_index = torch.empty((2, 0), dtype=torch.long)
    edge_feat = torch.empty((0, EDGE_FEATURE_DIM))
    prior = torch.empty((0, 4))
    encoder = TrajectoryEncoder(output_dim=8)
    trajectory_feat = encoder(trajectories, last_pos)
    geometry = build_trajectory_pair_geometry(
        trajectories, last_pos, edge_index, scene_index=torch.tensor([42]))
    relation = CandidateConditionedRelation(
        trajectory_dim=8, edge_dim=EDGE_FEATURE_DIM,
        num_relation_modes=4, hidden_dim=12)
    relation_output = relation(
        trajectory_feat, edge_index, edge_feat, prior, geometry,
        scene_index=torch.tensor([42]))
    energy = RelationConditionedTrajectoryPairEnergy(
        trajectory_dim=8, edge_dim=EDGE_FEATURE_DIM,
        num_relation_modes=4, rank=2, hidden_dim=12)
    energy_output = energy(
        trajectory_feat, edge_index, edge_feat,
        relation_output["relation_posterior"], geometry,
        scene_index=torch.tensor([42]))

    assert geometry.shape == (0, 3, 3, TRAJECTORY_PAIR_GEOMETRY_DIM)
    assert relation_output["relation_posterior"].shape == (0, 3, 3, 4)
    assert torch.isfinite(relation_output["relation_prior_kl"])
    assert energy_output["pair_score"].shape == (0, 3, 3)
    assert all(torch.isfinite(value).all() for value in energy_output.values())

    invalid = trajectories.clone()
    invalid[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        build_trajectory_pair_geometry(invalid, last_pos, edge_index)
