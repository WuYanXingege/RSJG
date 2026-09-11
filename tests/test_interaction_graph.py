import pytest
import torch

from src.models.interaction_graph import (
    EDGE_FEATURE_DIM,
    SparseInteractionGraph,
    build_interaction_graph,
    reverse_edge_features,
)


def test_single_agent_and_empty_agent_inputs_have_no_edges():
    for obs in (torch.zeros(1, 3, 2), torch.zeros(0, 3, 2)):
        edge_index, edge_feat, edge_weight = build_interaction_graph(obs)
        assert edge_index.shape == (2, 0)
        assert edge_index.dtype == torch.long
        assert edge_feat.shape == (0, EDGE_FEATURE_DIM)
        assert edge_weight.shape == (0,)


def test_radius_graph_emits_one_canonical_pair_with_oriented_features():
    obs = torch.tensor(
        [
            [[0.0, 0.0], [1.0, 0.0]],
            [[0.0, 1.0], [1.0, 1.0]],
        ]
    )
    edge_index, edge_feat, edge_weight = build_interaction_graph(
        obs, graph_type="radius-only", radius=1.1
    )

    assert torch.equal(edge_index, torch.tensor([[0], [1]]))
    assert edge_feat.shape == (1, EDGE_FEATURE_DIM)
    # Relative position follows target - source for the canonical 0 < 1 edge.
    assert torch.allclose(edge_feat[0, :2], torch.tensor([0.0, 1.0]))
    assert torch.allclose(edge_feat[0, 2:4], torch.zeros(2))
    assert edge_feat[0, 5].item() == pytest.approx(1.0)
    assert torch.equal(edge_weight, torch.ones(1))
    assert torch.isfinite(edge_feat).all()


def test_full_graph_respects_scene_boundaries_and_has_no_duplicate_pairs():
    obs = torch.zeros(5, 2, 2)
    scene_index = torch.tensor([0, 0, 0, 1, 1])
    edge_index, _, _ = build_interaction_graph(
        obs, scene_index=scene_index, graph_type="full"
    )

    # C(3,2) + C(2,2) canonical pairs.
    assert edge_index.shape == (2, 4)
    assert edge_index[0].lt(edge_index[1]).all()
    assert scene_index[edge_index[0]].eq(scene_index[edge_index[1]]).all()
    assert len(set(map(tuple, edge_index.t().tolist()))) == 4


def test_radius_ttc_adds_an_imminent_encounter_beyond_current_radius():
    approaching = torch.tensor(
        [
            [[-1.0, 0.0], [0.0, 0.0]],
            [[5.0, 0.0], [4.0, 0.0]],
        ]
    )
    radius_edges, _, _ = build_interaction_graph(
        approaching, graph_type="radius-only", radius=1.0
    )
    ttc_edges, ttc_feat, _ = build_interaction_graph(
        approaching,
        graph_type="radius+TTC",
        radius=1.0,
        ttc_threshold=3.0,
        dt=1.0,
    )

    assert radius_edges.shape == (2, 0)
    assert torch.equal(ttc_edges, torch.tensor([[0], [1]]))
    assert ttc_feat[0, 7].item() == pytest.approx(2.0)
    assert ttc_feat[0, 8].item() == pytest.approx(1.0)
    assert ttc_feat[0, 9].item() == pytest.approx(0.0)


def test_radius_ttc_rejects_distant_receding_pair():
    receding = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 0.0]],
            [[3.0, 0.0], [4.0, 0.0]],
        ]
    )
    edge_index, edge_feat, _ = build_interaction_graph(
        receding,
        graph_type="radius_ttc",
        radius=1.0,
        ttc_threshold=10.0,
    )
    assert edge_index.shape == (2, 0)
    assert edge_feat.shape == (0, EDGE_FEATURE_DIM)


def test_reverse_edge_features_is_an_involution():
    edge_feat = torch.randn(4, EDGE_FEATURE_DIM)
    reversed_twice = reverse_edge_features(reverse_edge_features(edge_feat))
    assert torch.equal(reversed_twice, edge_feat)


def test_invalid_shapes_and_nonfinite_observations_are_rejected():
    with pytest.raises(ValueError, match="shape"):
        build_interaction_graph(torch.zeros(2, 3))
    bad = torch.zeros(2, 3, 2)
    bad[0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="NaN"):
        build_interaction_graph(bad)


def test_adaptive_graph_learns_symmetric_finite_gates_and_prunes():
    torch.manual_seed(31)
    obs = torch.randn(5, 4, 2)
    graph = SparseInteractionGraph(
        graph_type="full",
        adaptive=True,
        gate_hidden_dim=12,
        top_k=1,
        gate_floor=0.05,
    )

    edge_index, edge_feat, edge_weight = graph(obs)

    assert edge_index.shape[1] <= obs.shape[0]
    assert edge_feat.shape == (edge_index.shape[1], EDGE_FEATURE_DIM)
    assert torch.isfinite(edge_weight).all()
    assert torch.all((edge_weight >= 0.05) & (edge_weight <= 1.0))
    assert graph.last_gate_prob.shape == (10,)  # full C(5,2) envelope
    assert graph.last_diagnostics["active_edges"] == edge_index.shape[1]
    graph.last_gate_prob.mean().backward()
    gradients = [
        parameter.grad for parameter in graph.gate_mlp.parameters()
        if parameter.grad is not None]
    assert gradients and all(torch.isfinite(value).all() for value in gradients)


def test_adaptive_gate_is_invariant_to_edge_orientation():
    torch.manual_seed(37)
    graph = SparseInteractionGraph(adaptive=True, gate_hidden_dim=8)
    features = torch.randn(7, EDGE_FEATURE_DIM)
    forward = graph._symmetric_gate(features)
    reverse = graph._symmetric_gate(reverse_edge_features(features))
    torch.testing.assert_close(forward, reverse, atol=1e-7, rtol=1e-7)


def test_disabled_adaptive_graph_has_no_checkpoint_parameters():
    graph = SparseInteractionGraph(adaptive=False)
    assert graph.state_dict() == {}
