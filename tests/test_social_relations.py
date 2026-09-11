import torch

from src.models.interaction_graph import build_interaction_graph
from src.models.relation_inference import RelationInference
from src.models.social_encoder import SocialMotionEncoder


def _tracks():
    return torch.tensor(
        [
            [[-2.0, 0.0], [-1.0, 0.0], [0.0, 0.0]],
            [[-2.0, 1.0], [-1.0, 1.0], [0.0, 1.0]],
            [[4.0, -2.0], [4.0, -1.0], [4.0, 0.0]],
            [[9.0, 9.0], [9.5, 9.0], [10.0, 9.0]],
        ]
    )


def test_social_encoder_shapes_for_sparse_and_zero_edge_graphs():
    torch.manual_seed(3)
    obs = _tracks()
    edge_index, edge_feat, edge_weight = build_interaction_graph(
        obs, graph_type="radius-only", radius=1.5
    )
    encoder = SocialMotionEncoder(hidden_dim=16, output_dim=11, dropout=0.0)
    encoded = encoder(obs, edge_index, edge_feat, edge_weight)
    assert encoded.shape == (4, 11)
    assert torch.isfinite(encoded).all()

    one_obs = obs[:1]
    no_edges = build_interaction_graph(one_obs)
    one_encoded = encoder(one_obs, *no_edges)
    assert one_encoded.shape == (1, 11)
    assert torch.isfinite(one_encoded).all()


def test_social_encoder_is_permutation_equivariant_with_scene_groups():
    torch.manual_seed(11)
    obs = _tracks()
    scene_index = torch.tensor([0, 0, 0, 1])
    graph = build_interaction_graph(
        obs, scene_index=scene_index, graph_type="full"
    )
    encoder = SocialMotionEncoder(hidden_dim=12, output_dim=9, dropout=0.0)
    encoder.eval()
    original = encoder(obs, *graph, scene_index=scene_index)

    permutation = torch.tensor([2, 0, 3, 1])
    permuted_obs = obs[permutation]
    permuted_scene = scene_index[permutation]
    permuted_graph = build_interaction_graph(
        permuted_obs, scene_index=permuted_scene, graph_type="full"
    )
    permuted_output = encoder(
        permuted_obs, *permuted_graph, scene_index=permuted_scene
    )

    assert torch.allclose(permuted_output, original[permutation], atol=1e-6)


def test_soft_relation_probabilities_are_normalized_and_finite():
    torch.manual_seed(5)
    obs = _tracks()[:3]
    edge_index, edge_feat, edge_weight = build_interaction_graph(
        obs, graph_type="full"
    )
    encoder = SocialMotionEncoder(hidden_dim=10, dropout=0.0)
    agent_feat = encoder(obs, edge_index, edge_feat, edge_weight)
    relation = RelationInference(agent_dim=10, num_relation_modes=5, hidden_dim=17)

    relation_prob, logits = relation(
        agent_feat, edge_index, edge_feat, return_logits=True
    )
    assert relation_prob.shape == (3, 5)
    assert logits.shape == (3, 5)
    assert torch.isfinite(relation_prob).all()
    assert torch.allclose(relation_prob.sum(dim=-1), torch.ones(3), atol=1e-6)
    assert (relation_prob > 0).all()


def test_relation_distribution_is_invariant_to_pair_reindexing():
    torch.manual_seed(17)
    obs = _tracks()[:2]
    edge_index, edge_feat, _ = build_interaction_graph(obs, graph_type="full")
    agent_feat = torch.randn(2, 7)
    relation = RelationInference(agent_dim=7, num_relation_modes=4, hidden_dim=13)
    original = relation(agent_feat, edge_index, edge_feat)

    flipped_obs = obs.flip(0)
    flipped_index, flipped_feat, _ = build_interaction_graph(
        flipped_obs, graph_type="full"
    )
    flipped = relation(agent_feat.flip(0), flipped_index, flipped_feat)
    assert torch.allclose(original, flipped, atol=1e-6)


def test_hard_relation_uses_straight_through_gradients():
    torch.manual_seed(23)
    obs = _tracks()[:3]
    edge_index, edge_feat, _ = build_interaction_graph(obs, graph_type="full")
    agent_feat = torch.randn(3, 8, requires_grad=True)
    relation = RelationInference(agent_dim=8, num_relation_modes=4, hidden_dim=16)
    relation_prob = relation(agent_feat, edge_index, edge_feat, hard=True)

    assert torch.equal(relation_prob.sum(dim=-1), torch.ones(3))
    assert torch.logical_or(relation_prob == 0, relation_prob == 1).all()
    weights = torch.arange(4, dtype=relation_prob.dtype)
    (relation_prob * weights).sum().backward()
    assert agent_feat.grad is not None
    assert torch.isfinite(agent_feat.grad).all()
    assert agent_feat.grad.abs().sum() > 0


def test_relation_inference_handles_no_edges():
    relation = RelationInference(agent_dim=6, num_relation_modes=4)
    agent_feat = torch.randn(1, 6)
    edge_index, edge_feat, _ = build_interaction_graph(torch.zeros(1, 2, 2))
    relation_prob, logits = relation(
        agent_feat, edge_index, edge_feat, return_logits=True
    )
    assert relation_prob.shape == (0, 4)
    assert logits.shape == (0, 4)
