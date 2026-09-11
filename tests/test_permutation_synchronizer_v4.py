import pytest
import torch

from src.models.hungarian_projection import (
    project_permutations,
    project_to_permutation,
    validate_bijection,
)
from src.models.permutation_synchronizer import (
    MultiwayPermutationSynchronizer,
    log_sinkhorn,
)


def _assignment(permutation):
    """Build P[candidate,slot] from perm[slot]=candidate."""
    permutation = torch.as_tensor(permutation, dtype=torch.long)
    return torch.nn.functional.one_hot(
        permutation, num_classes=permutation.numel()
    ).transpose(0, 1).float()


def _strong_synchronizer(**kwargs):
    defaults = dict(
        sync_iterations=4,
        sinkhorn_iterations=20,
        tau_sync=0.2,
        alpha_msg=1.0,
        alpha_prev=0.0,
        alpha_identity=0.0,
        use_inference_edge_topk=False,
        use_soft_confidence_mixing=False,
    )
    defaults.update(kwargs)
    return MultiwayPermutationSynchronizer(**defaults)


def test_n1_empty_graph_is_exact_identity_with_finite_diagnostics():
    synchronizer = MultiwayPermutationSynchronizer()
    output = synchronizer(
        torch.empty(0, 3, 3),
        torch.empty(2, 0, dtype=torch.long),
        num_agents=1,
    )
    torch.testing.assert_close(output["P"], torch.eye(3)[None], rtol=0, atol=0)
    assert output["component_ids"].tolist() == [0]
    assert output["component_sizes"].tolist() == [1]
    assert output["component_anchors"].tolist() == [0]
    assert output["confidence"].tolist() == [0.0]
    for key in (
        "sinkhorn_row_error",
        "sinkhorn_col_error",
        "permutation_entropy",
        "cycle_consistency_error",
    ):
        assert torch.isfinite(output[key])


def test_hungarian_k3_swap_uses_candidate_slot_convention_and_batches():
    swap = _assignment([1, 0, 2])
    cycle = _assignment([2, 0, 1])
    probability = torch.stack((0.1 + 5.0 * swap, 0.2 + 4.0 * cycle))
    result = project_permutations(probability)

    assert result["permutation"].tolist() == [[1, 0, 2], [2, 0, 1]]
    torch.testing.assert_close(
        result["hard_assignment"], torch.stack((swap, cycle)), rtol=0, atol=0
    )
    validate_bijection(result["permutation"])
    assert result["metadata"]["actual_method"] == "hungarian"
    assert result["metadata"]["fallback_used"] is False
    assert "candidate,slot" in result["metadata"]["convention"]

    permutation, metadata = project_to_permutation(
        probability[0], return_metadata=True
    )
    assert permutation.tolist() == [1, 0, 2]
    assert metadata["batch_size"] == 1


def test_inference_returns_hard_bijection_and_explicit_bypass_metadata():
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    score = _assignment([1, 0, 2])[None] * 10.0
    synchronizer = _strong_synchronizer(
        use_identity_bypass=False,
        hard_projection="hungarian",
    ).eval()
    output = synchronizer(score, edge_index, num_agents=2)
    assert output["hard_permutation"].tolist() == [[0, 1, 2], [1, 0, 2]]
    validate_bijection(output["hard_permutation"])
    assert output["hard_assignment"].shape == (2, 3, 3)
    assert output["projection_metadata"]["actual_method"] == "hungarian"
    assert not output["identity_bypass"].any()


def test_three_agent_chain_recovers_global_multiway_assignments():
    # The degree-max anchor is agent 1, which fixes the gauge at identity.  No
    # endpoint has a direct edge to the other: both graph edges are required.
    expected = torch.stack(
        (
            _assignment([1, 2, 0]),
            torch.eye(3),
            _assignment([2, 0, 1]),
        )
    )
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    pair_score = torch.stack(
        (
            expected[0] @ expected[1].T,
            expected[1] @ expected[2].T,
        )
    ) * 10.0
    output = _strong_synchronizer()(
        pair_score,
        edge_index,
        torch.ones(2),
        torch.zeros(3, dtype=torch.long),
        num_agents=3,
        inference=False,
    )
    hard = project_permutations(output["P"])["permutation"]
    expected_perm = torch.tensor([[1, 2, 0], [0, 1, 2], [2, 0, 1]])
    assert torch.equal(hard, expected_perm)
    assert output["component_anchors"].tolist() == [1]
    assert output["component_sizes"].tolist() == [3]
    assert output["sinkhorn_row_error_max"] < 1e-5
    assert output["sinkhorn_col_error_max"] < 1e-5


def test_triangle_is_cycle_consistent_under_one_global_assignment():
    expected = torch.stack(
        (
            torch.eye(3),
            _assignment([1, 2, 0]),
            _assignment([2, 0, 1]),
        )
    )
    edge_index = torch.tensor([[0, 0, 1], [1, 2, 2]], dtype=torch.long)
    source, target = edge_index
    pair_score = torch.stack(
        [expected[i] @ expected[j].T for i, j in zip(source, target)]
    ) * 10.0
    output = _strong_synchronizer(sync_iterations=5)(
        pair_score, edge_index, num_agents=3, inference=False
    )
    projected = project_permutations(output["P"])["hard_assignment"]
    q01 = projected[0] @ projected[1].T
    q12 = projected[1] @ projected[2].T
    q02 = projected[0] @ projected[2].T
    torch.testing.assert_close(q01 @ q12, q02, rtol=0, atol=0)
    assert output["triangle_count"].item() == 1
    assert output["cycle_consistency_error"] < 1e-5


def test_packed_scenes_are_isolated_and_cross_scene_edges_are_rejected():
    edge_index = torch.tensor([[0, 2], [1, 3]], dtype=torch.long)
    scene_index = torch.tensor([7, 7, 42, 42], dtype=torch.long)
    scene_a_score = _assignment([1, 0, 2]) * 8.0
    scene_b_score = _assignment([2, 0, 1]) * 8.0
    synchronizer = _strong_synchronizer(sync_iterations=2)

    first = synchronizer(
        torch.stack((scene_a_score, scene_b_score)),
        edge_index,
        scene_index=scene_index,
        num_agents=4,
        inference=False,
    )
    second = synchronizer(
        torch.stack((scene_a_score, scene_b_score.roll(1, dims=0))),
        edge_index,
        scene_index=scene_index,
        num_agents=4,
        inference=False,
    )
    torch.testing.assert_close(first["P"][:2], second["P"][:2], rtol=0, atol=0)
    assert first["component_sizes"].tolist() == [2, 2]
    assert first["component_scene_ids"].tolist() == [7, 42]
    assert first["component_ids"].tolist() == [0, 0, 1, 1]

    with pytest.raises(ValueError, match="cross scene"):
        synchronizer(
            torch.ones(1, 3, 3),
            torch.tensor([[0], [2]]),
            scene_index=scene_index,
            num_agents=4,
        )


def test_soft_sinkhorn_bijection_projection_and_gradients_are_finite():
    torch.manual_seed(1701)
    logits = torch.randn(5, 4, 4, requires_grad=True)
    soft = log_sinkhorn(logits, iterations=30, temperature=0.7)
    torch.testing.assert_close(
        soft.sum(dim=-1), torch.ones(5, 4), rtol=1e-5, atol=1e-5
    )
    torch.testing.assert_close(
        soft.sum(dim=-2), torch.ones(5, 4), rtol=1e-5, atol=1e-5
    )
    projected = project_permutations(soft.detach())["permutation"]
    validate_bijection(projected, 4)

    # A triangle gives non-anchor nodes multiple weighted messages, exercising
    # score, strength, iterative Sinkhorn and the learnable confidence path.
    pair_score = torch.randn(3, 4, 4, requires_grad=True)
    edge_strength = torch.tensor([0.4, 0.7, 0.9], requires_grad=True)
    edge_index = torch.tensor([[0, 0, 1], [1, 2, 2]], dtype=torch.long)
    synchronizer = MultiwayPermutationSynchronizer(
        sync_iterations=3,
        sinkhorn_iterations=12,
        tau_sync=0.7,
        alpha_msg=1.0,
        alpha_prev=0.05,
        alpha_identity=0.05,
    )
    output = synchronizer(
        pair_score,
        edge_index,
        edge_strength=edge_strength,
        scene_index=torch.zeros(3, dtype=torch.long),
        num_agents=3,
        inference=False,
    )
    objective = (
        output["P"].square().sum()
        + output["component_confidence"].sum()
        + output["cycle_consistency_error"]
        + soft.square().sum()
    )
    assert torch.isfinite(objective)
    objective.backward()
    assert pair_score.grad is not None and torch.isfinite(pair_score.grad).all()
    assert pair_score.grad.abs().sum() > 0
    assert edge_strength.grad is not None and torch.isfinite(edge_strength.grad).all()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    parameter_gradients = [
        parameter.grad
        for parameter in synchronizer.parameters()
        if parameter.grad is not None
    ]
    assert parameter_gradients
    assert all(torch.isfinite(gradient).all() for gradient in parameter_gradients)


def test_training_confidence_mixture_is_doubly_stochastic_and_learnable():
    score = _assignment([1, 0, 2])[None] * 8.0
    synchronizer = MultiwayPermutationSynchronizer(
        sync_iterations=2,
        sinkhorn_iterations=20,
        tau_sync=0.2,
        alpha_prev=0.0,
        alpha_identity=0.0,
        use_soft_confidence_mixing=True,
        use_inference_edge_topk=False,
    ).train()
    output = synchronizer(
        score,
        torch.tensor([[0], [1]], dtype=torch.long),
        num_agents=2,
    )
    identity = torch.eye(3)[None].expand(2, -1, -1)
    expected = (
        output["confidence"][:, None, None]
        * output["synchronized_permutation"]
        + (1.0 - output["confidence"][:, None, None]) * identity
    )
    torch.testing.assert_close(output["P"], expected)
    torch.testing.assert_close(output["P"].sum(-1), torch.ones(2, 3), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(output["P"].sum(-2), torch.ones(2, 3), atol=1e-6, rtol=1e-6)
    loss = -output["P"][1, 1, 0]
    loss.backward()
    assert synchronizer.keep_head[-1].bias.grad is not None
    assert synchronizer.keep_head[-1].bias.grad.abs().sum() > 0


def test_invalid_projection_is_detected_strictly():
    with pytest.raises(ValueError, match="not a bijection"):
        validate_bijection(torch.tensor([[0, 0, 2], [0, 1, 2]]), 3)
