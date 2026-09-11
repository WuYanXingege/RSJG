import math

import pytest
import torch

from src.multiway_coupling_loss import (
    aggregate_size_buckets,
    alignment_headroom_diagnostics,
    alignment_losses,
    assert_marginal_preservation,
    compute_multiway_coupling_loss,
    connected_component_labels,
    expected_joint_slot_costs,
    gate_regularization_loss,
    gather_by_permutation,
    hard_alignment_diagnostics,
    marginal_preservation_diagnostics,
    pair_assignment_loss,
    pair_oracle_costs,
    pair_score_loss,
    pair_target_distribution,
    permutation_entropy_loss,
    relation_prior_kl_loss,
    relative_assignment_matrices,
    softmin_cost,
    trajectory_gt_costs,
)


def test_masked_per_trajectory_cost_uses_last_valid_timestep_exactly():
    raw = torch.tensor([[[[1.0, 0.0], [2.0, 0.0], [100.0, 0.0]],
                         [[0.0, 0.0], [4.0, 0.0], [100.0, 0.0]]]])
    target = torch.zeros(1, 3, 2)
    mask = torch.tensor([[True, True, False]])

    cost, details = trajectory_gt_costs(
        raw, target, mask, lambda_fde=1.0, return_details=True)

    # ADEs are (1+2)/2 and (0+4)/2; FDE is at t=1, never at masked t=2.
    torch.testing.assert_close(details["ade"], torch.tensor([[1.5, 2.0]]))
    torch.testing.assert_close(details["fde"], torch.tensor([[2.0, 4.0]]))
    torch.testing.assert_close(cost, torch.tensor([[3.5, 6.0]]))
    assert details["last_valid_index"].item() == 1


def test_all_false_future_mask_is_finite_zero_and_marks_agent_invalid():
    raw = torch.randn(2, 3, 4, 2)
    target = torch.randn(2, 4, 2)
    cost, details = trajectory_gt_costs(
        raw, target, torch.zeros(2, 4, dtype=torch.bool),
        return_details=True)
    assert torch.equal(cost, torch.zeros_like(cost))
    assert not details["valid_agent_mask"].any()
    assert torch.isfinite(cost).all()


def test_expected_joint_cost_respects_candidate_row_joint_slot_column():
    cost = torch.tensor([[1.0, 4.0], [2.0, 8.0]])
    identity = torch.eye(2)
    swap = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    permutation = torch.stack((identity, swap))

    expected = expected_joint_slot_costs(permutation, cost)
    torch.testing.assert_close(expected, torch.tensor([[4.5, 3.0]]))

    # Packed scenes remain isolated rather than being averaged together.
    isolated = expected_joint_slot_costs(
        permutation, cost, scene_index=torch.tensor([10, 20]))
    torch.testing.assert_close(isolated, torch.tensor([[1.0, 4.0], [8.0, 2.0]]))


def test_normalized_softmin_and_no_harm_have_exact_values():
    values = torch.tensor([[2.0, 2.0], [1.0, 3.0]])
    result = softmin_cost(values, temperature=0.5)
    assert result[0].item() == pytest.approx(2.0)
    expected_second = -0.5 * torch.logsumexp(
        -values[1] / 0.5, dim=0) + 0.5 * math.log(2)
    assert result[1].item() == pytest.approx(expected_second.item())

    cost = torch.tensor([[0.0, 10.0], [0.0, 10.0]])
    # Uniform DS assignment destroys the good identity joint columns.
    uniform = torch.full((2, 2, 2), 0.5, requires_grad=True)
    losses = alignment_losses(uniform, cost, temperature=0.25)
    assert losses["loss_no_harm"].item() > 0
    losses["loss_alignment"].backward()
    assert uniform.grad is not None and torch.isfinite(uniform.grad).all()


def test_pair_oracle_combines_individual_path_and_endpoint_terms_exactly():
    raw = torch.tensor([
        [[[0.0, 0.0], [0.0, 0.0]],
         [[1.0, 0.0], [2.0, 0.0]]],
        [[[0.0, 0.0], [0.0, 0.0]],
         [[-1.0, 0.0], [-2.0, 0.0]]],
    ])
    target = torch.zeros(2, 2, 2)
    edges = torch.tensor([[0], [1]])
    individual_cost = torch.tensor([[2.0, 4.0], [6.0, 8.0]])

    pair_cost, details = pair_oracle_costs(
        raw, target, edges, trajectory_cost=individual_cost,
        lambda_rel_geom=2.0, lambda_rel_end=3.0, return_details=True)

    expected_individual = torch.tensor([[[4.0, 5.0], [5.0, 6.0]]])
    expected_path = torch.tensor([[[0.0, 1.5], [1.5, 3.0]]])
    expected_endpoint = torch.tensor([[[0.0, 2.0], [2.0, 4.0]]])
    torch.testing.assert_close(details["individual_cost"], expected_individual)
    torch.testing.assert_close(details["relative_path_error"], expected_path)
    torch.testing.assert_close(
        details["relative_endpoint_error"], expected_endpoint)
    torch.testing.assert_close(
        pair_cost,
        expected_individual + 2.0 * expected_path + 3.0 * expected_endpoint)


@pytest.mark.parametrize("loss_type", ["kl", "cross_entropy"])
def test_pair_score_soft_distribution_loss_is_finite_and_differentiable(loss_type):
    oracle = torch.tensor([[[0.0, 1.0], [2.0, 3.0]]])
    # These logits produce exactly the target distribution for tau_target=.5.
    score = (-2.0 * oracle).clone().requires_grad_()
    loss = pair_score_loss(
        score, oracle, target_temperature=0.5,
        prediction_temperature=1.0, loss_type=loss_type)
    assert torch.isfinite(loss)
    if loss_type == "kl":
        assert loss.item() == pytest.approx(0.0, abs=1e-6)
    loss.backward()
    assert score.grad is not None and torch.isfinite(score.grad).all()


def test_relative_assignment_uses_p_source_times_p_target_transpose():
    source = torch.tensor([[0.0, 1.0, 0.0],
                           [0.0, 0.0, 1.0],
                           [1.0, 0.0, 0.0]])
    target = torch.tensor([[0.0, 0.0, 1.0],
                           [1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0]])
    permutations = torch.stack((source, target))
    edges = torch.tensor([[0], [1]])

    relative = relative_assignment_matrices(permutations, edges)
    torch.testing.assert_close(relative[0], source @ target.T)
    assert not torch.equal(relative[0], source.T @ target)


def test_pair_assignment_loss_uses_global_pair_normalization_and_has_gradient():
    logits = torch.randn(2, 3, 3, requires_grad=True)
    # Column-softmax is enough for this loss-level gradient check; the real
    # synchronizer supplies doubly-stochastic matrices.
    permutations = torch.softmax(logits, dim=1)
    edge = torch.tensor([[0], [1]])
    target = torch.full((1, 3, 3), 1.0 / 9.0)
    loss, details = pair_assignment_loss(
        permutations, edge, target_prob=target, return_details=True)
    torch.testing.assert_close(
        details["relative_assignment_prob"].sum(dim=(1, 2)), torch.ones(1))
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


def test_permutation_entropy_has_expected_identity_and_uniform_values():
    identity = torch.eye(4).unsqueeze(0)
    uniform = torch.full((1, 4, 4), 0.25)
    assert permutation_entropy_loss(identity).item() == pytest.approx(0.0)
    assert permutation_entropy_loss(uniform).item() == pytest.approx(math.log(4))
    assert permutation_entropy_loss(
        uniform, normalize_by_log_k=True).item() == pytest.approx(1.0)


def test_relation_posterior_prior_kl_is_q_to_q0_and_differentiable():
    posterior = torch.tensor([[[[0.75, 0.25]]]], requires_grad=True)
    prior = torch.tensor([[0.5, 0.5]])
    loss = relation_prior_kl_loss(posterior, prior)
    expected = 0.75 * math.log(1.5) + 0.25 * math.log(0.5)
    assert loss.item() == pytest.approx(expected)
    loss.backward()
    assert posterior.grad is not None and torch.isfinite(posterior.grad).all()


def test_empty_graph_singleton_and_all_masked_objective_are_finite():
    raw = torch.randn(1, 3, 4, 2)
    target = torch.randn(1, 4, 2)
    logits = torch.randn(1, 3, 3, requires_grad=True)
    permutation = torch.softmax(logits, dim=1)
    pair_scores = torch.empty(0, 3, 3, requires_grad=True)
    edges = torch.empty(2, 0, dtype=torch.long)

    losses = compute_multiway_coupling_loss(
        permutation, pair_scores, raw, target, edges,
        future_mask=torch.zeros(1, 4, dtype=torch.bool))
    for value in losses.values():
        assert value.ndim == 0 and torch.isfinite(value)
    assert all(value.item() == pytest.approx(0.0) for value in losses.values())
    losses["loss_total"].backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_full_objective_shapes_finiteness_and_gradients():
    torch.manual_seed(403)
    num_agents, num_samples, num_steps, num_modes = 3, 3, 4, 4
    raw = torch.randn(num_agents, num_samples, num_steps, 2)
    target = torch.randn(num_agents, num_steps, 2)
    permutation_logits = torch.randn(
        num_agents, num_samples, num_samples, requires_grad=True)
    permutations = torch.softmax(permutation_logits, dim=1)
    pair_scores = torch.randn(2, num_samples, num_samples, requires_grad=True)
    relation_logits = torch.randn(
        2, num_samples, num_samples, num_modes, requires_grad=True)
    relation_prior_logits = torch.randn(2, num_modes, requires_grad=True)
    gate_logits = torch.randn(2, num_samples, num_samples, requires_grad=True)
    relation_posterior = torch.softmax(relation_logits, dim=-1)
    relation_prior = torch.softmax(relation_prior_logits, dim=-1)
    pair_gate = torch.sigmoid(gate_logits)
    edges = torch.tensor([[0, 1], [1, 2]])

    losses = compute_multiway_coupling_loss(
        permutations, pair_scores, raw, target, edges,
        relation_posterior=relation_posterior,
        relation_prior=relation_prior, pair_gate=pair_gate,
        gate_regularization="target_sparsity", weight_gate_reg=0.01,
        return_details=True)
    for key in (
            "loss_total", "loss_alignment", "loss_pair_score",
            "loss_pair_assignment", "loss_no_harm", "loss_perm_entropy",
            "loss_relation_prior", "loss_gate_reg"):
        assert losses[key].shape == () and torch.isfinite(losses[key])
    assert losses["trajectory_cost"].shape == (3, 3)
    assert losses["pair_oracle_cost"].shape == (2, 3, 3)
    losses["loss_total"].backward()
    for parameter in (
            permutation_logits, pair_scores, relation_logits,
            relation_prior_logits, gate_logits):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_hard_headroom_oracle_alignment_and_recovery_are_exact():
    cost = torch.tensor([[0.0, 10.0], [10.0, 0.0]])
    permutation = torch.tensor([[0, 1], [1, 0]])
    result = alignment_headroom_diagnostics(cost, permutation)

    assert result["alignment_oracle"].item() == pytest.approx(0.0)
    assert result["raw_joint_cost"].item() == pytest.approx(5.0)
    assert result["aligned_joint_cost"].item() == pytest.approx(0.0)
    assert result["alignment_headroom"].item() == pytest.approx(5.0)
    assert result["delta_joint"].item() == pytest.approx(5.0)
    assert result["alignment_recovery_ratio"].item() == pytest.approx(1.0)


def test_hard_gather_preserves_marginals_and_builds_n_component_buckets():
    raw = torch.tensor([
        [[[[0.0, 0.0]]], [[[10.0, 0.0]]]],
        [[[[10.0, 0.0]]], [[[0.0, 0.0]]]],
    ]).reshape(2, 2, 1, 2)
    target = torch.zeros(2, 1, 2)
    permutation = torch.tensor([[0, 1], [1, 0]])
    aligned = gather_by_permutation(raw, permutation)

    marginal = assert_marginal_preservation(raw, aligned, target)
    assert marginal["marginal_preserved"]
    result = hard_alignment_diagnostics(
        raw, target, permutation, edge_index=torch.tensor([[0], [1]]),
        lambda_fde=0.0)
    assert result["Raw_JADE"].item() == pytest.approx(5.0)
    assert result["Aligned_JADE"].item() == pytest.approx(0.0)
    assert result["delta_JADE"].item() == pytest.approx(5.0)
    assert result["N_size_buckets"]["N=2"]["count"] == 1
    component = result["component_size_buckets"]["comp_size=2"]
    assert component["count"] == 1
    assert component["recovery_ratio"] == pytest.approx(1.0)


def test_marginal_check_detects_non_permutation_coordinate_change():
    raw = torch.tensor([[[[0.0, 0.0]], [[2.0, 0.0]]]])
    target = torch.zeros(1, 1, 2)
    changed = raw + 1.0
    result = marginal_preservation_diagnostics(raw, changed, target)
    assert not result["marginal_preserved"]
    with pytest.raises(AssertionError, match="changed marginal metrics"):
        assert_marginal_preservation(raw, changed, target)


def test_components_and_bucket_aggregation_are_scene_safe():
    edges = torch.tensor([[0, 2], [1, 3]])
    scene = torch.tensor([10, 10, 20, 20, 20])
    labels, sizes = connected_component_labels(edges, 5, scene)
    assert torch.equal(sizes, torch.tensor([2, 2, 2, 2, 1]))
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[1] != labels[2]

    buckets = aggregate_size_buckets(
        {"JADE": torch.tensor([1.0, 3.0, 9.0])},
        torch.tensor([1, 2, 7]), "component")
    assert buckets["comp_size=1"] == {"count": 1, "JADE": 1.0}
    assert buckets["comp_size=2"] == {"count": 1, "JADE": 3.0}
    assert buckets["comp_size=6~10"] == {"count": 1, "JADE": 9.0}

    with pytest.raises(ValueError, match="different scenes"):
        connected_component_labels(
            torch.tensor([[1], [2]]), 5, scene)


def test_empty_edge_losses_and_gate_regularizer_remain_finite():
    empty_score = torch.empty(0, 2, 2, requires_grad=True)
    empty_cost = torch.empty(0, 2, 2)
    target = pair_target_distribution(empty_cost)
    score_loss = pair_score_loss(empty_score, target_prob=target)
    assignment = pair_assignment_loss(
        torch.eye(2).expand(1, -1, -1),
        torch.empty(2, 0, dtype=torch.long), target_prob=target)
    gate_loss = gate_regularization_loss(
        torch.empty(0, 2, 2, requires_grad=True), "l1")
    for value in (score_loss, assignment, gate_loss):
        assert value.shape == () and torch.isfinite(value)
