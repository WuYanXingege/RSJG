import itertools

import pytest
import torch

import src.models.joint_dependency_v2.joint_sampler as sampler_module
from tools.tie_semantics_localization_audit import (
    TieLocalizationController,
    _match_world_sets,
    deterministic_selection,
    graph_class,
    graph_components,
    permutation_localization_records,
    row_degeneracy_records,
)


def _score(agents=3, slots=3, candidates=4, device="cpu"):
    score = torch.tensor([
        [8.0, 2.0, 1.0, 0.0],
        [0.0, 7.0, 2.0, 1.0],
        [1.0, 0.0, 6.0, 2.0],
    ], device=device)[:slots, :candidates]
    score = score[None].expand(agents, -1, -1).clone()
    mask = torch.ones_like(score, dtype=torch.bool)
    return score, mask


def test_graph_components_include_isolates_and_classify_mixed_graphs():
    edge_index = torch.tensor([[0, 1], [1, 0]])
    components = graph_components(4, edge_index)
    assert components == ((0, 1), (2,), (3,))
    assert graph_class(components) == "multiple_components_with_singleton"

    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
    components = graph_components(4, edge_index)
    assert components == ((0, 1), (2, 3))
    assert graph_class(components) == "multiple_components_all_size_ge2"


def test_d0_degree_zero_is_identity_and_degree_positive_matches_policy_d():
    score, mask = _score()
    previous = torch.tensor([[3, 1, 0], [2, 0, 3], [1, 3, 2]])
    degree = torch.tensor([1, 1, 0])
    policy_d = deterministic_selection(
        "D_deterministic", score, mask, previous, degree)
    d0 = deterministic_selection(
        "D0_degree0_identity", score, mask, previous, degree)
    assert torch.equal(d0[:2], policy_d[:2])
    assert torch.equal(d0[2], previous[2])


def test_d0_common_slot_permutation_permutates_previous_ids_too():
    score, mask = _score()
    previous = torch.tensor([[0, 1, 2], [2, 0, 1], [3, 2, 1]])
    degree = torch.tensor([1, 1, 0])
    original = deterministic_selection(
        "D0_degree0_identity", score, mask, previous, degree)
    permutation = torch.tensor([2, 0, 1])
    permuted = deterministic_selection(
        "D0_degree0_identity", score[:, permutation], mask[:, permutation],
        previous[:, permutation], degree)
    restored = torch.empty_like(permuted)
    restored[:, permutation] = permuted
    assert torch.equal(restored, original)


def test_isolated_agent_synthetic_permutation_is_pathwise_invariant():
    score, mask = _score(agents=3)
    previous = torch.tensor([[0, 1, 2], [2, 0, 1], [3, 2, 1]])
    degree = torch.tensor([1, 1, 0])
    round1 = deterministic_selection(
        "D0_degree0_identity", score, mask, previous, degree)
    round2 = deterministic_selection(
        "D0_degree0_identity", score, mask, round1, degree)
    trace = [
        {"score": score, "mask": mask, "candidate_index": previous},
        {"score": score, "mask": mask, "candidate_index": round1},
        {"score": score, "mask": mask, "candidate_index": round2},
    ]
    edge_index = torch.tensor([[0, 1], [1, 0]])
    agents, scenes, components = permutation_localization_records(
        "D0_degree0_identity", trace, edge_index, 2035, 0)
    isolated = [row for row in agents if row["degree_bin"] == "degree=0"]
    assert isolated
    assert all(row["pathwise_invariant"] for row in isolated)
    assert all(row["slot_change_rate"] == 0 for row in isolated)
    singleton = [row for row in components if row["is_singleton"]]
    assert singleton
    assert all(row["exact_world_set_equal"] for row in singleton)
    assert all(row["matched_mean_normalized_hamming"] == 0 for row in singleton)
    assert scenes


def test_component_world_matching_is_order_invariant_and_detects_change():
    left = torch.tensor([[0, 1], [2, 3], [4, 5]])
    equal = _match_world_sets(left, left[[2, 0, 1]])
    assert equal["exact_world_set_equal"]
    assert equal["matched_exact_world_fraction"] == 1
    changed = left.clone()
    changed[0, 0] = 9
    unequal = _match_world_sets(left, changed)
    assert not unequal["exact_world_set_equal"]
    assert unequal["matched_mean_normalized_hamming"] > 0


def test_degree_zero_rows_are_exactly_degenerate_in_mixed_graph():
    score, mask = _score(agents=3)
    score[1, 1, 0] += 0.25
    score[2] = score[2, :1].expand_as(score[2])
    ids = torch.tensor([[0, 1, 2], [0, 1, 2], [2, 1, 0]])
    trace = [
        {"score": score, "mask": mask, "candidate_index": ids},
        {"score": score, "mask": mask, "candidate_index": ids},
        {"score": score, "mask": mask, "candidate_index": ids},
    ]
    rows = row_degeneracy_records(
        "D0_degree0_identity", trace,
        torch.tensor([[0, 1], [1, 0]]), 2035, 0)
    isolated = [row for row in rows if row["degree_bin"] == "degree=0"]
    assert isolated
    assert all(row["exact_slot_invariant_rows"] for row in isolated)
    interacting = [row for row in rows if row["agent"] == 1]
    assert all(not row["exact_slot_invariant_rows"] for row in interacting)


def test_d0_refinement_does_not_consume_generator_and_restores_patch():
    sampler = sampler_module.ParallelConditionalSampler(strict_no_z=True)
    controller = TieLocalizationController(
        sampler, "D0_degree0_identity", {}, {}, use_cuda=False)
    controller.degree = torch.tensor([1, 0])
    score, mask = _score(agents=2)
    previous = torch.tensor([[2, 1, 0], [3, 2, 1]])
    controller.trace = [{
        "score": score,
        "mask": mask,
        "candidate_index": previous,
    }]
    original = sampler_module.structured_gumbel_assignment
    with controller:
        generator = torch.Generator().manual_seed(2035)
        before = generator.get_state().clone()
        selected = sampler_module.structured_gumbel_assignment(
            score, mask, 1.0, generator)
        after = generator.get_state()
    assert sampler_module.structured_gumbel_assignment is original
    assert torch.equal(before, after)
    assert torch.equal(selected[1], previous[1])
    assert controller.boundary_checks["round_generator_unchanged"] == 1


def test_assignment_remains_synchronous_agent_local():
    score, mask = _score(agents=3)
    previous = torch.tensor([[0, 1, 2], [2, 0, 1], [3, 2, 1]])
    degree = torch.tensor([1, 1, 0])
    original = deterministic_selection(
        "D0_degree0_identity", score, mask, previous, degree)
    changed = score.clone()
    changed[0] = changed[0].flip(-1)
    updated = deterministic_selection(
        "D0_degree0_identity", changed, mask, previous, degree)
    assert not torch.equal(original[0], updated[0])
    assert torch.equal(original[1:], updated[1:])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_d0_bf16_cuda_audit_path():
    score, mask = _score(agents=3, device="cuda")
    score = score.to(torch.bfloat16)
    previous = torch.tensor(
        [[0, 1, 2], [2, 0, 1], [3, 2, 1]], device="cuda")
    degree = torch.tensor([1, 1, 0], device="cuda")
    selected = deterministic_selection(
        "D0_degree0_identity", score, mask, previous, degree)
    assert selected.is_cuda
    assert torch.equal(selected[2], previous[2])
    assert all(torch.unique(row).numel() == row.numel()
               for row in selected)


def test_tiny_assignment_reference_remains_globally_optimal():
    score, mask = _score(agents=1)
    selected = deterministic_selection(
        "D_deterministic", score, mask,
        torch.zeros((1, 3), dtype=torch.long), torch.ones(1))[0]
    actual = sum(float(score[0, row, selected[row]]) for row in range(3))
    expected = max(
        sum(float(score[0, row, column])
            for row, column in enumerate(columns))
        for columns in itertools.permutations(range(4), 3))
    assert actual == pytest.approx(expected)
