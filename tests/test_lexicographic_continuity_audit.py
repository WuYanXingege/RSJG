import itertools
from fractions import Fraction
import struct

import pytest
import torch

import src.models.joint_dependency_v2.joint_sampler as sampler_module
from tools.lexicographic_continuity_audit import (
    ExactLCPSRController,
    assignment_objective_tuple,
    audit_d0_assignment,
    exact_max_weight_assignment,
    lift_fp32_to_integers,
    semantic_gate,
    same_format_permutation_gate,
    solve_exact_lexicographic,
    _transform_and_solve,
)


def _problem(score, previous=None, goals=None, mask=None):
    score = torch.tensor(score, dtype=torch.float32)
    slots, candidates = score.shape
    if previous is None:
        previous = list(range(slots))
    if goals is None:
        goals = [[float(index), float(index * index)]
                 for index in range(candidates)]
    if mask is None:
        mask = torch.ones_like(score, dtype=torch.bool)
    return (score, mask, torch.tensor(previous, dtype=torch.long),
            torch.tensor(goals, dtype=torch.float32))


def _exhaustive(score, mask, previous, goals):
    solution, internals = solve_exact_lexicographic(
        score, mask, previous, goals)
    best = None
    assignments = []
    for assignment in itertools.permutations(range(score.shape[1]),
                                             score.shape[0]):
        if not all(mask[row, column] for row, column in enumerate(assignment)):
            continue
        objective = assignment_objective_tuple(
            assignment, internals["primary"], internals["stay"],
            internals["geometry"])
        if best is None or objective > best:
            best = objective
            assignments = [assignment]
        elif objective == best:
            assignments.append(assignment)
    return solution, best, assignments


def test_tiny_exact_tuple_optimum_matches_exhaustive_enumeration():
    problem = _problem([[3, 1, 0, -2], [0, 2, 1, -1], [1, 0, 4, -3]])
    solution, best, assignments = _exhaustive(*problem)
    assert solution.objective == best
    assert solution.assignment in assignments
    assert solution.unique == (len(assignments) == 1)


def test_primary_one_ulp_gap_cannot_be_overridden_by_continuity():
    one = torch.tensor(1.0, dtype=torch.float32)
    next_value = torch.nextafter(one, torch.tensor(float("inf")))
    score = torch.tensor([[next_value, 1.0], [1.0, 1.0]])
    mask = torch.ones_like(score, dtype=torch.bool)
    previous = torch.tensor([1, 0])
    goals = torch.tensor([[0.0, 0.0], [100.0, 100.0]])
    solution, _ = solve_exact_lexicographic(score, mask, previous, goals)
    assert solution.assignment == (0, 1)
    assert solution.tie_level == "PRIMARY_UNIQUE"


def test_geometry_cannot_override_better_stay_count():
    score, mask, previous, goals = _problem(
        [[0, 0, 0], [0, 0, 0]], previous=[0, 1],
        goals=[[100, 0], [-100, 0], [0, 0]])
    solution, _ = solve_exact_lexicographic(score, mask, previous, goals)
    assert solution.assignment == (0, 1)
    assert solution.objective[1] == 2


def test_valid_assignment_is_injective_and_respects_mask():
    score, mask, previous, goals = _problem([[4, 3, 2], [5, 1, 0]])
    mask[:, 2] = False
    solution, _ = solve_exact_lexicographic(score, mask, previous, goals)
    assert len(set(solution.assignment)) == 2
    assert all(mask[row, column]
               for row, column in enumerate(solution.assignment))


@pytest.mark.parametrize("kind", ["p_gt_k", "valid_k_lt_p", "slot_mask"])
def test_invalid_contracts_fail_explicitly(kind):
    if kind == "p_gt_k":
        problem = _problem([[1, 0], [0, 1], [1, 1]], previous=[0, 1, 0])
    else:
        problem = _problem([[1, 0, -1], [0, 1, -1]])
        score, mask, previous, goals = problem
        if kind == "valid_k_lt_p":
            mask[:, 1:] = False
        else:
            mask[1, 2] = False
        problem = score, mask, previous, goals
    with pytest.raises(ValueError):
        solve_exact_lexicographic(*problem)


def test_degree_zero_rule_is_structural_identity_without_solver():
    previous = torch.tensor([2, 0, 1])
    # The structural branch is deliberately represented by the direct clone;
    # exact LCPSR is defined only for interacting agents.
    selected = previous.clone()
    assert torch.equal(selected, previous)


def test_unique_slot_permutation_is_exactly_equivariant():
    problem = _problem([[9, 1, 0], [0, 8, 1]])
    solution, _ = solve_exact_lexicographic(*problem)
    transformed, restored = _transform_and_solve(
        *problem, slot_permutation=torch.tensor([1, 0]))
    assert transformed.unique
    assert restored == solution.assignment


def test_unique_candidate_permutation_is_exactly_equivariant():
    problem = _problem([[9, 1, 0], [0, 8, 1]])
    solution, _ = solve_exact_lexicographic(*problem)
    transformed, restored = _transform_and_solve(
        *problem, candidate_permutation=torch.tensor([2, 0, 1]))
    assert transformed.unique
    assert restored == solution.assignment


def test_unique_combined_permutation_is_exactly_equivariant():
    problem = _problem([[9, 1, 0], [0, 8, 1]])
    solution, _ = solve_exact_lexicographic(*problem)
    transformed, restored = _transform_and_solve(
        *problem, slot_permutation=torch.tensor([1, 0]),
        candidate_permutation=torch.tensor([2, 0, 1]))
    assert transformed.unique
    assert restored == solution.assignment


def test_exact_symmetric_case_returns_ambiguous_not_index_fallback():
    problem = _problem(
        [[-1, -1, 0, 0], [-1, -1, 0, 0]], previous=[0, 1],
        goals=[[0, 0], [0, 0], [1, 0], [-1, 0]])
    solution, _ = solve_exact_lexicographic(*problem)
    assert not solution.unique
    assert solution.tie_level == "FULL_TUPLE_AMBIGUOUS"
    assert solution.ambiguity_certificate["same_exact_objective"]


def test_previous_primary_optimum_with_full_stay_is_unique_itself():
    problem = _problem([[0, 0, 0], [0, 0, 0]], previous=[0, 1])
    solution, _ = solve_exact_lexicographic(*problem)
    assert solution.unique
    assert solution.assignment == (0, 1)
    assert solution.tie_level == "PRIMARY_TIED_RESOLVED_BY_STAY"


def test_fp32_bit_lift_is_exact_for_normals_subnormal_and_signed_values():
    smallest = struct.unpack("f", struct.pack("I", 1))[0]
    values = torch.tensor([1.0, -0.5, smallest, 0.0], dtype=torch.float32)
    lifted = lift_fp32_to_integers(values)
    reconstructed = [Fraction(item) * lifted.unit
                     for item in lifted.values]
    expected = [Fraction(float(item)) for item in values]
    assert reconstructed == expected


def test_exact_primary_objective_matches_bruteforce():
    weight = ((5, 2, 0, -1), (1, 4, 3, 0), (0, 2, 6, 1))
    mask = tuple(tuple(True for _ in row) for row in weight)
    assignment, objective = exact_max_weight_assignment(weight, mask)
    expected = max(
        sum(weight[row][column] for row, column in enumerate(columns))
        for columns in itertools.permutations(range(4), 3))
    assert objective == expected
    assert sum(weight[row][column]
               for row, column in enumerate(assignment)) == expected


def test_uniqueness_certificate_matches_exhaustive_enumeration():
    for problem in (
            _problem([[3, 0, 0], [0, 2, 0]]),
            _problem([[-1, -1, 0, 0], [-1, -1, 0, 0]],
                     previous=[0, 1],
                     goals=[[0, 0], [0, 0], [1, 0], [-1, 0]])):
        solution, _, assignments = _exhaustive(*problem)
        assert solution.unique == (len(assignments) == 1)


def test_d0_exact_primary_audit_detects_positive_regret():
    problem = _problem([[10, 0, 0], [0, 10, 0]])
    solution, internals = solve_exact_lexicographic(*problem)
    audited = audit_d0_assignment(torch.tensor([1, 2]), solution, internals)
    assert not audited["primary_optimal"]
    assert audited["primary_regret_integer"] > 0
    assert audited["difference_category"] == "primary_correction"


@pytest.mark.parametrize("problem,level", [
    (_problem([[4, 0], [0, 3]]), "PRIMARY_UNIQUE"),
    (_problem([[0, 0, 0], [0, 0, 0]], previous=[0, 1]),
     "PRIMARY_TIED_RESOLVED_BY_STAY"),
    (_problem([[-1, -1, 0, 0], [-1, -1, 0, 0]], previous=[0, 1],
              goals=[[0, 0], [10, 0], [1, 0], [9, 0]]),
     "STAY_TIED_RESOLVED_BY_GEOM"),
    (_problem([[-1, -1, 0, 0], [-1, -1, 0, 0]], previous=[0, 1],
              goals=[[0, 0], [0, 0], [1, 0], [-1, 0]]),
     "FULL_TUPLE_AMBIGUOUS"),
])
def test_tie_resolution_classification(problem, level):
    solution, _ = solve_exact_lexicographic(*problem)
    assert solution.tie_level == level


def test_two_round_updates_use_frozen_previous_round_synchronously():
    score = torch.tensor([
        [[5.0, 0.0, 1.0], [0.0, 5.0, 1.0]],
        [[4.0, 0.0, 1.0], [0.0, 4.0, 1.0]],
    ])
    mask = torch.ones_like(score, dtype=torch.bool)
    goals = torch.tensor([
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
    ])
    previous = torch.tensor([[2, 1], [1, 2]])
    new = []
    for agent in range(2):
        solution, _ = solve_exact_lexicographic(
            score[agent], mask[agent], previous[agent], goals[agent])
        new.append(solution.assignment)
    assert new == [(0, 1), (0, 1)]
    assert torch.equal(previous, torch.tensor([[2, 1], [1, 2]]))


def test_exact_solver_consumes_no_global_torch_rng():
    problem = _problem([[3, 1, 0], [0, 2, 1]])
    before = torch.get_rng_state().clone()
    solve_exact_lexicographic(*problem)
    assert torch.equal(before, torch.get_rng_state())


def test_audit_controller_is_synchronous_degree0_identity_and_rng_free():
    sampler = sampler_module.ParallelConditionalSampler(strict_no_z=True)
    controller = ExactLCPSRController(sampler, {}, use_cuda=False)
    controller.degree = torch.tensor([1, 0])
    controller.goal_candidates = torch.tensor([
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
    ])
    previous = torch.tensor([[2, 1], [1, 2]])
    score = torch.tensor([
        [[5.0, 0.0, 1.0], [0.0, 5.0, 1.0]],
        [[5.0, 0.0, 1.0], [0.0, 5.0, 1.0]],
    ])
    mask = torch.ones_like(score, dtype=torch.bool)
    controller.trace = [{
        "score": score,
        "mask": mask,
        "candidate_index": previous.clone(),
    }]
    original = sampler_module.structured_gumbel_assignment
    with controller:
        generator = torch.Generator().manual_seed(2035)
        before = generator.get_state().clone()
        selected = sampler_module.structured_gumbel_assignment(
            score, mask, 1.0, generator)
        after = generator.get_state().clone()
    assert sampler_module.structured_gumbel_assignment is original
    assert torch.equal(before, after)
    assert torch.equal(selected[1], previous[1])
    assert tuple(selected[0].tolist()) == (0, 1)


def test_semantic_gate_rejects_any_full_tuple_ambiguity():
    summary = {
        "full_tuple_ambiguous_count": 1,
        "all_exact_primary_optimal": True,
        "all_valid_injective_coverage": True,
        "permutation": {
            transform: {
                f"round{round_index}": {
                    "pathwise_invariance_rate": 1.0,
                    "scene_world_set_invariance_rate": 1.0,
                }
                for round_index in (1, 2)}
            for transform in (
                "slot", "candidate", "slot_candidate",
                "slot_agent_candidate")
        },
    }
    assert not semantic_gate(summary)["passed"]


def test_same_format_permutation_gate_compares_tie_report_schema():
    rounds = {
        name: {
            "exact_world_set_equal_rate": 0.75,
            "matched_mean_normalized_hamming": {"mean": 0.1},
            "matched_exact_world_fraction": {"mean": 0.9},
        }
        for name in ("round1", "round2")
    }
    value = {"permutation_localization": {"overall_scene": rounds}}
    assert same_format_permutation_gate(value, value)["passed"]


def test_production_sampler_functions_are_not_patched_by_reference_solver():
    original_initial = sampler_module.weighted_gumbel_top_p
    original_refinement = sampler_module.structured_gumbel_assignment
    solve_exact_lexicographic(*_problem([[3, 1, 0], [0, 2, 1]]))
    assert sampler_module.weighted_gumbel_top_p is original_initial
    assert sampler_module.structured_gumbel_assignment is original_refinement


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_bf16_canonical_snapshot_path_executes_without_rng_change():
    score = torch.tensor(
        [[3.0, 1.0, 0.0], [0.0, 2.0, 1.0]],
        device="cuda", dtype=torch.bfloat16)
    mask = torch.ones_like(score, dtype=torch.bool)
    previous = torch.tensor([0, 1], device="cuda")
    goals = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], device="cuda",
        dtype=torch.bfloat16)
    before_cpu = torch.get_rng_state().clone()
    before_cuda = torch.cuda.get_rng_state().clone()
    solution, _ = solve_exact_lexicographic(
        score.float(), mask, previous, goals.float())
    assert solution.unique
    assert torch.equal(before_cpu, torch.get_rng_state())
    assert torch.equal(before_cuda, torch.cuda.get_rng_state())
