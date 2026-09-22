import itertools
import random
from collections import defaultdict

import numpy as np
import pytest
import torch

import src.models.joint_dependency_v2.joint_sampler as sampler_module
from tools.lexicographic_continuity_audit import (
    assignment_objective_tuple,
    solve_exact_lexicographic,
)
from tools.residual_equivalence_stochastic_audit import (
    ResidualAuditController,
    accumulate_window_metric,
    exchangeable_edge_priorities,
    extend_exact_full_weight,
    random_objective,
    resolve_residual_equivalence,
    stable_tie_seed,
    transform_residual_problem,
)


def _ambiguous(device="cpu", dtype=torch.float32):
    score = torch.tensor(
        [[-1, -1, 0, 0], [-1, -1, 0, 0]],
        dtype=dtype, device=device)
    mask = torch.ones_like(score, dtype=torch.bool)
    previous = torch.tensor([0, 1], device=device)
    goals = torch.tensor(
        [[0, 0], [0, 0], [1, 0], [-1, 0]],
        dtype=dtype, device=device)
    return score, mask, previous, goals


def _unique():
    score = torch.tensor([[9.0, 1.0, 0.0], [0.0, 8.0, 1.0]])
    return (score, torch.ones_like(score, dtype=torch.bool),
            torch.tensor([2, 0]),
            torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]))


def _all_assignments(mask):
    return [assignment for assignment in itertools.permutations(
        range(mask.shape[1]), mask.shape[0]) if all(
            mask[row, column] for row, column in enumerate(assignment))]


def test_unique_deterministic_case_is_tensor_exact_and_does_not_invoke_r():
    deterministic, _ = solve_exact_lexicographic(*_unique())
    result = resolve_residual_equivalence(*_unique(), priority_seed=1)
    assert result["assignment"] == deterministic.assignment
    assert result["objective"][:3] == deterministic.objective
    assert not result["r_invoked"]


def test_full_ambiguous_case_invokes_r_and_preserves_exact_triple():
    deterministic, _ = solve_exact_lexicographic(*_ambiguous())
    result = resolve_residual_equivalence(
        *_ambiguous(), priority_seed=stable_tie_seed(2035, 0, 0, 0),
        certify_extended=True)
    assert result["r_invoked"]
    assert result["objective"][:3] == deterministic.objective
    assert result["extended_unique"]


def test_distinct_power_priorities_make_all_assignment_sums_unique():
    mask = torch.ones((3, 4), dtype=torch.bool)
    priorities, count = exchangeable_edge_priorities(mask, 17)
    values = [random_objective(assignment, priorities)
              for assignment in _all_assignments(mask)]
    assert count == 12
    assert len(values) == len(set(values))


def test_radix_encoding_exactly_matches_lexicographic_order_exhaustively():
    mask = torch.ones((2, 3), dtype=torch.bool)
    weight = ((7, 7, 1), (7, 7, 1))
    priorities, count = exchangeable_edge_priorities(mask, 23)
    extended, radix = extend_exact_full_weight(weight, priorities, count)
    assignments = _all_assignments(mask)
    by_tuple = max(assignments, key=lambda assignment: (
        sum(weight[row][column] for row, column in enumerate(assignment)),
        random_objective(assignment, priorities)))
    by_scalar = max(assignments, key=lambda assignment: sum(
        extended[row][column] for row, column in enumerate(assignment)))
    assert by_scalar == by_tuple
    assert radix == 1 << count


def test_same_four_key_reproduces_priority_and_assignment():
    seed = stable_tie_seed(2035, 4, 2, 1)
    left = resolve_residual_equivalence(*_ambiguous(), priority_seed=seed)
    right = resolve_residual_equivalence(*_ambiguous(), priority_seed=seed)
    assert left["priorities"] == right["priorities"]
    assert left["assignment"] == right["assignment"]


def test_different_tie_replicas_can_choose_different_representatives():
    representatives = {
        resolve_residual_equivalence(
            *_ambiguous(),
            priority_seed=stable_tie_seed(2035, 4, replica, 1))["assignment"]
        for replica in range(32)
    }
    assert len(representatives) > 1


def test_r_seed_is_persistent_because_round_is_not_in_key():
    assert stable_tie_seed(2035, 9, 2, 3) == \
        stable_tie_seed(2035, 9, 2, 3)


@pytest.mark.parametrize("transform", ["slot", "candidate", "combined"])
def test_conditional_pathwise_equivariance_when_r_is_permuted(transform):
    score, mask, previous, goals = _ambiguous()
    seed = stable_tie_seed(2035, 0, 0, 0)
    priorities, _ = exchangeable_edge_priorities(mask, seed)
    original = resolve_residual_equivalence(
        score, mask, previous, goals, priorities=priorities)
    slots = torch.tensor([1, 0])
    candidates = torch.tensor([2, 0, 3, 1])
    transformed, restored = transform_residual_problem(
        score, mask, previous, goals, priorities,
        slot_permutation=slots if transform != "candidate" else None,
        candidate_permutation=(candidates if transform != "slot" else None))
    assert restored == original["assignment"]
    assert transformed["objective"] == original["objective"]


def test_agent_relabeling_is_equivariant_when_priority_payload_travels():
    problems = [_ambiguous(), _unique()]
    outputs = []
    for agent, problem in enumerate(problems):
        outputs.append(resolve_residual_equivalence(
            *problem, priority_seed=stable_tie_seed(2035, 0, 0, agent))[
                "assignment"])
    permutation = [1, 0]
    relabeled = [outputs[index] for index in permutation]
    restored = [None, None]
    for new_index, old_index in enumerate(permutation):
        restored[old_index] = relabeled[new_index]
    assert restored == outputs


def test_distribution_changes_only_inside_full_equivalence_class():
    unique = {
        resolve_residual_equivalence(*_unique(), priority_seed=seed)[
            "assignment"] for seed in range(16)}
    ambiguous = {
        resolve_residual_equivalence(*_ambiguous(), priority_seed=seed)[
            "assignment"] for seed in range(16)}
    assert len(unique) == 1
    assert len(ambiguous) > 1


def test_local_r_generation_does_not_touch_global_python_numpy_or_torch_rng():
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()
    exchangeable_edge_priorities(torch.ones((2, 4), dtype=torch.bool), 99)
    assert python_before == random.getstate()
    assert all(np.array_equal(left, right) if isinstance(left, np.ndarray)
               else left == right
               for left, right in zip(numpy_before, np.random.get_state()))
    assert torch.equal(torch_before, torch.get_rng_state())


def test_metric_accumulator_preserves_native_scene_or_agent_cardinality():
    metrics = defaultdict(lambda: defaultdict(list))
    counts = defaultdict(lambda: defaultdict(list))
    accumulate_window_metric(
        metrics, counts, "E>0", "N=3-4", "scene_metric", [0.5])
    accumulate_window_metric(
        metrics, counts, "E>0", "N=3-4", "agent_metric",
        [0.1, 0.2, 0.3])
    assert metrics["overall"]["scene_metric"] == [0.5]
    assert metrics["overall"]["agent_metric"] == [0.1, 0.2, 0.3]
    assert counts["N=3-4"]["scene_metric"] == [0.5]


def test_invalid_or_non_power_priority_is_rejected():
    priorities = ((1, 2, 3, 4), (8, 16, 32, 64))
    with pytest.raises(ValueError, match="distinct powers"):
        resolve_residual_equivalence(
            *_ambiguous(), priorities=priorities)


def test_masked_candidate_is_never_selected():
    score, mask, previous, goals = _ambiguous()
    mask[:, 3] = False
    result = resolve_residual_equivalence(
        score, mask, previous, goals, priority_seed=4)
    assert 3 not in result["assignment"]


def test_controller_degree_zero_identity_and_round_rng_unchanged():
    sampler = sampler_module.ParallelConditionalSampler(strict_no_z=True)
    controller = ResidualAuditController(
        sampler, "persistent_exchangeable_exact_tie", {}, use_cuda=False)
    controller.key = (2035, 0)
    controller.degree = torch.tensor([1, 0])
    score, mask, previous, goals = _ambiguous()
    controller.goal_candidates = torch.stack((goals, goals))
    scores = torch.stack((score, score))
    masks = torch.stack((mask, mask))
    old = torch.stack((previous, previous.flip(0)))
    controller.trace = [{
        "score": scores, "mask": masks, "candidate_index": old.clone()}]
    original = sampler_module.structured_gumbel_assignment
    with controller:
        generator = torch.Generator().manual_seed(2035)
        state = generator.get_state().clone()
        selected = sampler_module.structured_gumbel_assignment(
            scores, masks, 1.0, generator)
        assert torch.equal(state, generator.get_state())
    assert sampler_module.structured_gumbel_assignment is original
    assert torch.equal(selected[1], old[1])
    assert torch.equal(old, controller.trace[0]["candidate_index"])


def test_controller_priority_cache_is_reused_across_rounds():
    sampler = sampler_module.ParallelConditionalSampler(strict_no_z=True)
    controller = ResidualAuditController(
        sampler, "persistent_exchangeable_exact_tie", {}, use_cuda=False)
    controller.key = (2035, 0)
    mask = torch.ones((2, 4), dtype=torch.bool)
    first = controller._priority(0, mask)
    second = controller._priority(0, mask.clone())
    assert first is second
    assert controller.boundary_checks["tie_state_created"] == 1


def test_fourth_objective_never_changes_the_deterministic_tuple_exhaustive():
    deterministic, internals = solve_exact_lexicographic(*_ambiguous())
    for seed in range(20):
        result = resolve_residual_equivalence(
            *_ambiguous(), priority_seed=seed)
        assert result["objective"][:3] == deterministic.objective
        assert assignment_objective_tuple(
            result["assignment"], internals["primary"], internals["stay"],
            internals["geometry"]) == deterministic.objective


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_bf16_cuda_inputs_resolve_on_audit_cpu_without_rng_leak():
    score, mask, previous, goals = _ambiguous("cuda", torch.bfloat16)
    before = torch.cuda.get_rng_state().clone()
    result = resolve_residual_equivalence(
        score, mask, previous, goals, priority_seed=2035)
    assert len(set(result["assignment"])) == score.shape[0]
    assert torch.equal(before, torch.cuda.get_rng_state())
