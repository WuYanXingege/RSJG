import itertools
import random

import numpy as np
import pytest
import torch

import src.models.joint_dependency_v2.exact_lexicographic_assignment as exact
import src.models.joint_dependency_v2.joint_sampler as sampler_module
from src.models.interaction_graph import EDGE_FEATURE_DIM
from src.models.joint_dependency_v2 import (
    DynamicHypothesisRelation,
    RelationSpecificJointEnergy,
)
from src.models.joint_dependency_v2.exact_lexicographic_assignment import (
    exact_persistent_refinement,
    exchangeable_edge_priorities,
    solve_exact_persistent_tie,
    stable_tie_seed,
)
from src.models.joint_dependency_v2.joint_sampler import (
    ParallelConditionalSampler,
    make_sampling_generators,
    structured_gumbel_assignment,
    weighted_gumbel_top_p,
)
from src.parser import check_and_add_additional_args, get_parser
from tools.lexicographic_continuity_audit import (
    assignment_objective_tuple,
    solve_exact_lexicographic,
)
from tools.residual_equivalence_stochastic_audit import (
    random_objective,
    resolve_residual_equivalence,
)


def _problem(device="cpu", dtype=torch.float32, ambiguous=False):
    if ambiguous:
        score = torch.tensor(
            [[-1, -1, 0, 0], [-1, -1, 0, 0]],
            dtype=dtype, device=device)
        previous = torch.tensor([0, 1], device=device)
        goals = torch.tensor(
            [[0, 0], [0, 0], [1, 0], [-1, 0]],
            dtype=dtype, device=device)
    else:
        score = torch.tensor(
            [[3, 1, 0, -2], [0, 2, 1, -1], [1, 0, 4, -3]],
            dtype=dtype, device=device)
        previous = torch.tensor([2, 0, 1], device=device)
        goals = torch.tensor(
            [[0, 0], [1, 1], [2, 4], [3, 9]],
            dtype=dtype, device=device)
    return score, torch.ones_like(score, dtype=torch.bool), previous, goals


def _valid_assignments(mask):
    return [assignment for assignment in itertools.permutations(
        range(mask.shape[1]), mask.shape[0]) if all(
            bool(mask[row, column])
            for row, column in enumerate(assignment))]


def _objective(problem, assignment, priorities):
    deterministic, internals = solve_exact_lexicographic(*problem)
    del deterministic
    return (*assignment_objective_tuple(
        assignment, internals["primary"], internals["stay"],
        internals["geometry"]), random_objective(assignment, priorities))


def test_production_solver_matches_tiny_exhaustive_four_tuple():
    problem = _problem()
    priorities, _ = exchangeable_edge_priorities(problem[1], 17)
    actual = solve_exact_persistent_tie(*problem, priorities=priorities)
    assignments = _valid_assignments(problem[1])
    expected = max(assignments, key=lambda item: _objective(
        problem, item, priorities))
    assert actual.assignment == expected
    assert actual.objective == _objective(problem, expected, priorities)


def test_production_radix_matches_reference_tuple_for_all_assignments():
    problem = _problem(ambiguous=True)
    priorities, _ = exchangeable_edge_priorities(problem[1], 23)
    actual = solve_exact_persistent_tie(*problem, priorities=priorities)
    expected = max(_valid_assignments(problem[1]), key=lambda item: _objective(
        problem, item, priorities))
    assert actual.assignment == expected
    assert actual.objective == _objective(problem, expected, priorities)


def test_one_ulp_primary_gap_cannot_be_overridden_by_lower_objectives():
    one = torch.tensor(1.0, dtype=torch.float32)
    higher = torch.nextafter(one, torch.tensor(float("inf")))
    score = torch.tensor([[higher, 1.0], [1.0, 1.0]])
    mask = torch.ones_like(score, dtype=torch.bool)
    previous = torch.tensor([1, 0])
    goals = torch.tensor([[0.0, 0.0], [1000.0, 1000.0]])
    result = solve_exact_persistent_tie(
        score, mask, previous, goals, priorities=((1, 2), (4, 8)))
    assert result.assignment == (0, 1)


def test_stay_gap_cannot_be_overridden_by_geometry_or_r():
    score = torch.zeros((2, 3), dtype=torch.float32)
    mask = torch.ones_like(score, dtype=torch.bool)
    previous = torch.tensor([0, 1])
    goals = torch.tensor([[100.0, 0.0], [-100.0, 0.0], [0.0, 0.0]])
    priorities, _ = exchangeable_edge_priorities(mask, 29)
    result = solve_exact_persistent_tie(
        score, mask, previous, goals, priorities=priorities)
    assert result.assignment == (0, 1)
    assert result.objective[1] == 2


def test_geometry_gap_cannot_be_overridden_by_r():
    score = torch.tensor([[-1.0, -1.0, 0.0, 0.0],
                          [-1.0, -1.0, 0.0, 0.0]])
    mask = torch.ones_like(score, dtype=torch.bool)
    previous = torch.tensor([0, 1])
    goals = torch.tensor([[0.0, 0.0], [10.0, 0.0],
                          [1.0, 0.0], [9.0, 0.0]])
    priorities, _ = exchangeable_edge_priorities(mask, 31)
    result = solve_exact_persistent_tie(
        score, mask, previous, goals, priorities=priorities)
    assert result.assignment == (2, 3)


def test_distinct_assignment_has_distinct_power_sum():
    mask = torch.ones((3, 4), dtype=torch.bool)
    priorities, _ = exchangeable_edge_priorities(mask, 37)
    totals = [random_objective(item, priorities)
              for item in _valid_assignments(mask)]
    assert len(totals) == len(set(totals))


def test_same_production_key_replays_exactly():
    left = solve_exact_persistent_tie(
        *_problem(ambiguous=True), evaluation_seed=2035,
        window_index=9, agent_index=2)
    right = solve_exact_persistent_tie(
        *_problem(ambiguous=True), evaluation_seed=2035,
        window_index=9, agent_index=2)
    assert left == right


def test_r_payload_is_persistent_across_rounds():
    problem = _problem(ambiguous=True)
    first = solve_exact_persistent_tie(
        *problem, evaluation_seed=2035, window_index=4, agent_index=1)
    second = solve_exact_persistent_tie(
        *problem, evaluation_seed=2035, window_index=4, agent_index=1)
    assert first.priorities == second.priorities
    assert first.assignment == second.assignment


def test_unique_deterministic_triple_is_independent_of_r():
    problem = _problem()
    outputs = {
        solve_exact_persistent_tie(
            *problem, evaluation_seed=2035, window_index=0,
            agent_index=seed).assignment
        for seed in range(12)}
    assert len(outputs) == 1


def test_ambiguous_triple_matches_independent_audit_replica_zero():
    problem = _problem(ambiguous=True)
    seed = stable_tie_seed(2035, 7, 3)
    production = solve_exact_persistent_tie(
        *problem, evaluation_seed=2035, window_index=7, agent_index=3)
    reference = resolve_residual_equivalence(*problem, priority_seed=seed)
    assert production.assignment == reference["assignment"]
    assert production.objective[:3] == reference["objective"][:3]
    assert production.objective[3] == random_objective(
        reference["assignment"], production.priorities)


def test_degree_zero_is_tensor_exact_identity_and_skips_solver(monkeypatch):
    score, mask, previous, goals = _problem()
    called = []
    monkeypatch.setattr(exact, "solve_exact_persistent_tie",
                        lambda *args, **kwargs: called.append(True))
    actual = exact_persistent_refinement(
        score[None], mask[None], previous[None], goals[None],
        torch.tensor([0]), evaluation_seed=2035, window_index=0)
    assert torch.equal(actual[0], previous)
    assert called == []


def _transform(problem, priorities, slot=None, candidate=None):
    score, mask, previous, goals = problem
    rows, columns = score.shape
    slot = torch.arange(rows) if slot is None else slot
    candidate = torch.arange(columns) if candidate is None else candidate
    inverse_candidate = torch.empty_like(candidate)
    inverse_candidate[candidate] = torch.arange(columns)
    transformed_score = score[slot][:, candidate]
    transformed_mask = mask[slot][:, candidate]
    transformed_previous = inverse_candidate[previous[slot]]
    transformed_goals = goals[candidate]
    transformed_priorities = tuple(tuple(
        priorities[int(old_row)][int(old_column)]
        for old_column in candidate.tolist())
        for old_row in slot.tolist())
    result = solve_exact_persistent_tie(
        transformed_score, transformed_mask, transformed_previous,
        transformed_goals, priorities=transformed_priorities)
    restored = [None] * rows
    for new_row, old_row in enumerate(slot.tolist()):
        restored[old_row] = int(candidate[result.assignment[new_row]])
    return result, tuple(restored)


@pytest.mark.parametrize("kind", ["slot", "candidate", "combined"])
def test_frozen_r_slot_and_candidate_pathwise_equivariance(kind):
    problem = _problem(ambiguous=True)
    priorities, _ = exchangeable_edge_priorities(problem[1], 41)
    original = solve_exact_persistent_tie(*problem, priorities=priorities)
    slot = torch.tensor([1, 0]) if kind != "candidate" else None
    candidate = torch.tensor([2, 0, 3, 1]) if kind != "slot" else None
    transformed, restored = _transform(
        problem, priorities, slot=slot, candidate=candidate)
    assert restored == original.assignment
    assert transformed.objective == original.objective


def test_frozen_r_agent_and_combined_pathwise_equivariance():
    problems = [_problem(ambiguous=True), _problem()]
    payloads = [exchangeable_edge_priorities(problem[1], 50 + index)[0]
                for index, problem in enumerate(problems)]
    original = [solve_exact_persistent_tie(
        *problem, priorities=payload).assignment
        for problem, payload in zip(problems, payloads)]
    permutation = [1, 0]
    relabeled = []
    for new_index, old_index in enumerate(permutation):
        problem = problems[old_index]
        payload = payloads[old_index]
        if new_index == 1:
            _, restored = _transform(
                problem, payload, slot=torch.tensor([1, 0]),
                candidate=torch.tensor([2, 0, 3, 1]))
            relabeled.append(restored)
        else:
            relabeled.append(solve_exact_persistent_tie(
                *problem, priorities=payload).assignment)
    restored_agents = [None, None]
    for new_index, old_index in enumerate(permutation):
        restored_agents[old_index] = relabeled[new_index]
    assert restored_agents == original


@pytest.mark.parametrize("kind", ["p_gt_k", "valid_k_lt_p", "slot_mask"])
def test_contract_rejects_invalid_mask_or_capacity(kind):
    if kind == "p_gt_k":
        score = torch.zeros((3, 2))
        previous = torch.tensor([0, 1, 0])
        goals = torch.zeros((2, 2))
        mask = torch.ones_like(score, dtype=torch.bool)
    else:
        score, mask, previous, goals = _problem(ambiguous=True)
        if kind == "valid_k_lt_p":
            mask[:, 1:] = False
        else:
            mask[1, 3] = False
    with pytest.raises(ValueError):
        solve_exact_persistent_tie(
            score, mask, previous, goals, evaluation_seed=1,
            window_index=2, agent_index=3)


def test_batch_update_is_synchronous_and_input_previous_is_unchanged():
    score, mask, previous, goals = _problem()
    batch_score = torch.stack((score, score.flip(0)))
    batch_mask = torch.stack((mask, mask))
    batch_previous = torch.stack((previous, previous.flip(0)))
    frozen = batch_previous.clone()
    batch_goals = torch.stack((goals, goals))
    actual = exact_persistent_refinement(
        batch_score, batch_mask, batch_previous, batch_goals,
        torch.tensor([1, 1]), evaluation_seed=2035, window_index=0)
    assert actual.shape == batch_previous.shape
    assert torch.equal(batch_previous, frozen)


def test_production_solver_uses_exactly_one_assignment_solve(monkeypatch):
    calls = []
    original = exact.exact_max_weight_assignment

    def wrapped(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(exact, "exact_max_weight_assignment", wrapped)
    solve_exact_persistent_tie(
        *_problem(ambiguous=True), evaluation_seed=2035,
        window_index=0, agent_index=0)
    assert len(calls) == 1


def test_local_priority_and_solver_consume_no_global_rng():
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()
    solve_exact_persistent_tie(
        *_problem(ambiguous=True), evaluation_seed=2035,
        window_index=0, agent_index=0)
    assert python_before == random.getstate()
    assert all(np.array_equal(left, right) if isinstance(left, np.ndarray)
               else left == right
               for left, right in zip(numpy_before, np.random.get_state()))
    assert torch.equal(torch_before, torch.get_rng_state())


def test_round_zero_stream_and_default_paths_remain_unchanged():
    score = torch.linspace(-1.0, 1.0, 21).view(1, 1, 21).expand(2, 20, 21)
    mask = torch.ones_like(score, dtype=torch.bool)
    expected = weighted_gumbel_top_p(
        score, mask, 1.0,
        make_sampling_generators(2035, 4, torch.device("cpu"))["initial"])
    actual = weighted_gumbel_top_p(
        score, mask, 1.0,
        make_sampling_generators(2035, 4, torch.device("cpu"))["initial"])
    assert torch.equal(actual, expected)
    assert ParallelConditionalSampler(strict_no_z=True).refinement_policy == \
        "categorical"
    cpsr_score = torch.randn((1, 3, 4), generator=torch.Generator().manual_seed(8))
    cpsr_mask = torch.ones_like(cpsr_score, dtype=torch.bool)
    left = structured_gumbel_assignment(
        cpsr_score, cpsr_mask, 1.0, torch.Generator().manual_seed(9))
    right = structured_gumbel_assignment(
        cpsr_score, cpsr_mask, 1.0, torch.Generator().manual_seed(9))
    assert torch.equal(left, right)


def test_parser_default_and_old_configuration_are_compatible():
    assert get_parser().parse_args([]).jdv2_refinement_policy == "categorical"
    args = get_parser().parse_args([
        "--device", "cpu", "--goal_model_type", "jdv2",
        "--training_stage", "joint_goal", "--data_augmentation", "False",
        "--use_scene_latent", "False", "--jdv2_latent_objective",
        "strict_no_z", "--jdv2_refinement_policy",
        "exact_lexicographic_persistent_tie",
    ])
    checked = check_and_add_additional_args(args)
    assert checked.jdv2_refinement_policy == \
        "exact_lexicographic_persistent_tie"


def test_reference_parity_on_random_fp32_matrices():
    for window in range(12):
        generator = torch.Generator().manual_seed(100 + window)
        score = torch.randn((4, 5), generator=generator)
        mask = torch.ones_like(score, dtype=torch.bool)
        previous = torch.randperm(5, generator=generator)[:4]
        goals = torch.randn((5, 2), generator=generator)
        problem = score, mask, previous, goals
        production = solve_exact_persistent_tie(
            *problem, evaluation_seed=2035, window_index=window,
            agent_index=2)
        reference = resolve_residual_equivalence(
            *problem, priority_seed=stable_tie_seed(2035, window, 2))
        assert production.assignment == reference["assignment"]
        assert production.objective[:3] == reference["objective"][:3]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_real_cuda_bf16_source_policy_execution_is_rng_isolated():
    score, mask, previous, goals = _problem("cuda", torch.bfloat16, True)
    before_cpu = torch.get_rng_state().clone()
    before_cuda = torch.cuda.get_rng_state_all()
    result = solve_exact_persistent_tie(
        score, mask, previous, goals, evaluation_seed=2035,
        window_index=0, agent_index=0)
    assert len(set(result.assignment)) == score.shape[0]
    assert torch.equal(before_cpu, torch.get_rng_state())
    assert all(torch.equal(left, right) for left, right in zip(
        before_cuda, torch.cuda.get_rng_state_all()))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_real_cuda_bf16_full_source_sampler_executes_two_rounds():
    device = torch.device("cuda")
    agents, candidates = 2, 21
    sampler = ParallelConditionalSampler(
        strict_no_z=True,
        refinement_policy="exact_lexicographic_persistent_tie").to(device)
    relation = DynamicHypothesisRelation(
        6.0, use_scene_latent=False).to(device).eval()
    energy = RelationSpecificJointEnergy(
        use_scene_latent=False).to(device).eval()
    edge_index = torch.tensor([[0], [1]], device=device)
    trace = []
    sampler.diagnostic_callback = lambda **row: trace.append(
        row["candidate_index"].detach().clone())
    sampler.set_sampling_context(2035, 0)
    unary = torch.randn((agents, candidates), device=device)
    goals = torch.randn((agents, candidates, 2), device=device)
    edge_feat = torch.randn((1, EDGE_FEATURE_DIM), device=device)
    agent_feat = torch.randn((agents, 128), device=device)
    last = torch.randn((agents, 2), device=device)
    base = torch.randn((1, 4), device=device)
    before_cpu = torch.get_rng_state().clone()
    before_cuda = torch.cuda.get_rng_state_all()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = sampler(
            unary, goals,
            None, torch.zeros(agents, dtype=torch.long, device=device),
            edge_index, edge_feat, agent_feat, last, base, relation, energy,
            sampling_mode="sample", use_scene_latent=False)
    assert output["candidate_index"].shape == (agents, 20)
    assert len(trace) == 3
    assert all(torch.unique(ids[agent]).numel() == 20
               for ids in trace for agent in range(agents))
    assert torch.equal(before_cpu, torch.get_rng_state())
    assert all(torch.equal(left, right) for left, right in zip(
        before_cuda, torch.cuda.get_rng_state_all()))
