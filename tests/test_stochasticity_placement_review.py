import torch

import src.models.joint_dependency_v2.joint_sampler as sampler_module
from src.models.joint_dependency_v2.joint_sampler import (
    ParallelConditionalSampler,
    make_sampling_generators,
    maximum_weight_injective_assignment,
)
from tools.sampler_coverage_intervention import _initial_generator
from tools.stochasticity_placement_review import (
    PairedStructuredController,
    real_permutation_records,
    summarize_cross_seed_support,
    synthetic_tie_audit,
)


def _initial_score(agents=2, slots=3, candidates=4):
    base = torch.linspace(-1.0, 1.0, candidates)
    score = base.view(1, 1, -1).expand(agents, slots, -1).clone()
    return score, torch.ones_like(score, dtype=torch.bool)


def _conditional_score(agents=2, slots=3, candidates=4):
    base = torch.tensor([
        [9.0, 1.0, 0.0, -1.0],
        [0.0, 8.0, 1.0, -1.0],
        [0.0, 1.0, 7.0, -1.0],
    ])[:slots, :candidates]
    score = base[None].expand(agents, -1, -1).clone()
    return score, torch.ones_like(score, dtype=torch.bool)


def test_production_and_historical_initial_seed_derivations_differ():
    production = make_sampling_generators(
        2035, 7, torch.device("cpu"))["initial"]
    historical = _initial_generator(2035, 7, torch.device("cpu"))
    assert not torch.equal(
        torch.rand(16, generator=production),
        torch.rand(16, generator=historical))


def test_paired_controller_shares_round0_and_deterministic_round_is_noise_free():
    sampler = ParallelConditionalSampler(strict_no_z=True)
    key = (2035, 4)
    shared = {}
    initial_score, initial_mask = _initial_score()
    conditional, conditional_mask = _conditional_score()

    fresh = PairedStructuredController(
        sampler, "shared_round0_fresh_gumbel", {key: {}}, shared,
        use_cuda=False)
    fresh.key = key
    with fresh:
        generators = make_sampling_generators(
            key[0], key[1], torch.device("cpu"))
        fresh_ids = sampler_module.weighted_gumbel_top_p(
            initial_score, initial_mask, 1.0, generators["initial"])
    assert torch.equal(shared[key], fresh_ids)

    deterministic = PairedStructuredController(
        sampler, "shared_round0_deterministic", {key: {}}, shared,
        use_cuda=False)
    deterministic.key = key
    with deterministic:
        generators = make_sampling_generators(
            key[0], key[1], torch.device("cpu"))
        deterministic_ids = sampler_module.weighted_gumbel_top_p(
            initial_score, initial_mask, 1.0, generators["initial"])
        before = generators["round_1"].get_state().clone()
        refined = sampler_module.structured_gumbel_assignment(
            conditional, conditional_mask, 1.0, generators["round_1"])
        after = generators["round_1"].get_state()
    assert torch.equal(fresh_ids, deterministic_ids)
    assert torch.equal(before, after)
    assert torch.equal(
        refined,
        maximum_weight_injective_assignment(
            conditional, conditional_mask))
    assert deterministic.boundary_checks == {
        "shared_round0_exact": 1,
        "round_generator_unchanged": 1,
    }


def test_cross_seed_support_uses_unordered_sets_and_world_matching():
    support_rows = []
    world_rows = []
    for seed, excluded in zip(range(2035, 2040), (3, 3, 2, 3, 1)):
        support = tuple(value for value in range(4) if value != excluded)
        support_rows.append({
            "round": "final", "seed": seed, "window": 0, "agent": 0,
            "support": support, "excluded": (excluded,),
        })
        worlds = torch.tensor([[0, 0], [1, 1], [2, 2]])
        if seed % 2:
            worlds = worlds[[2, 0, 1]]
        world_rows.extend((
            {"round": "round0", "seed": seed, "window": 0,
             "worlds": worlds},
            {"round": "final", "seed": seed, "window": 0,
             "worlds": worlds},
        ))
    summary = summarize_cross_seed_support(support_rows, world_rows)
    assert summary["distinct_excluded_candidate_ids"]["mean"] == 3
    assert summary["all_five_seeds_same_excluded_id_rate"] == 0
    assert summary["joint_world_set_matching"]["final"][
        "exact_whole_set_equality_rate"] == 1


def test_synthetic_tie_audit_distinguishes_unique_and_multiple_optima():
    result = synthetic_tie_audit()
    assert result["unique_optimum"]["pathwise_equal"]
    assert result["unique_optimum"]["objective_equal"]
    assert result["identical_rows"]["objective_equal"]
    assert result["identical_rows"]["unordered_support_equal"]
    assert not result["identical_rows"]["pathwise_equal"]


def test_real_permutation_audit_preserves_unique_optimum_world_set():
    score, mask = _conditional_score(agents=2)
    selected = maximum_weight_injective_assignment(score, mask)
    trace = [
        {"score": score, "mask": mask, "candidate_index": selected},
        {"score": score, "mask": mask, "candidate_index": selected},
        {"score": score, "mask": mask, "candidate_index": selected},
    ]
    rows = real_permutation_records(
        trace, torch.tensor([[0], [1]]), 2035, 0)
    assert len(rows) == 6
    assert all(row["maximum_objective_delta"] == 0 for row in rows)
    assert all(row["all_agent_support_equal"] for row in rows)
    assert all(row["exact_whole_world_set_equal"] for row in rows)
