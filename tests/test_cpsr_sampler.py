import itertools

import pytest
import torch

from src.models.interaction_graph import EDGE_FEATURE_DIM
from src.models.joint_dependency_v2 import (
    DynamicHypothesisRelation,
    ParallelConditionalSampler,
    RelationSpecificJointEnergy,
)
from src.models.joint_dependency_v2.joint_sampler import (
    make_sampling_generators,
    maximum_weight_injective_assignment,
    structured_gumbel_assignment,
    weighted_gumbel_top_p,
)
import src.models.joint_dependency_v2.joint_sampler as sampler_module
from src.parser import check_and_add_additional_args, get_parser
from tools.cpsr_inference_validation import (
    CPSRController,
    _churn_records,
)
from tools.sampler_coverage_intervention import capture_rng_state


def _score(agents=2, slots=3, candidates=4, device="cpu"):
    score = torch.tensor(
        [[[1.0, 0.2, -0.3, 0.7],
          [0.1, 1.2, 0.3, -0.2],
          [-0.4, 0.5, 1.1, 0.0]]],
        device=device,
    )[:, :slots, :candidates].expand(agents, -1, -1).clone()
    mask = torch.ones_like(score, dtype=torch.bool)
    return score, mask


def _initial_score(agents=2, slots=3, candidates=4, device="cpu"):
    base = torch.linspace(-1.0, 1.0, candidates, device=device)
    score = base.view(1, 1, -1).expand(agents, slots, -1).clone()
    mask = torch.ones_like(score, dtype=torch.bool)
    return score, mask


def test_default_categorical_path_is_tensor_exact():
    score, mask = _initial_score(agents=3)
    torch.manual_seed(101)
    expected = sampler_module._categorical(
        score, mask, 1.0, "sample", None)
    sampler = ParallelConditionalSampler(strict_no_z=True)
    assert sampler.refinement_policy == "categorical"
    torch.manual_seed(101)
    actual = sampler_module._categorical(
        score, mask, sampler.temperature, "sample", None)
    assert torch.equal(actual, expected)


def test_weighted_initialization_is_unique_and_covers_all_when_p_equals_k():
    score, mask = _initial_score(slots=4, candidates=4)
    ids = weighted_gumbel_top_p(
        score, mask, 1.0, torch.Generator().manual_seed(7))
    expected = torch.arange(4)
    for row in ids:
        assert torch.equal(row.sort().values, expected)


def test_cpsr_assignment_rows_and_capacity_are_feasible():
    score, mask = _score()
    ids = structured_gumbel_assignment(
        score, mask, 1.0, torch.Generator().manual_seed(13))
    assert ids.shape == (2, 3)
    for agent in range(2):
        assert torch.unique(ids[agent]).numel() == 3
        assert mask[agent].gather(-1, ids[agent, :, None]).all()


def test_masked_candidate_is_never_selected():
    score, mask = _score(agents=1)
    score[0, :, 0] = 1e6
    mask[0, :, 0] = False
    ids = structured_gumbel_assignment(
        score, mask, 1.0, torch.Generator().manual_seed(17))
    assert not (ids == 0).any()


def test_cpsr_v1_rejects_slot_varying_mask_and_valid_k_less_than_p():
    score, mask = _score(agents=1)
    mask[0, 1, 0] = False
    with pytest.raises(ValueError, match="slot-invariant"):
        structured_gumbel_assignment(
            score, mask, 1.0, torch.Generator().manual_seed(19))
    score, mask = _score(agents=1)
    mask[:, :, :2] = False
    with pytest.raises(ValueError, match="valid_K >= P"):
        structured_gumbel_assignment(
            score, mask, 1.0, torch.Generator().manual_seed(19))


def test_cpsr_rejects_p_greater_than_k():
    score = torch.randn(1, 5, 4)
    mask = torch.ones_like(score, dtype=torch.bool)
    with pytest.raises(ValueError, match="P<=K"):
        structured_gumbel_assignment(
            score, mask, 1.0, torch.Generator().manual_seed(23))


def test_fixed_generators_reproduce_and_different_seed_changes_assignment():
    score, mask = _score(agents=1)
    first = structured_gumbel_assignment(
        score, mask, 1.0, torch.Generator().manual_seed(29))
    second = structured_gumbel_assignment(
        score, mask, 1.0, torch.Generator().manual_seed(29))
    third = structured_gumbel_assignment(
        score, mask, 1.0, torch.Generator().manual_seed(31))
    assert torch.equal(first, second)
    assert not torch.equal(first, third)


def test_exact_assignment_matches_brute_force():
    score, mask = _score(agents=1)
    actual = maximum_weight_injective_assignment(score, mask)[0]
    best = max(
        itertools.permutations(range(4), 3),
        key=lambda columns: sum(
            float(score[0, row, column])
            for row, column in enumerate(columns)),
    )
    actual_value = sum(
        float(score[0, row, actual[row]]) for row in range(3))
    expected_value = sum(
        float(score[0, row, best[row]]) for row in range(3))
    assert actual_value == pytest.approx(expected_value)


def test_existing_temperature_is_applied_once():
    score, mask = _score(agents=1)
    noise = torch.zeros_like(score)
    actual = structured_gumbel_assignment(
        score, mask, 2.0, None, gumbel_noise=noise)
    expected = maximum_weight_injective_assignment(score / 2.0, mask)
    assert torch.equal(actual, expected)


def test_slot_permutation_equivariance_with_coupled_noise():
    score, mask = _score(agents=1)
    noise = torch.randn_like(score)
    original = structured_gumbel_assignment(
        score, mask, 1.0, None, gumbel_noise=noise)
    permutation = torch.tensor([2, 0, 1])
    permuted = structured_gumbel_assignment(
        score[:, permutation], mask[:, permutation], 1.0, None,
        gumbel_noise=noise[:, permutation])
    assert torch.equal(permuted, original[:, permutation])


def test_agent_permutation_equivariance_with_coupled_noise():
    score, mask = _score(agents=2)
    score[1] += torch.tensor([0.3, -0.1, 0.2, -0.2])
    noise = torch.randn_like(score)
    original = structured_gumbel_assignment(
        score, mask, 1.0, None, gumbel_noise=noise)
    permutation = torch.tensor([1, 0])
    permuted = structured_gumbel_assignment(
        score[permutation], mask[permutation], 1.0, None,
        gumbel_noise=noise[permutation])
    assert torch.equal(permuted, original[permutation])


def test_agent_and_scene_isolation():
    score, mask = _score(agents=2)
    noise = torch.randn_like(score)
    original = structured_gumbel_assignment(
        score, mask, 1.0, None, gumbel_noise=noise)
    changed = score.clone()
    changed[0] = changed[0].flip(-1)
    updated = structured_gumbel_assignment(
        changed, mask, 1.0, None, gumbel_noise=noise)
    assert torch.equal(original[1], updated[1])


def test_sampling_streams_are_separate_and_reproducible():
    first = make_sampling_generators(2035, 9, torch.device("cpu"))
    second = make_sampling_generators(2035, 9, torch.device("cpu"))
    values_first = {name: torch.rand(8, generator=value)
                    for name, value in first.items()}
    values_second = {name: torch.rand(8, generator=value)
                     for name, value in second.items()}
    assert all(torch.equal(values_first[name], values_second[name])
               for name in values_first)
    assert not torch.equal(values_first["initial"], values_first["round_1"])


def test_single_agent_e0_full_sampler_and_context_consumption():
    n, k = 1, 21
    goals = torch.randn(n, k, 2)
    last = torch.randn(n, 2)
    agent = torch.randn(n, 128)
    edge = torch.empty((2, 0), dtype=torch.long)
    edge_feat = torch.empty((0, EDGE_FEATURE_DIM))
    base = torch.empty((0, 4))
    unary = torch.linspace(-1, 1, k).view(1, -1)
    relation = DynamicHypothesisRelation(
        6.0, use_scene_latent=False)
    energy = RelationSpecificJointEnergy(use_scene_latent=False)
    sampler = ParallelConditionalSampler(
        strict_no_z=True,
        refinement_policy="structured_gumbel_assignment")
    sampler.set_sampling_context(2035, 0)
    output = sampler(
        torch.randn(n, k), goals, None, torch.zeros(n, dtype=torch.long),
        edge, edge_feat, agent, last, base,
        DynamicHypothesisRelation(6.0, use_scene_latent=False),
        RelationSpecificJointEnergy(use_scene_latent=False),
        sampling_mode="sample", use_scene_latent=False)
    assert output["candidate_index"].shape == (1, 20)
    assert torch.unique(output["candidate_index"]).numel() == 20
    with pytest.raises(RuntimeError, match="sampling context"):
        sampler(
            torch.randn(n, k), goals, None,
            torch.zeros(n, dtype=torch.long), edge, edge_feat, agent, last,
            base, DynamicHypothesisRelation(
                6.0, use_scene_latent=False),
            RelationSpecificJointEnergy(use_scene_latent=False),
            sampling_mode="sample", use_scene_latent=False)


def test_parser_cpsr_contract_and_default():
    default = get_parser().parse_args([])
    assert default.jdv2_refinement_policy == "categorical"
    args = get_parser().parse_args([
        "--device", "cpu", "--goal_model_type", "jdv2",
        "--training_stage", "joint_goal", "--data_augmentation", "False",
        "--use_scene_latent", "False", "--jdv2_latent_objective",
        "strict_no_z", "--jdv2_refinement_policy",
        "structured_gumbel_assignment",
    ])
    checked = check_and_add_additional_args(args)
    assert checked.jdv2_refinement_policy == "structured_gumbel_assignment"


def test_diagnostic_callback_has_round_shapes_and_churn_inputs():
    n, k = 2, 21
    goals = torch.randn(n, k, 2)
    last = torch.randn(n, 2)
    agent = torch.randn(n, 128)
    edge = torch.tensor([[0], [1]])
    edge_feat = torch.randn(1, EDGE_FEATURE_DIM)
    base = torch.randn(1, 4)
    sampler = ParallelConditionalSampler(
        strict_no_z=True,
        refinement_policy="structured_gumbel_assignment")
    trace = []

    def callback(**record):
        trace.append({key: (value.detach().clone()
                            if torch.is_tensor(value) else value)
                      for key, value in record.items()})

    sampler.diagnostic_callback = callback
    sampler.set_sampling_context(2035, 1)
    sampler(
        torch.randn(n, k), goals, None, torch.zeros(n, dtype=torch.long),
        edge, edge_feat, agent, last, base,
        DynamicHypothesisRelation(6.0, use_scene_latent=False),
        RelationSpecificJointEnergy(use_scene_latent=False),
        sampling_mode="sample", use_scene_latent=False)
    assert [row["round_index"] for row in trace] == [0, 1, 2]
    assert trace[0]["previous_candidate_index"] is None
    assert trace[1]["previous_candidate_index"].shape == (n, 20)
    assert trace[2]["candidate_index"].shape == (n, 20)


def test_nonfinite_score_and_noise_are_rejected():
    score, mask = _score(agents=1)
    score[0, 0, 0] = float("nan")
    with pytest.raises(FloatingPointError):
        structured_gumbel_assignment(score, mask, 1.0, None,
                                     gumbel_noise=torch.zeros_like(score))
    score, mask = _score(agents=1)
    noise = torch.zeros_like(score)
    noise[0, 0, 0] = float("inf")
    with pytest.raises(FloatingPointError):
        structured_gumbel_assignment(
            score, mask, 1.0, None, gumbel_noise=noise)


def test_churn_and_two_cycle_diagnostic():
    trace = [
        {"candidate_index": torch.tensor([[0, 1, 2, 3]])},
        {"candidate_index": torch.tensor([[1, 0, 2, 3]])},
        {"candidate_index": torch.tensor([[0, 1, 3, 2]])},
    ]
    metadata = [{
        "variant": "structured_gumbel_assignment", "seed": 1,
        "window": 0, "agent": 0, "edge_class": "E>0",
        "agent_count_bin": "N=1", "degree_bin": "degree=0",
    }]
    row = _churn_records(trace, metadata, torch.tensor([True]))[0]
    assert row["round0_to_round1_slot_churn"] == pytest.approx(0.5)
    assert row["round1_to_round2_slot_churn"] == pytest.approx(1.0)
    assert row["two_cycle_rate"] == pytest.approx(0.5)


def test_cpsr_controller_pairs_diffusion_and_preserves_global_rng():
    n, k = 1, 21
    goals = torch.randn(n, k, 2)
    last = torch.randn(n, 2)
    agent = torch.randn(n, 128)
    edge = torch.empty((2, 0), dtype=torch.long)
    edge_feat = torch.empty((0, EDGE_FEATURE_DIM))
    base = torch.empty((0, 4))
    unary = torch.linspace(-1, 1, k).view(1, -1)
    relation = DynamicHypothesisRelation(
        6.0, use_scene_latent=False)
    energy = RelationSpecificJointEnergy(use_scene_latent=False)
    sampler = ParallelConditionalSampler(strict_no_z=True)
    key = (2035, 0)
    torch.manual_seed(101)
    pre_initial = capture_rng_state(use_cuda=False)
    torch.rand(37)
    pre_diffusion = capture_rng_state(use_cuda=False)
    expected_diffusion = torch.randn(8)
    window_end = capture_rng_state(use_cuda=False)
    reference = {key: {
        "pre_initial": pre_initial,
        "pre_diffusion": pre_diffusion,
        "window_end": window_end,
    }}
    torch.manual_seed(999)
    with CPSRController(
            sampler, reference, use_cuda=False) as controller:
        controller.begin_window(2035, 0, edge_count=0,
                                device=torch.device("cpu"))
        sampler(
            unary, goals, None,
            torch.zeros(n, dtype=torch.long), edge, edge_feat, agent, last,
            base, relation, energy,
            sampling_mode="sample", use_scene_latent=False)
        actual_diffusion = torch.randn(8)
        controller.end_window()
    assert torch.equal(actual_diffusion, expected_diffusion)
    assert controller.boundary_checks == {
        "sampler_global_rng_unchanged": 1,
        "pre_diffusion": 1,
        "window_end": 1,
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cpsr_cuda_dtype_safety():
    score, mask = _score(agents=1, device="cuda")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        ids = structured_gumbel_assignment(
            score.to(torch.bfloat16), mask, 1.0,
            torch.Generator(device="cuda").manual_seed(43))
    assert ids.is_cuda
    assert torch.unique(ids).numel() == 3
