import random

import numpy as np
import pytest
import torch

import src.models.joint_dependency_v2.joint_sampler as sampler_module
from tools.sampler_coverage_intervention import (
    POLICIES,
    PairedSamplerIntervention,
    capture_rng_state,
    initial_candidate_ids,
    restore_rng_state,
    rng_states_equal,
)


def fixture_score(agents=3, samples=20, candidates=21, device="cpu"):
    base = torch.linspace(-2, 2, candidates, device=device)[None].expand(
        agents, -1).clone()
    base += torch.arange(agents, device=device)[:, None] * 0.03
    score = base[:, None].expand(-1, samples, -1).clone()
    mask = torch.ones_like(score, dtype=torch.bool)
    return score, mask


def test_iid_policy_is_exact_existing_primitive():
    score, mask = fixture_score()
    torch.manual_seed(19)
    expected = sampler_module._categorical(
        score, mask, 1.0, "sample", None)
    torch.manual_seed(19)
    actual = initial_candidate_ids(
        score, mask, 1.0, "sample", None, POLICIES[0])
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("policy", POLICIES[1:])
def test_coverage_policies_are_unique_and_valid(policy):
    score, mask = fixture_score()
    generator = torch.Generator().manual_seed(31)
    selected = initial_candidate_ids(
        score, mask, 1.0, "sample", generator, policy)
    assert selected.shape == (3, 20)
    assert torch.all((selected >= 0) & (selected < 21))
    assert all(torch.unique(row).numel() == 20 for row in selected)


def test_weighted_without_replacement_p_equals_k_covers_all_once():
    score, mask = fixture_score(samples=21)
    selected = initial_candidate_ids(
        score, mask, 1.0, "sample",
        torch.Generator().manual_seed(7),
        "weighted_without_replacement")
    expected = torch.arange(21)
    for row in selected.cpu():
        assert torch.equal(row.sort().values, expected)


def test_deterministic_top_p_is_deterministic():
    score, mask = fixture_score()
    first = initial_candidate_ids(
        score, mask, 1.0, "sample", None, "deterministic_top_p")
    torch.manual_seed(999)
    second = initial_candidate_ids(
        score, mask, 1.0, "sample", None, "deterministic_top_p")
    assert torch.equal(first, second)
    assert torch.equal(first[:, 0], torch.full((3,), 20))


def test_weighted_without_replacement_fixed_generator_repeats():
    score, mask = fixture_score()
    first = initial_candidate_ids(
        score, mask, 1.0, "sample",
        torch.Generator().manual_seed(41),
        "weighted_without_replacement")
    second = initial_candidate_ids(
        score, mask, 1.0, "sample",
        torch.Generator().manual_seed(41),
        "weighted_without_replacement")
    assert torch.equal(first, second)


def test_weighted_without_replacement_different_seeds_change_legal_subset():
    score, mask = fixture_score(agents=1)
    first = initial_candidate_ids(
        score, mask, 1.0, "sample",
        torch.Generator().manual_seed(1),
        "weighted_without_replacement")
    second = initial_candidate_ids(
        score, mask, 1.0, "sample",
        torch.Generator().manual_seed(2),
        "weighted_without_replacement")
    assert not torch.equal(first, second)
    assert torch.unique(first).numel() == torch.unique(second).numel() == 20


def test_mask_and_agent_isolation_no_cross_agent_leakage():
    score, mask = fixture_score(agents=2)
    mask[0, :, 0] = False
    first = initial_candidate_ids(
        score, mask, 1.0, "sample",
        torch.Generator().manual_seed(53),
        "weighted_without_replacement")
    changed = score.clone()
    changed[0, :, 1:] = changed[0, :, 1:].flip(-1)
    second = initial_candidate_ids(
        changed, mask, 1.0, "sample",
        torch.Generator().manual_seed(53),
        "weighted_without_replacement")
    assert not (first[0] == 0).any()
    assert torch.equal(first[1], second[1])


def test_single_agent_e_zero_style_input_is_valid():
    score, mask = fixture_score(agents=1)
    selected = initial_candidate_ids(
        score, mask, 1.0, "sample",
        torch.Generator().manual_seed(67),
        "weighted_without_replacement")
    assert selected.shape == (1, 20)
    assert torch.unique(selected).numel() == 20


def test_p_greater_than_k_is_explicit_error():
    score, mask = fixture_score(samples=22)
    with pytest.raises(ValueError, match="P<=K"):
        initial_candidate_ids(
            score, mask, 1.0, "sample",
            torch.Generator().manual_seed(71),
            "weighted_without_replacement")


def test_nonfinite_valid_logits_are_rejected():
    score, mask = fixture_score()
    score[0, :, 2] = float("nan")
    with pytest.raises(FloatingPointError):
        initial_candidate_ids(
            score, mask, 1.0, "sample",
            torch.Generator().manual_seed(73),
            "weighted_without_replacement")


def test_complete_rng_snapshot_restores_python_numpy_and_torch():
    random.seed(11)
    np.random.seed(11)
    torch.manual_seed(11)
    saved = capture_rng_state(use_cuda=False)
    expected = (random.random(), np.random.rand(), torch.rand(3))
    restore_rng_state(saved)
    actual = (random.random(), np.random.rand(), torch.rand(3))
    assert expected[0] == actual[0]
    assert expected[1] == actual[1]
    assert torch.equal(expected[2], actual[2])


def test_paired_downstream_rng_and_default_primitive_restoration():
    original = sampler_module._categorical
    score, mask = fixture_score(agents=1)
    reference = {}
    torch.manual_seed(101)
    with PairedSamplerIntervention(
            "iid_with_replacement", reference, use_cuda=False) as controller:
        controller.begin_window(2035, 0, edge_count=1,
                                device=torch.device("cpu"))
        for _ in range(3):
            sampler_module._categorical(
                score, mask, 1.0, "sample", None)
        diffusion_a = torch.randn(8)
        controller.end_window()

    torch.manual_seed(999)
    with PairedSamplerIntervention(
            "weighted_without_replacement", reference,
            use_cuda=False) as controller:
        controller.begin_window(2035, 0, edge_count=1,
                                device=torch.device("cpu"))
        for _ in range(3):
            sampler_module._categorical(
                score, mask, 1.0, "sample", None)
        diffusion_b = torch.randn(8)
        controller.end_window()
    assert torch.equal(diffusion_a, diffusion_b)
    assert controller.boundary_checks == {
        "pre_refinement": 1, "pre_diffusion": 1, "window_end": 1}
    assert sampler_module._categorical is original


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_rng_snapshot_roundtrip_all_devices():
    torch.cuda.manual_seed_all(79)
    saved = capture_rng_state(use_cuda=True)
    expected = torch.rand(8, device="cuda")
    restore_rng_state(saved)
    actual = torch.rand(8, device="cuda")
    assert torch.equal(expected, actual)
    restore_rng_state(saved)
    assert rng_states_equal(capture_rng_state(use_cuda=True), saved)
