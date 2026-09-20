import itertools

import numpy as np
import pytest
import torch

import src.models.joint_dependency_v2.joint_sampler as sampler_module
from src.models.joint_dependency_v2.joint_sampler import ParallelConditionalSampler
from tools.refinement_coalescence_audit import (
    RefinementAuditController,
    _world_records,
    coverage_assignment,
    refinement_map,
)
from tools.sampler_coverage_intervention import capture_rng_state


def fixture_score(agents=2, slots=3, candidates=4, device="cpu"):
    base = torch.tensor([0.2, 1.3, -0.7, 0.8], device=device)
    score = base[:candidates].view(1, 1, -1).expand(
        agents, slots, candidates).clone()
    score += torch.arange(slots, device=device).view(1, -1, 1) * torch.tensor(
        [0.1, -0.2, 0.3, -0.1], device=device)[:candidates]
    mask = torch.ones_like(score, dtype=torch.bool)
    return score, mask


def test_map_is_deterministic_and_valid():
    score, mask = fixture_score()
    first = refinement_map(score, mask)
    torch.manual_seed(999)
    second = refinement_map(score, mask)
    assert torch.equal(first, second)
    assert first.shape == (2, 3)
    assert mask.gather(-1, first[..., None]).all()


def test_assignment_is_unique_valid_and_agent_local():
    score, mask = fixture_score()
    mask[0, 1, 1] = False
    first = coverage_assignment(score, mask)
    changed = score.clone()
    changed[0] = changed[0].flip(-1)
    second = coverage_assignment(changed, mask)
    assert torch.equal(first[1], second[1])
    for agent in range(2):
        assert torch.unique(first[agent]).numel() == 3
        assert mask[agent].gather(-1, first[agent, :, None]).all()


def test_assignment_matches_tiny_brute_force_optimum():
    score, mask = fixture_score(agents=1)
    actual = coverage_assignment(score, mask)[0]
    best_value = -float("inf")
    best = None
    for columns in itertools.permutations(range(4), 3):
        value = sum(float(score[0, row, column])
                    for row, column in enumerate(columns))
        if value > best_value:
            best_value = value
            best = columns
    actual_value = sum(float(score[0, row, actual[row]]) for row in range(3))
    assert actual_value == pytest.approx(best_value)
    assert tuple(actual.tolist()) == best


def test_assignment_never_selects_invalid_high_score():
    score, mask = fixture_score(agents=1)
    score[0, 0, 0] = 1e6
    mask[0, 0, 0] = False
    selected = coverage_assignment(score, mask)[0]
    assert int(selected[0]) != 0
    assert mask[0].gather(-1, selected[:, None]).all()


def test_assignment_rejects_p_greater_than_k_and_nonfinite():
    score = torch.randn(1, 5, 4)
    mask = torch.ones_like(score, dtype=torch.bool)
    with pytest.raises(ValueError, match="P<=K"):
        coverage_assignment(score, mask)
    score, mask = fixture_score(agents=1)
    score[0, 0, 0] = float("nan")
    with pytest.raises(FloatingPointError):
        coverage_assignment(score, mask)


def _run_controller_variant(variant, reference_states, seed=2035):
    sampler = ParallelConditionalSampler(strict_no_z=True)
    original = sampler_module._categorical
    score = torch.linspace(-2, 2, 21)[None, None].expand(1, 20, -1).clone()
    mask = torch.ones_like(score, dtype=torch.bool)
    with RefinementAuditController(
            sampler, variant, reference_states, use_cuda=False) as controller:
        controller.begin_window(seed, 0, edge_count=1, device=torch.device("cpu"))
        outputs = []
        calls = 1 + {"iid_control": 2, "R0": 0, "R1": 1,
                     "R2": 2, "MAP": 2, "ASSIGN": 2}[variant]
        for index in range(calls):
            local = score if index == 0 else score + index * torch.randn_like(score)
            outputs.append(sampler_module._categorical(
                local, mask, 1.0, "sample", None))
        diffusion = torch.randn(7)
        controller.end_window()
    assert sampler_module._categorical is original
    assert sampler.num_refinement_steps == 2
    return outputs, diffusion, controller


def test_r0_r1_r2_round_shapes_and_exact_weighted_r2_control():
    reference = {}
    torch.manual_seed(101)
    iid, diffusion_iid, _ = _run_controller_variant("iid_control", reference)
    assert len(iid) == 3
    torch.manual_seed(999)
    r2_a, diffusion_r2, _ = _run_controller_variant("R2", reference)
    torch.manual_seed(999)
    r2_b, _, _ = _run_controller_variant("R2", reference)
    assert len(r2_a) == 3
    assert all(value.shape == (1, 20) for value in r2_a)
    assert all(torch.equal(a, b) for a, b in zip(r2_a, r2_b))
    assert torch.equal(diffusion_iid, diffusion_r2)

    _, _, r0 = _run_controller_variant("R0", reference)
    _, _, r1 = _run_controller_variant("R1", reference)
    assert len(r0.trace) == 1
    assert len(r1.trace) == 2


@pytest.mark.parametrize("variant", ["MAP", "ASSIGN"])
def test_diagnostic_variants_pair_diffusion_rng(variant):
    reference = {}
    torch.manual_seed(113)
    _, expected_diffusion, _ = _run_controller_variant(
        "iid_control", reference)
    torch.manual_seed(777)
    output, actual_diffusion, controller = _run_controller_variant(
        variant, reference)
    assert torch.equal(expected_diffusion, actual_diffusion)
    assert controller.boundary_checks == {
        "pre_round1": 1,
        "pre_round2": 1,
        "pre_diffusion": 1,
        "window_end": 1,
    }
    if variant == "ASSIGN":
        assert torch.unique(output[1]).numel() == 20
        assert torch.unique(output[2]).numel() == 20


def test_single_agent_e0_has_only_round0_and_paired_diffusion():
    sampler = ParallelConditionalSampler(strict_no_z=True)
    reference = {}
    score = torch.randn(1, 20, 21)
    score[:] = score[:, :1]
    mask = torch.ones_like(score, dtype=torch.bool)
    with RefinementAuditController(
            sampler, "iid_control", reference, use_cuda=False) as control:
        control.begin_window(2035, 0, edge_count=0, device=torch.device("cpu"))
        sampler_module._categorical(score, mask, 1.0, "sample", None)
        expected = torch.randn(5)
        control.end_window()
    with RefinementAuditController(
            sampler, "ASSIGN", reference, use_cuda=False) as diagnostic:
        diagnostic.begin_window(2035, 0, edge_count=0, device=torch.device("cpu"))
        sampler_module._categorical(score, mask, 1.0, "sample", None)
        actual = torch.randn(5)
        diagnostic.end_window()
    assert len(diagnostic.trace) == 1
    assert torch.equal(expected, actual)


def test_world_diversity_is_scene_local():
    ids = torch.tensor([
        [0, 0, 1], [2, 2, 3],
        [0, 1, 1], [2, 3, 3],
    ])
    scene_index = torch.tensor([0, 0, 1, 1])
    rows = _world_records([ids], scene_index, "R0", 1, 0)
    assert len(rows) == 2
    assert all(row["num_agents"] == 2 for row in rows)
    assert all(row["unique_joint_worlds"] == 2 for row in rows)


def test_default_production_primitive_and_rng_snapshot_unchanged():
    original = sampler_module._categorical
    before = capture_rng_state(use_cuda=False)
    reference = {}
    _run_controller_variant("iid_control", reference)
    assert sampler_module._categorical is original
    assert before.torch_cpu.shape == capture_rng_state(use_cuda=False).torch_cpu.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_assignment_and_controller_on_cuda():
    score, mask = fixture_score(agents=1, slots=3, candidates=4, device="cuda")
    selected = coverage_assignment(score, mask)
    assert selected.is_cuda
    assert torch.unique(selected).numel() == 3

