"""Contracts for the read-only Stage-B V1 failure-mechanism audit."""

from __future__ import annotations

import copy

import pytest
import torch

from tools.jdv2_stage_b_v1_failure_mechanism_audit import (
    NoiseTape,
    connected_components,
    decompose_component_residual,
    edge_relative,
    gradient_comparison,
    relation_for_mode,
    rng_restore,
    rng_snapshot,
    summarize_branch_rows,
    transform_delta,
)


def _graph(device="cpu"):
    return torch.tensor([[0, 1, 3], [1, 2, 4]], device=device)


def test_component_decomposition_reconstructs_delta():
    torch.manual_seed(4)
    delta = torch.randn(6, 12, 2)
    delta[5].zero_()  # production corrector is exact-zero at degree zero
    parts = decompose_component_residual(delta, _graph())
    assert torch.allclose(parts["common"] + parts["centered"], delta)


def test_component_centered_sums_to_zero_and_degree_zero_is_zero():
    delta = torch.randn(6, 12, 2)
    parts = decompose_component_residual(delta, _graph())
    for component in parts["components"]:
        assert torch.allclose(
            parts["centered"][component].sum(0), torch.zeros(12, 2),
            atol=1e-6)
    assert torch.equal(parts["centered"][5], torch.zeros_like(delta[5]))
    assert torch.equal(parts["common"][5], torch.zeros_like(delta[5]))


def test_edge_relative_residual_unchanged_by_component_centering():
    delta = torch.randn(6, 12, 2)
    parts = decompose_component_residual(delta, _graph())
    assert torch.allclose(edge_relative(parts["centered"], _graph()),
                          edge_relative(delta, _graph()), atol=1e-6)


def test_connected_components_excludes_isolated_agent():
    components, degree = connected_components(6, _graph())
    assert [component.tolist() for component in components] == [[0, 1, 2],
                                                                [3, 4]]
    assert degree.tolist() == [1, 2, 1, 1, 1, 0]


def test_full_and_none_transform_contracts():
    delta = torch.randn(6, 12, 2)
    scene = torch.zeros(6, dtype=torch.long)
    assert torch.equal(transform_delta(
        delta, "full_v1", _graph(), scene, 0), delta)
    assert torch.equal(transform_delta(
        delta, "stage_a", _graph(), scene, 0), torch.zeros_like(delta))


def test_oracle_only_modifies_only_selected_scene_branch():
    delta = torch.ones(5, 2, 2)
    edge = torch.tensor([[0, 2], [1, 3]])
    scene = torch.tensor([0, 0, 1, 1, 2])
    selected = torch.tensor([3, 3, 7, 7, 1])
    result = transform_delta(delta, "oracle_only", edge, scene, 7, selected)
    assert torch.equal(result[2:4], delta[2:4])
    assert torch.equal(result[[0, 1, 4]], torch.zeros_like(result[[0, 1, 4]]))


def test_scene_centered_uses_only_degree_positive_agents():
    delta = torch.arange(5 * 2 * 2, dtype=torch.float32).reshape(5, 2, 2)
    edge = torch.tensor([[0, 2], [1, 3]])
    scene = torch.tensor([0, 0, 1, 1, 1])
    result = transform_delta(delta, "scene_centered", edge, scene, 0)
    assert torch.allclose(result[:2].sum(0), torch.zeros(2, 2))
    assert torch.allclose(result[2:4].sum(0), torch.zeros(2, 2))
    assert torch.equal(result[4], torch.zeros_like(result[4]))


def test_relation_variants_preserve_shape_and_semantics():
    relation = torch.randn(3, 20, 16)
    assert relation_for_mode(relation, "full_v1", 4).shape == (3, 16)
    assert torch.equal(relation_for_mode(relation, "zero_relation", 4),
                       torch.zeros(3, 16))
    assert torch.allclose(relation_for_mode(relation, "mean_relation", 4),
                          relation.mean(1))
    assert torch.equal(relation_for_mode(
        relation, "branch_shuffled_relation", 4), relation[:, 5])


def test_rng_snapshot_replay_is_deterministic():
    state = rng_snapshot(use_cuda=False)
    left = torch.randn(7)
    rng_restore(state)
    right = torch.randn(7)
    assert torch.equal(left, right)


def test_noise_tape_container_does_not_mutate_tensors():
    x = torch.randn(2, 12, 2)
    noise = tuple(tuple(torch.randn_like(x) for _ in range(6))
                  for _ in range(20))
    tape = NoiseTape(x, noise)
    copied = copy.deepcopy(tape)
    assert torch.equal(tape.x_T, copied.x_T)
    assert all(torch.equal(a, b) for row_a, row_b in zip(
        tape.branch_noise, copied.branch_noise) for a, b in zip(row_a, row_b))


def test_goal_and_trajectory_oracle_ranking_logic():
    rows = []
    for branch in range(20):
        rows.append({
            "goal_error": float(branch), "goal_rank": branch + 1,
            "trajectory_rank": 20 - branch, "ade_stage_a": float(20 - branch),
            "fde_stage_a": 0.0, "delta_ade": 0.1 * branch,
            "delta_fde": 0.2 * branch, "delta_relative": 0.0,
            "goal_oracle": branch == 0,
            "trajectory_oracle": branch == 19,
        })
    result = summarize_branch_rows(rows)
    assert result["scene_count"] == 1
    assert result["goal_oracle_equals_trajectory_oracle_rate"] == 0.0
    assert result["trajectory_oracle_goal_rank"]["mean"] == 20.0


def test_gradient_comparison_reports_alignment_and_conflict():
    aligned = gradient_comparison(torch.tensor([1.0, 0]),
                                  torch.tensor([2.0, 0]))
    conflict = gradient_comparison(torch.tensor([1.0, 0]),
                                   torch.tensor([-2.0, 0]))
    assert aligned["cosine_raw"] == pytest.approx(1.0)
    assert not aligned["conflict"]
    assert conflict["cosine_raw"] == pytest.approx(-1.0)
    assert conflict["conflict"]


def test_counterfactuals_do_not_mutate_ids_or_goals():
    ids = torch.arange(20).repeat(4, 1)
    goals = torch.randn(4, 20, 2)
    ids_before, goals_before = ids.clone(), goals.clone()
    delta = torch.randn(4, 12, 2)
    edge = torch.tensor([[0, 1], [1, 2]])
    scene = torch.zeros(4, dtype=torch.long)
    for mode in ("stage_a", "full_v1", "common_only",
                 "component_centered", "scene_centered"):
        transform_delta(delta, mode, edge, scene, 0)
    assert torch.equal(ids, ids_before)
    assert torch.equal(goals, goals_before)
    assert all(torch.unique(row).numel() == 20 for row in ids)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_bf16_component_counterfactual_executes():
    delta = torch.randn(5, 12, 2, device="cuda", dtype=torch.bfloat16)
    edge = _graph("cuda")[:, :2]
    scene = torch.zeros(5, dtype=torch.long, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        centered = transform_delta(
            delta, "component_centered", edge, scene, 0)
    assert centered.shape == delta.shape
    assert torch.isfinite(centered.float()).all()


def test_audit_file_does_not_patch_production_source():
    # The audit contract is structural: all intervention code is local.
    import inspect
    from tools import jdv2_stage_b_v1_failure_mechanism_audit as audit
    source = inspect.getsource(audit)
    assert "src/models" not in source
    assert "optimizer.step" not in source
