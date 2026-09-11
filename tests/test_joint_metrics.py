import math

import pytest
import torch

from src.metrics import (
    ADE_best_of,
    FDE_best_of,
    JADE,
    JFDE,
    collision_rate,
    goal_minFDE,
    goal_recall_at_K,
    joint_goal_compatibility,
    joint_goal_endpoint_error,
    minADE_at_K,
    minFDE_at_K,
)


def test_legacy_ade_fde_regression_remains_available():
    """The original marginal metrics retain per-agent best-of-K behavior."""
    ground_truth = torch.zeros(2, 2, 2)
    predictions = torch.zeros(2, 2, 2, 2)
    predictions[0, :, 1, 0] = 4.0
    predictions[1, :, 0, 0] = 3.0
    metric_mask = torch.tensor([True, True])

    assert ADE_best_of(
        predictions, ground_truth, metric_mask, obs_length=0
    ) == pytest.approx([0.0, 0.0])
    assert FDE_best_of(
        predictions, ground_truth, metric_mask, obs_length=0
    ) == pytest.approx([0.0, 0.0])


def test_joint_metrics_cannot_mix_best_sample_across_agents():
    ground_truth = torch.zeros(1, 2, 2)
    predictions = torch.zeros(2, 1, 2, 2)
    predictions[0, 0, 1, 0] = 10.0
    predictions[1, 0, 0, 0] = 10.0

    assert minADE_at_K(
        predictions, ground_truth, obs_length=0) == pytest.approx([0.0, 0.0])
    assert minFDE_at_K(
        predictions, ground_truth, obs_length=0) == pytest.approx([0.0, 0.0])
    assert JADE(
        predictions, ground_truth, obs_length=0) == pytest.approx([5.0])
    assert JFDE(
        predictions, ground_truth, obs_length=0) == pytest.approx([5.0])


def test_scene_index_selects_joint_sample_per_scene():
    ground_truth = torch.zeros(1, 4, 2)
    predictions = torch.zeros(2, 1, 4, 2)
    predictions[0, :, 2:, 0] = 10.0
    predictions[1, :, :2, 0] = 10.0
    scene_index = torch.tensor([3, 3, 8, 8])

    assert JADE(
        predictions, ground_truth, scene_index=scene_index,
        obs_length=0) == pytest.approx([0.0, 0.0])
    assert JFDE(
        predictions, ground_truth, scene_index=scene_index,
        obs_length=0) == pytest.approx([0.0, 0.0])


def test_collision_rate_never_pairs_agents_from_different_scenes():
    predictions = torch.tensor(
        [[[[0.0, 0.0], [2.0, 0.0],
           [0.0, 0.0], [2.0, 0.0]]]])
    scene_index = torch.tensor([0, 0, 1, 1])

    rates = collision_rate(
        predictions, collision_threshold_meter=0.2,
        scene_index=scene_index, obs_length=0)
    assert rates == pytest.approx([0.0, 0.0])


@pytest.mark.parametrize('method', ['discrete', 'segment', 'interpolate'])
def test_single_agent_collision_rate_is_finite(method):
    predictions = torch.zeros(2, 3, 1, 2)
    rates = collision_rate(
        predictions, collision_threshold_meter=0.2,
        obs_length=0, method=method, interpolation_steps=1)

    assert rates == [0.0]
    assert math.isfinite(rates[0])


def test_segment_and_interpolation_detect_between_frame_collision():
    predictions = torch.tensor(
        [[
            [[-1.0, 0.0], [1.0, 0.0]],
            [[1.0, 0.0], [-1.0, 0.0]],
        ]])

    discrete = collision_rate(
        predictions, collision_threshold_meter=0.1,
        obs_length=0, method='discrete')
    segment = collision_rate(
        predictions, collision_threshold_meter=0.1,
        obs_length=0, method='segment')
    interpolated = collision_rate(
        predictions, collision_threshold_meter=0.1,
        obs_length=0, method='interpolate', interpolation_steps=1)

    assert discrete == [0.0]
    assert segment == [1.0]
    assert interpolated == [1.0]


def test_continuous_collision_includes_observation_prediction_boundary():
    predictions = torch.tensor(
        [[
            [[-1.0, 0.0], [1.0, 0.0]],  # final observation
            [[1.0, 0.0], [-1.0, 0.0]],  # first future step
            [[2.0, 0.0], [-2.0, 0.0]],
        ]])

    assert collision_rate(
        predictions, collision_threshold_meter=0.1,
        obs_length=1, method='discrete') == [0.0]
    assert collision_rate(
        predictions, collision_threshold_meter=0.1,
        obs_length=1, method='segment') == [1.0]
    assert collision_rate(
        predictions, collision_threshold_meter=0.1,
        obs_length=1, method='interpolate', interpolation_steps=1) == [1.0]


def test_collision_threshold_is_inclusive_and_must_be_finite():
    predictions = torch.tensor(
        [[[[0.0, 0.0], [0.2, 0.0]]]], dtype=torch.float64)

    assert collision_rate(
        predictions, collision_threshold_meter=0.2,
        obs_length=0) == [1.0]
    assert collision_rate(
        predictions, collision_threshold_meter=0.19,
        obs_length=0) == [0.0]
    with pytest.raises(ValueError):
        collision_rate(
            predictions, collision_threshold_meter=float('nan'),
            obs_length=0)


def test_goal_marginal_and_joint_metrics_use_distinct_selection_rules():
    ground_truth = torch.zeros(2, 2)
    predictions = torch.zeros(2, 2, 2)
    predictions[0, 1, 0] = 10.0
    predictions[1, 0, 0] = 10.0

    assert goal_minFDE(
        predictions, ground_truth) == pytest.approx([0.0, 0.0])
    assert goal_recall_at_K(
        predictions, ground_truth,
        threshold_meter=0.5) == pytest.approx([1.0, 1.0])
    assert joint_goal_endpoint_error(
        predictions, ground_truth) == pytest.approx([5.0])
    assert joint_goal_compatibility(
        predictions, ground_truth) == pytest.approx([10.0])


def test_goal_compatibility_measures_relative_formation():
    ground_truth = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    translated_goals = torch.tensor([[[10.0, 4.0], [11.0, 4.0]]])

    assert joint_goal_endpoint_error(
        translated_goals, ground_truth) == pytest.approx([
            math.sqrt(116.0),
        ])
    assert joint_goal_compatibility(
        translated_goals, ground_truth) == pytest.approx([0.0])


def test_goal_compatibility_never_constructs_cross_scene_pairs():
    ground_truth = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
    translated_goals = torch.tensor(
        [[[10.0, 0.0], [11.0, 0.0], [-10.0, 0.0], [-9.0, 0.0]]])
    scene_index = torch.tensor([0, 0, 1, 1])

    assert joint_goal_compatibility(
        translated_goals, ground_truth,
        scene_index=scene_index) == pytest.approx([0.0, 0.0])


def test_agent_mask_applies_before_joint_and_collision_metrics():
    ground_truth = torch.zeros(1, 3, 2)
    predictions = torch.zeros(1, 1, 3, 2)
    predictions[:, :, 1, 0] = 2.0
    predictions[:, :, 2, 0] = 100.0
    metric_mask = torch.tensor([True, True, False])

    assert JADE(
        predictions, ground_truth, metric_mask=metric_mask,
        obs_length=0) == pytest.approx([1.0])
    assert collision_rate(
        predictions, collision_threshold_meter=0.2,
        metric_mask=metric_mask, obs_length=0) == [0.0]
