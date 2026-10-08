import torch

from ebjd.metrics import (
    MetricAccumulator, _softmin, hard_metrics, packed_metric_adapter,
    scene_soft_metrics,
)


def test_metrics_match_hand_calculation_and_preserve_worlds():
    target = torch.zeros(1, 2, 12, 2)
    prediction = torch.zeros(1, 2, 2, 12, 2)
    prediction[:, 0, 0, :, 0] = 1
    prediction[:, 0, 1, :, 0] = 3
    prediction[:, 1, 0, :, 0] = 2
    prediction[:, 1, 1, :, 0] = 1
    valid = torch.ones(1, 2, dtype=torch.bool)
    result = hard_metrics(prediction, target, valid)
    assert result.minADE == 1.0 and result.minFDE == 1.0
    assert result.JADE == 1.5 and result.JFDE == 1.5
    assert result.coordination_gap_ADE == 0.5
    soft = scene_soft_metrics(prediction, prediction[..., -1, :], target, valid, 0.05)
    assert soft[0] <= soft[2] + 1e-6 and soft[1] <= soft[3] + 1e-6
    packed = packed_metric_adapter(prediction, target, valid)
    assert packed[0][0].shape == (2, 20, 2, 2)
    assert packed[0][1].shape == (20, 2, 2)


def test_agent_weighting_scene_weighting_and_partition_invariance():
    target = torch.zeros(2, 3, 12, 2)
    prediction = torch.zeros(2, 1, 3, 12, 2)
    prediction[0, 0, 0, :, 0] = 1
    valid = torch.tensor([[True, False, False], [True, True, True]])

    together = MetricAccumulator()
    together.update(prediction, target, valid)
    summary = together.summary()
    assert summary.minADE == 0.25
    assert summary.scene_weighted_minADE == 0.5
    assert summary.JADE == 0.5
    assert summary.coordination_gap_ADE == 0.0
    assert summary.agent_count == 4 and summary.scene_count == 2

    partitioned = MetricAccumulator()
    partitioned.update(prediction[:1], target[:1], valid[:1])
    partitioned.update(prediction[1:], target[1:], valid[1:])
    assert partitioned.as_dict() == together.as_dict()

    padded_prediction = torch.nn.functional.pad(prediction, (0, 0, 0, 0, 0, 2))
    padded_target = torch.nn.functional.pad(target, (0, 0, 0, 0, 0, 2))
    padded_valid = torch.nn.functional.pad(valid, (0, 2))
    padded = MetricAccumulator()
    padded.update(padded_prediction, padded_target, padded_valid)
    assert padded.as_dict() == together.as_dict()


def test_normalized_softmin_zero_is_zero():
    values = torch.zeros(2, 4)
    torch.testing.assert_close(_softmin(values, 0.05, 1), torch.zeros(2), atol=1e-7, rtol=0)
