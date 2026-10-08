import torch

from ebjd.metrics import hard_metrics, packed_metric_adapter, scene_soft_metrics


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
