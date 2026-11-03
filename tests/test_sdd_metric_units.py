import torch

from ebjd.data import SceneBatch
from ebjd.metrics import hard_metrics


def test_per_scene_paper_metric_scale_converts_metres_to_pixels():
    prediction = torch.zeros(2, 1, 1, 12, 2)
    target = torch.zeros(2, 1, 12, 2)
    prediction[:, :, :, :, 0] = 0.5
    batch = SceneBatch(
        torch.zeros(2, 1, 8, 2), target, torch.zeros(2, 1, 6, 4, 4),
        torch.ones(2, 1, dtype=torch.bool), ["a", "b"], [["0"], ["1"]],
        [list(range(20)), list(range(20))],
        [[0.4 * x for x in range(20)]] * 2, ["a", "b"], {},
        torch.tensor([2.0, 4.0]))
    converted, converted_target = batch.paper_coordinates(prediction, target)
    summary = hard_metrics(converted, converted_target, batch.valid)
    assert summary.minADE == 1.5
    assert summary.minFDE == 1.5
    assert summary.JADE == 1.5
    assert summary.JFDE == 1.5


def test_default_scale_preserves_existing_fold_values():
    prediction = torch.randn(1, 2, 3, 12, 2)
    target = torch.randn(1, 3, 12, 2)
    batch = SceneBatch(
        torch.zeros(1, 3, 8, 2), target, torch.zeros(1, 3, 6, 4, 4),
        torch.ones(1, 3, dtype=torch.bool), ["a"], [["0", "1", "2"]],
        [list(range(20))], [[0.4 * x for x in range(20)]], ["a"], {})
    converted, converted_target = batch.paper_coordinates(prediction, target)
    assert converted is prediction
    assert converted_target is target
