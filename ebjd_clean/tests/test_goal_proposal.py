import numpy as np
import pytest

from ebjd.goal_proposal import (
    AgentRecord, effective_unique, paired_delta, pixel_to_local_m,
    split_native_ttst, summarize_rows,
)


def test_align_corners_pixel_to_local_contract():
    pixels = np.asarray([[0, 0], [255, 255], [127.5, 127.5]])
    actual = pixel_to_local_m(pixels)
    np.testing.assert_allclose(actual, [[-16, -16], [16, 16], [0, 0]])


def test_native_ttst_uses_twenty_clusters_and_excludes_argmax():
    samples = np.arange(21 * 2).reshape(21, 1, 1, 2)
    clusters, argmax = split_native_ttst(samples)
    assert clusters.shape == (20, 1, 1, 2)
    np.testing.assert_array_equal(argmax, samples[20])
    with pytest.raises(ValueError):
        split_native_ttst(samples[:20])


def test_effective_unique_uses_metric_tolerance():
    points = np.asarray([[0, 0], [0.00001, 0], [1, 0]])
    assert effective_unique(points) == 2


def _record(route, shift):
    return AgentRecord(
        route, "window", "scene", "source", "agent", 1,
        np.asarray([0.0, 0.0]), np.asarray([0.0, 0.0]),
        np.asarray([[shift, 0.0], [2.0, 0.0]]), np.asarray([shift, 0.0]))


def test_metrics_and_pairing_contract():
    left = _record("A", 1.0).row()
    right = _record("B", 0.5).row()
    summary = summarize_rows([right])
    assert summary["minFDE20_mean_m"] == pytest.approx(0.5)
    assert summary["recall_0p5_mean"] == pytest.approx(1.0)
    delta = paired_delta([left], [right])
    assert delta["mean_m"] == pytest.approx(-0.5)
    assert delta["right_better_fraction"] == pytest.approx(1.0)
