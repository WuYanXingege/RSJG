import numpy as np
import pytest

from ebjd.diagnostic_bank import BankScene
from ebjd.endpoint_calibration import (
    bridge_residual, calibrate_endpoint_spread, summarize_factor,
)


def _scene() -> BankScene:
    rng = np.random.default_rng(19)
    predictions = rng.normal(size=(20, 2, 12, 2)).astype(np.float32)
    future = rng.normal(size=(2, 12, 2)).astype(np.float32)
    history = rng.normal(size=(2, 8, 2)).astype(np.float32)
    return BankScene(
        predictions, future, history, np.arange(20, dtype=np.int64),
        np.asarray(["a", "b"]), "toy", "toy-scene", "toy:0-19", 0.4)


def test_factor_one_is_literal_identity():
    scene = _scene()
    calibrated = calibrate_endpoint_spread(scene.predictions, 1.0)
    assert np.array_equal(calibrated, scene.predictions.astype(np.float64))


@pytest.mark.parametrize("factor", [1.25, 1.5, 2.0])
def test_mean_radius_and_bridge_invariants(factor):
    scene = _scene()
    before = scene.predictions.astype(np.float64)
    after = calibrate_endpoint_spread(before, factor)
    assert np.max(np.abs(after.mean(axis=0) - before.mean(axis=0))) < 1e-14
    before_endpoint = before[..., -1, :]
    after_endpoint = after[..., -1, :]
    before_radius = np.sqrt(np.square(
        before_endpoint - before_endpoint.mean(axis=0, keepdims=True)).mean())
    after_radius = np.sqrt(np.square(
        after_endpoint - after_endpoint.mean(axis=0, keepdims=True)).mean())
    assert after_radius / before_radius == pytest.approx(factor, abs=1e-14)
    assert np.max(np.abs(
        bridge_residual(after, scene.history)
        - bridge_residual(before, scene.history))) < 1e-14


def test_nested_k_curve_is_monotonic():
    result = summarize_factor([_scene()], 1.5)
    rows = result["k_curve"]
    for metric in ("minADE", "minFDE", "JADE", "JFDE"):
        values = [row[metric] for row in rows]
        assert all(right <= left + 1e-12
                   for left, right in zip(values, values[1:]))
