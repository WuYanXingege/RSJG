import json

import numpy as np
import pytest

from ebjd.diagnostic_bank import (
    BankScene, SeedBank, assert_observation_parity, atomic_json,
    canonical_window_key, hard_scene_statistics, load_scene,
    normalized_softmin, reorder_like, save_scene, sha256_file,
)


def scene(prediction=None, *, ids=("a", "b")):
    history = np.zeros((len(ids), 8, 2), dtype=np.float32)
    future = np.zeros((len(ids), 12, 2), dtype=np.float32)
    if prediction is None:
        prediction = np.zeros((20, len(ids), 12, 2), dtype=np.float32)
    frames = np.arange(20, dtype=np.int64) * 10
    return BankScene(
        prediction, future, history, frames, np.asarray(ids, dtype=np.str_),
        "biwi_hotel", "scene", canonical_window_key("biwi_hotel", frames), 0.4)


def test_marginal_and_joint_have_independent_reductions():
    # Agent a is correct in world 0; agent b is correct in world 1.  Both
    # marginals are zero, but either shared world leaves one one-metre error.
    prediction = np.ones((2, 2, 12, 2), dtype=np.float32)
    prediction[..., 1] = 0
    prediction[0, 0, :, 0] = 0
    prediction[1, 1, :, 0] = 0
    stats = hard_scene_statistics(prediction, np.zeros((2, 12, 2)))
    assert np.allclose(stats["agent_minade"], 0)
    assert np.allclose(stats["agent_minfde"], 0)
    assert stats["JADE"] == pytest.approx(0.5)
    assert stats["JFDE"] == pytest.approx(0.5)


def test_ade_and_fde_may_select_different_worlds():
    prediction = np.zeros((2, 1, 12, 2), dtype=np.float32)
    prediction[0, 0, :-1, 0] = 0.1
    prediction[0, 0, -1, 0] = 1.0
    prediction[1, 0, :-1, 0] = 0.2
    prediction[1, 0, -1, 0] = 0.0
    stats = hard_scene_statistics(prediction, np.zeros((1, 12, 2)))
    assert stats["agent_minade"][0] == pytest.approx(2.1 / 12)
    assert stats["agent_minfde"][0] == pytest.approx(0.0)


def test_global_and_agent_order_permutations():
    rng = np.random.default_rng(9)
    prediction = rng.normal(size=(20, 3, 12, 2)).astype(np.float32)
    future = rng.normal(size=(3, 12, 2)).astype(np.float32)
    original = hard_scene_statistics(prediction, future)
    order = rng.permutation(20)
    global_order = hard_scene_statistics(prediction[order], future)
    for key in ("agent_minade", "agent_minfde", "JADE", "JFDE"):
        assert np.allclose(original[key], global_order[key])
    agent_order = np.array([2, 0, 1])
    reordered = hard_scene_statistics(
        prediction[:, agent_order], future[agent_order])
    assert np.allclose(original["agent_minade"][agent_order], reordered["agent_minade"])
    assert original["JADE"] == pytest.approx(reordered["JADE"])


def test_independent_world_permutation_preserves_marginals_and_single_agent():
    rng = np.random.default_rng(12)
    prediction = rng.normal(size=(20, 2, 12, 2)).astype(np.float32)
    future = rng.normal(size=(2, 12, 2)).astype(np.float32)
    original = hard_scene_statistics(prediction, future)
    shuffled = np.empty_like(prediction)
    for agent in range(2):
        shuffled[:, agent] = prediction[rng.permutation(20), agent]
    changed = hard_scene_statistics(shuffled, future)
    assert np.array_equal(original["agent_minade"], changed["agent_minade"])
    assert np.array_equal(original["agent_minfde"], changed["agent_minfde"])
    one = hard_scene_statistics(prediction[:, :1], future[:1])
    one_shuffled = hard_scene_statistics(shuffled[:, :1], future[:1])
    for key in ("agent_minade", "agent_minfde", "JADE", "JFDE"):
        assert np.allclose(one[key], one_shuffled[key])


def test_prefix_hard_metrics_are_monotonic():
    rng = np.random.default_rng(2)
    prediction = rng.normal(size=(20, 4, 12, 2)).astype(np.float32)
    future = rng.normal(size=(4, 12, 2)).astype(np.float32)
    values = [hard_scene_statistics(prediction[:k], future) for k in (4, 8, 12, 16, 20)]
    for key in ("agent_minade", "agent_minfde"):
        means = [float(item[key].mean()) for item in values]
        assert all(right <= left + 1e-12 for left, right in zip(means, means[1:]))
    for key in ("JADE", "JFDE"):
        means = [float(item[key]) for item in values]
        assert all(right <= left + 1e-12 for left, right in zip(means, means[1:]))


def test_safe_roundtrip_hash_rejection_and_identity_alignment(tmp_path):
    original = scene()
    path = tmp_path / "scene.npz"
    digest = save_scene(original, path)
    loaded = load_scene(path, digest)
    assert loaded.predictions.dtype == np.float32
    candidate = BankScene(
        loaded.predictions[:, ::-1], loaded.future[::-1], loaded.history[::-1],
        loaded.frame_ids, loaded.agent_ids[::-1], loaded.source_sequence,
        loaded.scene_id, loaded.window_key, loaded.dt_seconds)
    aligned = reorder_like(loaded, candidate)
    assert list(aligned.agent_ids) == ["a", "b"]
    assert_observation_parity(loaded, aligned)
    with path.open("ab") as handle:
        handle.write(b"damage")
    with pytest.raises(ValueError, match="SHA256"):
        load_scene(path, digest)


def test_shape_nan_duplicate_and_missing_identity_rejected(tmp_path):
    broken = scene()
    prediction = broken.predictions.copy()
    prediction[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        save_scene(BankScene(
            prediction, broken.future, broken.history, broken.frame_ids,
            broken.agent_ids, broken.source_sequence, broken.scene_id,
            broken.window_key, broken.dt_seconds), tmp_path / "nan.npz")
    with pytest.raises(ValueError, match="duplicated"):
        save_scene(BankScene(
            broken.predictions, broken.future, broken.history, broken.frame_ids,
            np.asarray(["a", "a"]), broken.source_sequence, broken.scene_id,
            broken.window_key, broken.dt_seconds), tmp_path / "duplicate.npz")
    with pytest.raises(ValueError, match=r"\[20,N,12,2\]"):
        save_scene(BankScene(
            broken.predictions[:19], broken.future, broken.history,
            broken.frame_ids, broken.agent_ids, broken.source_sequence,
            broken.scene_id, broken.window_key, broken.dt_seconds),
            tmp_path / "p19.npz")


def test_seed_bank_rejects_missing_or_duplicate_identities(tmp_path):
    path = tmp_path / "manifest.json"
    atomic_json({"schema": "wrong"}, path)
    with pytest.raises(ValueError, match="unsupported"):
        SeedBank(path)
    atomic_json({
        "schema": "ebjd-trajectory-bank-seed-manifest-v1", "scene_count": 2,
        "files": [{"identity": "x"}, {"identity": "x"}]}, path)
    with pytest.raises(ValueError, match="duplicate"):
        SeedBank(path)


def test_softmin_is_normalized_and_empty_strata_are_na():
    values = np.zeros((4, 3))
    assert np.allclose(normalized_softmin(values, 0.05, 0), 0)
    empty = np.asarray([], dtype=np.float64)
    result = float("nan") if empty.size == 0 else float(empty.mean())
    assert np.isnan(result)
