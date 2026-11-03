import numpy as np

from ebjd.export_sdd import semantic_crops, synchronized_windows


def test_synchronized_sdd_windows_keep_complete_agents(tmp_path):
    rows = []
    for time in range(21):
        frame = time * 12
        rows.append((frame, 7, time, time + 1))
        if time < 20:
            rows.append((frame, 9, 3 * time, 2 * time))
    path = tmp_path / "scene.txt"
    np.savetxt(path, np.asarray(rows), fmt="%.8f")
    windows = synchronized_windows(path, 0.04)
    assert len(windows) == 2
    assert windows[0]["agent_ids"] == [7, 9]
    assert windows[1]["agent_ids"] == [7]
    np.testing.assert_allclose(
        windows[0]["coordinates"], windows[0]["pixels"] * 0.04)


def test_sdd_semantic_crop_is_six_channel_one_hot_or_oob():
    labels = np.arange(36, dtype=np.uint8).reshape(6, 6) % 6
    crop = semantic_crops(np.asarray([[2.0, 2.0]]), labels, 1.0, 4.0, 9)
    assert crop.shape == (1, 6, 9, 9)
    assert np.isin(crop, (0, 1)).all()
    assert (crop.sum(axis=1) <= 1).all()
