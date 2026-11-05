import pickle

import numpy as np
import pytest

from ebjd.legacy_sdd_input_pipeline import (
    augmentation_transform, decompress_zstd_frame,
    load_batch_cache_inprocess, replay_traj_maps_preallocated)


def test_inprocess_zstd_matches_cli(tmp_path):
    import subprocess

    value = {"array": np.arange(96, dtype=np.float32).reshape(8, 12)}
    raw = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    compressed = subprocess.run(
        ["zstd", "-1", "-q", "-c"], input=raw,
        check=True, stdout=subprocess.PIPE).stdout
    assert decompress_zstd_frame(compressed) == raw
    path = tmp_path / "batch.pkl.zst"
    path.write_bytes(compressed)
    loaded = load_batch_cache_inprocess(path)
    assert np.array_equal(loaded["array"], value["array"])


@pytest.mark.parametrize("seed", [0, 1, 3, 14, 18, 21])
def test_preallocated_replay_matches_legacy_concat(seed):
    import albumentations as A

    rng = np.random.default_rng(seed)
    image = rng.normal(size=(19, 23, 3)).astype(np.float32)
    maps = rng.normal(size=(19, 23, 521)).astype(np.float32)
    transform = augmentation_transform()
    transform.set_random_seed(seed)
    sampled = transform(
        image=image, keypoints=[(3.0, 4.0), (12.0, 9.0)])["replay"]
    chunks = []
    for start in range(0, maps.shape[-1], 256):
        chunks.append(A.ReplayCompose.replay(
            sampled, image=np.ascontiguousarray(
                maps[..., start:start + 256]))["image"])
    expected = np.concatenate(chunks, axis=-1)
    actual = replay_traj_maps_preallocated(sampled, maps)
    assert np.array_equal(actual, expected)
