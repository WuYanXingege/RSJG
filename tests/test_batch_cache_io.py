import os

import numpy as np
import pytest
import torch

from src.batch_cache_io import dump_batch_cache, load_batch_cache


def test_zstd_batch_cache_round_trip_is_exact(tmp_path):
    value = ({
        'tensor': torch.linspace(0, 1, 4096, dtype=torch.float32),
        'array': np.arange(128, dtype=np.int64),
    }, {'scene_name': 'scene'})
    path = tmp_path / 'batch.pkl.zst'

    try:
        dump_batch_cache(value, str(path), compressed=True)
    except RuntimeError as exc:
        if 'zstd executable' in str(exc):
            pytest.skip(str(exc))
        raise
    loaded = load_batch_cache(str(path))

    assert torch.equal(loaded[0]['tensor'], value[0]['tensor'])
    assert np.array_equal(loaded[0]['array'], value[0]['array'])
    assert loaded[1] == value[1]
    assert os.path.getsize(path) > 0
