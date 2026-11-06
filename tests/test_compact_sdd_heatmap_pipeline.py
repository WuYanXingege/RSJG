import numpy as np
import torch
import cv2
import pytest

from ebjd.compact_sdd_heatmap_pipeline import (
    AFFINE_MODE,
    MODE,
    _requires_exact_cpu_replay,
    gaussian_maps_gpu,
    replay_maps_gpu,
)


def test_gaussian_matches_legacy_formula():
    centers = torch.tensor([[[4.25, 3.5]], [[8.0, 7.0]]])
    actual = gaussian_maps_gpu(centers, 11, 13)
    yy, xx = torch.meshgrid(torch.arange(11), torch.arange(13), indexing="ij")
    expected = []
    variance = (11 / 64) ** 2
    for point in centers[:, 0]:
        value = torch.exp(-((xx - point[0]).square() +
                            (yy - point[1]).square()) / (2 * variance))
        expected.append(value / value.max())
    expected = torch.stack(expected).unsqueeze(0)
    torch.testing.assert_close(actual, expected)


def test_exact_index_replays_match_numpy():
    maps = torch.arange(2 * 3 * 5 * 7, dtype=torch.float32).view(2, 3, 5, 7)
    replay = {"transforms": [
        {"__class_fullname__": "HorizontalFlip", "applied": True,
         "params": {"shape": (5, 7, 1)}},
        {"__class_fullname__": "Transpose", "applied": True,
         "params": {"shape": (5, 7, 1)}},
        {"__class_fullname__": "RandomRotate90", "applied": True,
         "params": {"factor": 3, "shape": (7, 5, 1)}},
    ]}
    numpy_maps = maps.numpy()
    expected = np.rot90(
        np.transpose(np.flip(numpy_maps, axis=-1), (0, 1, 3, 2)),
        k=3, axes=(-2, -1)).copy()
    expected = torch.from_numpy(expected)
    torch.testing.assert_close(replay_maps_gpu(maps, replay), expected)


def test_affine_replay_tracks_opencv_on_non_square_map():
    maps = gaussian_maps_gpu(
        torch.tensor([[[13.4, 8.7]], [[22.1, 16.2]]]), 23, 37)
    matrix = np.array([
        [0.997, -0.061, 2.3], [0.061, 0.997, -1.7], [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    replay = {"transforms": [{
        "__class_fullname__": "Affine", "applied": True,
        "params": {"matrix": matrix, "output_shape": (23, 37)},
    }]}
    expected = np.stack([
        cv2.warpPerspective(
            item.numpy(), matrix, (37, 23), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        for item in maps.reshape(-1, 23, 37)
    ]).reshape(tuple(maps.shape))
    difference = (replay_maps_gpu(maps, replay) -
                  torch.from_numpy(expected)).abs()
    assert float(difference.mean()) < 1e-5
    assert float(difference.max()) < 1e-2


@pytest.mark.parametrize("name", ["Affine", "ShiftScaleRotate", "Rotate"])
def test_supported_affine_families_track_opencv(name):
    maps = gaussian_maps_gpu(
        torch.tensor([[[3.1, 2.8], [34.7, 20.4]],
                      [[18.2, 10.6], [1.2, 21.1]]]), 23, 37)
    matrix = np.array([
        [0.9914, 0.1305, -1.3], [-0.1305, 0.9914, 2.1],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    replay = {"transforms": [{
        "__class_fullname__": name, "applied": True,
        "params": {"matrix": matrix, "output_shape": (23, 37)},
    }]}
    expected = np.stack([
        cv2.warpPerspective(
            item.numpy(), matrix, (37, 23), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        for item in maps.reshape(-1, 23, 37)
    ]).reshape(tuple(maps.shape))
    difference = (replay_maps_gpu(maps, replay) -
                  torch.from_numpy(expected)).abs()
    assert float(difference.mean()) < 3e-6
    assert float(difference.max()) < 2e-2


def test_multiple_resampling_operations_keep_execution_order():
    maps = gaussian_maps_gpu(torch.tensor([[[12.2, 8.4]]]), 21, 35)
    first = np.array([
        [1.0, 0.07, -0.7], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    second = np.array([
        [0.995, -0.1, 1.2], [0.1, 0.995, -1.1], [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    replay = {"transforms": [
        {"__class_fullname__": "Affine", "applied": True,
         "params": {"matrix": first, "output_shape": (21, 35)}},
        {"__class_fullname__": "Rotate", "applied": True,
         "params": {"matrix": second, "output_shape": (21, 35)}},
    ]}
    expected = maps.reshape(-1, 21, 35).numpy()
    for matrix in (first, second):
        expected = np.stack([
            cv2.warpPerspective(
                item, matrix, (35, 21), flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            for item in expected])
    actual = replay_maps_gpu(maps, replay).reshape(-1, 21, 35)
    difference = (actual - torch.from_numpy(expected)).abs()
    assert float(difference.mean()) < 4e-6
    assert float(difference.max()) < 2e-2


def test_v4_only_keeps_perspective_on_cpu_fallback():
    affine = {"transforms": [{
        "__class_fullname__": "Affine", "applied": True,
        "params": {"matrix": np.eye(3), "output_shape": (5, 7)},
    }]}
    perspective = {"transforms": [{
        "__class_fullname__": "Perspective", "applied": True,
        "params": {"matrix": np.eye(3), "max_height": 5,
                   "max_width": 7},
        "keep_size": True,
    }]}
    assert _requires_exact_cpu_replay(affine, MODE)
    assert not _requires_exact_cpu_replay(affine, AFFINE_MODE)
    assert _requires_exact_cpu_replay(perspective, AFFINE_MODE)
