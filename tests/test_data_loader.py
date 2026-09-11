import random
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

import src.data_loader as data_loader_module
from src.data_loader import dataset_set_name


class _TransformPlaceholder:
    def __init__(self, *args, **kwargs):
        del args, kwargs


class _IdentityReplayCompose:
    def __init__(self, transforms, **kwargs):
        self.transforms = transforms
        self.kwargs = kwargs

    def __call__(self, image, keypoints):
        return {'image': image, 'keypoints': keypoints, 'replay': {}}

    @staticmethod
    def replay(replay, image):
        del replay
        return {'image': image}


def test_legacy_augmentation_uses_stable_albumentations_top_level_api(
        monkeypatch):
    """The baseline augmentation path resolves against Albumentations 1.x."""
    albumentations = ModuleType('albumentations')
    for name in (
            'HorizontalFlip', 'VerticalFlip', 'Transpose', 'RandomRotate90',
            'Perspective', 'Affine', 'ShiftScaleRotate', 'Rotate', 'OneOf',
            'KeypointParams'):
        setattr(albumentations, name, _TransformPlaceholder)
    albumentations.ReplayCompose = _IdentityReplayCompose
    cv2 = ModuleType('cv2')
    cv2.BORDER_CONSTANT = 0
    monkeypatch.setitem(sys.modules, 'albumentations', albumentations)
    monkeypatch.setitem(sys.modules, 'cv2', cv2)
    monkeypatch.setattr(random, 'random', lambda: 0.0)

    loader = object.__new__(dataset_set_name)
    batch = {
        'tensor_image': torch.zeros(6, 8, 8),
        'abs_pixel_coord': torch.tensor([
            [[1.0, 1.0], [2.0, 2.0]],
            [[2.0, 1.0], [3.0, 2.0]],
        ]),
        'input_traj_maps': torch.zeros(2, 2, 8, 8),
    }
    output = loader.augment_traj_and_images(batch)

    assert output['tensor_image'].shape == (6, 8, 8)
    assert output['abs_pixel_coord'].shape == (2, 2, 2)
    assert output['input_traj_maps'].shape == (2, 2, 8, 8)
    assert np.isfinite(output['abs_pixel_coord'].numpy()).all()


def test_real_augmentation_chunks_large_legacy_map_tensor(monkeypatch):
    """B*T channels above OpenCV's limit share one replayed transform."""
    pytest.importorskip('albumentations')
    monkeypatch.setattr(random, 'random', lambda: 0.0)

    loader = object.__new__(dataset_set_name)
    batch_size, timesteps, height, width = 32, 20, 8, 8
    batch = {
        'tensor_image': torch.zeros(6, height, width),
        'abs_pixel_coord': torch.ones(timesteps, batch_size, 2),
        'input_traj_maps': torch.zeros(
            batch_size, timesteps, height, width),
    }

    output = loader.augment_traj_and_images(batch)

    assert output['tensor_image'].shape == (6, height, width)
    assert output['abs_pixel_coord'].shape == (timesteps, batch_size, 2)
    assert output['input_traj_maps'].shape == (
        batch_size, timesteps, height, width)
    assert torch.isfinite(output['input_traj_maps']).all()


def test_internal_validation_uses_reproducible_disjoint_train_windows(
        tmp_path, monkeypatch):
    cache = tmp_path / 'cache'
    for split in ('train', 'valid', 'test'):
        (cache / f'{split}_batches').mkdir(parents=True)
    for index in range(20):
        (cache / 'train_batches' /
         f'train_batch_{index:04d}.pkl').touch()
    monkeypatch.setattr(
        data_loader_module, 'batch_cache_path', lambda _args: str(cache))
    args = SimpleNamespace(
        model_selection_split='internal_train',
        internal_validation_seed=37,
        internal_validation_fraction=0.2,
        internal_validation_strategy='window_random',
        data_augmentation=False,
        shuffle_train_batches=False,
        shuffle_test_batches=False,
        num_workers=0,
        seed=11,
    )
    train = data_loader_module.get_dataloader(args, 'train')
    valid = data_loader_module.get_dataloader(args, 'valid')
    train_ids = set(train.dataset.source_ids)
    valid_ids = set(valid.dataset.source_ids)
    assert len(train_ids) == 16
    assert len(valid_ids) == 4
    assert not train_ids & valid_ids
    assert len(train_ids | valid_ids) == 20


def test_source_block_validation_holds_out_complete_final_source(
        tmp_path, monkeypatch):
    cache = tmp_path / 'cache'
    for split in ('train', 'valid', 'test'):
        (cache / f'{split}_batches').mkdir(parents=True)
    for index in range(10):
        (cache / 'train_batches' /
         f'train_batch_{index:04d}.pkl').touch()
    monkeypatch.setattr(
        data_loader_module, 'batch_cache_path', lambda _args: str(cache))
    monkeypatch.setattr(
        data_loader_module, '_batch_source_identifier',
        lambda _dataset, index: 'source_a' if index < 7 else 'source_b')
    data_loader_module._SOURCE_BLOCK_SPLIT_CACHE.clear()
    args = SimpleNamespace(
        model_selection_split='internal_train',
        internal_validation_seed=37,
        internal_validation_fraction=0.2,
        internal_validation_strategy='source_block',
        data_augmentation=False,
        shuffle_train_batches=False,
        shuffle_test_batches=False,
        num_workers=0,
        seed=11,
    )
    train = data_loader_module.get_dataloader(args, 'train')
    valid = data_loader_module.get_dataloader(args, 'valid')
    assert train.dataset.source_ids == [
        f'train_batch_{index:04d}.pkl' for index in range(7)]
    assert valid.dataset.source_ids == [
        f'train_batch_{index:04d}.pkl' for index in range(7, 10)]
    assert valid.dataset.internal_validation_source == 'source_b'
