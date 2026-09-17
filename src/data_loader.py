import os
import random
import math
import json

import torch
import numpy as np

from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from torch.utils.data import Subset

from src.data_grouping import batch_cache_path
from src.batch_cache_io import (
    batch_cache_files,
    is_batch_cache_file,
    load_batch_cache,
)
from src.joint_dependency_v2_cache import (
    JDV2_MANIFEST,
    build_manifest,
    jdv2_cache_root,
    load_cache_record,
    stable_json_hash,
    validate_manifest,
)


# OpenCV treats arrays with more than CV_CN_MAX channels as higher-dimensional
# matrices, for which geometric transforms such as ``cv2.flip`` are invalid.
# A legacy GDTS batch can contain B*T=64*20=1280 trajectory-map channels, so
# replay the same sampled augmentation over bounded channel chunks.
CV2_SAFE_CHANNEL_CHUNK = 256
_SOURCE_BLOCK_SPLIT_CACHE = {}
_JDV2_MANIFEST_CACHE = {}


def _validated_jdv2_manifest(args):
    """Validate the explicit cache once per process/configuration."""
    checkpoint = args.jdv2_source_checkpoint or args.pretrain_path
    if checkpoint is None:
        raise RuntimeError(
            'Active JDV2 requires --jdv2_source_checkpoint and an explicit '
            'build-jdv2-cache phase')
    root = jdv2_cache_root(args)
    manifest_path = os.path.join(root, JDV2_MANIFEST)
    key = (manifest_path, os.path.abspath(os.path.expanduser(checkpoint)))
    if key in _JDV2_MANIFEST_CACHE:
        return _JDV2_MANIFEST_CACHE[key]
    try:
        with open(manifest_path, 'r') as handle:
            actual = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
        raise RuntimeError(
            f'Missing/incomplete JDV2 cache manifest at {manifest_path}; '
            'run --phase build-jdv2-cache') from exc
    source_root = batch_cache_path(args)
    source_files = {
        split: batch_cache_files(os.path.join(
            source_root, f'{split}_batches'))
        for split in ('train', 'valid', 'test')}
    expected = build_manifest(
        args, source_files,
        os.path.abspath(os.path.expanduser(checkpoint)),
        completed_splits=('train', 'valid', 'test'))
    validate_manifest(actual, expected)
    if set(actual.get('completed_splits', ())) != {'train', 'valid', 'test'}:
        raise RuntimeError(
            'JDV2 cache is not complete for train/valid/test; rebuild it')
    actual = dict(actual)
    actual['manifest_hash'] = stable_json_hash(actual)
    _JDV2_MANIFEST_CACHE[key] = actual
    return actual


def _batch_source_identifier(dataset, index):
    """Read only the few cache entries needed for a source-boundary search."""
    batch_path = os.path.join(dataset.path_to_folder, dataset.ids[index])
    _, batch_id = load_batch_cache(batch_path)
    return str(batch_id.get('data_file_path', batch_id.get('scene_name', '')))


def _source_block_partition(dataset):
    """Hold out the final complete raw-source block without window leakage.

    Preprocessing writes all synchronized windows from one raw file
    contiguously.  A binary search therefore locates the final file boundary
    with O(log n) cache reads instead of deserializing the entire cache.
    """
    key = (dataset.path_to_folder, len(dataset.ids),
           dataset.ids[-1] if dataset.ids else None)
    if key in _SOURCE_BLOCK_SPLIT_CACHE:
        return _SOURCE_BLOCK_SPLIT_CACHE[key]
    if len(dataset.ids) < 2:
        raise ValueError('source_block validation requires at least two windows')
    final_source = _batch_source_identifier(dataset, len(dataset.ids) - 1)
    low, high = 0, len(dataset.ids) - 1
    while low < high:
        middle = (low + high) // 2
        if _batch_source_identifier(dataset, middle) == final_source:
            high = middle
        else:
            low = middle + 1
    boundary = low
    if boundary <= 0:
        raise ValueError(
            'source_block validation requires more than one raw source file')
    result = (
        list(range(boundary)), list(range(boundary, len(dataset.ids))),
        final_source)
    _SOURCE_BLOCK_SPLIT_CACHE[key] = result
    return result


class dataset_set_name(BaseDataset):
    """
    Dataset class to load iteratively pre-made batches saved as pickle files.
    Apply data augmentation when needed.
    """

    def __init__(self, args, set_name):

        self.path_to_folder = os.path.join(
            batch_cache_path(args), f"{set_name}_batches")

        self.ids = sorted(name for name in os.listdir(self.path_to_folder)
                          if is_batch_cache_file(name))
        self.args = args
        self.set_name = set_name
        self.data_augmentation = args.data_augmentation \
            if set_name == 'train' else False
        self.jdv2_manifest = None
        if (getattr(args, 'goal_model_type', None) == 'joint_dependency_v2'
                and getattr(args, 'jdv2_active', False)
                and getattr(args, 'phase', None) != 'build-jdv2-cache'):
            self.jdv2_manifest = _validated_jdv2_manifest(args)

        print(f"{set_name.title()} dataset contains {len(self.ids)} data "
              f"batches.")

    def __len__(self):
        return len(self.ids)

    def augment_traj_and_images(self, batch_data):

        try:
            import albumentations as A
            import cv2
        except ImportError as exc:
            raise ImportError(
                'Data augmentation requires albumentations and OpenCV; '
                'install them or pass --data_augmentation False') from exc

        image = batch_data["tensor_image"]
        abs_pixel_coord = batch_data["abs_pixel_coord"]
        input_traj_maps = batch_data["input_traj_maps"]

        # images from torch to numpy. float32 is needed by openCV
        image = image.permute(1, 2, 0).numpy().astype('float32')
        # traj_maps to numpy with bs * T channels
        bs, T, old_H, old_W = input_traj_maps.shape
        input_traj_maps = input_traj_maps.view(bs * T, old_H, old_W).\
            permute(1, 2, 0).numpy().astype('float32')
        # keypoints to list of tuples
        # need to clamp because some slightly exit from the image
        abs_pixel_coord[:, :, 0] = np.clip(abs_pixel_coord[:, :, 0],
                                           a_min=0, a_max=old_W - 1e-3)
        abs_pixel_coord[:, :, 1] = np.clip(abs_pixel_coord[:, :, 1],
                                           a_min=0, a_max=old_H - 1e-3)
        keypoints = list(map(tuple, abs_pixel_coord.reshape(-1, 2)))

        transform = A.ReplayCompose([
            # SAFE AUGS, flips and 90rots
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Transpose(p=0.5),
            A.RandomRotate90(p=1.0),

            # HIGH RISKS - HIGH PROBABILITY OF KEYPOINTS GOING OUT
            A.OneOf([  # perspective or shear
                A.Perspective(
                    scale=0.05, pad_mode=cv2.BORDER_CONSTANT, p=1.0),
                A.Affine(
                    shear=(-10, 10), mode=cv2.BORDER_CONSTANT, p=1.0),  # shear
            ], p=0.2),

            A.OneOf([  # translate
                A.ShiftScaleRotate(
                    shift_limit_x=0.01, shift_limit_y=0, scale_limit=0,
                    rotate_limit=0, border_mode=cv2.BORDER_CONSTANT,
                    p=1.0),  # x translations
                A.ShiftScaleRotate(
                    shift_limit_x=0, shift_limit_y=0.01, scale_limit=0,
                    rotate_limit=0, border_mode=cv2.BORDER_CONSTANT,
                    p=1.0),  # y translations
                A.Affine(
                    translate_percent=(0, 0.01),
                    mode=cv2.BORDER_CONSTANT, p=1.0),  # random xy translate
            ], p=0.2),
            # random rotation
            A.Rotate(
                limit=10, border_mode=cv2.BORDER_CONSTANT,
                p=0.4),
        ],
            keypoint_params=A.KeypointParams(format='xy',
                                             remove_invisible=False),
        )
        transformed = transform(image=image, keypoints=keypoints)

        # input_traj_maps: [H,W,B*T]. Replay keeps every trajectory channel
        # spatially aligned with the image/keypoints without passing an array
        # wider than OpenCV's channel limit to any individual transform.
        transformed_traj_chunks = []
        for start in range(0, input_traj_maps.shape[-1],
                           CV2_SAFE_CHANNEL_CHUNK):
            chunk = np.ascontiguousarray(input_traj_maps[
                ..., start:start + CV2_SAFE_CHANNEL_CHUNK])
            replayed = A.ReplayCompose.replay(
                transformed['replay'], image=chunk)
            transformed_traj_chunks.append(replayed['image'])
        transformed_traj_maps = np.concatenate(
            transformed_traj_chunks, axis=-1)

        # FROM NUMPY BACK TO TENSOR
        image = torch.from_numpy(np.ascontiguousarray(
            transformed['image'])).permute(2, 0, 1)
        C, new_H, new_W = image.shape
        abs_pixel_coord = torch.tensor(transformed['keypoints']).\
            view(batch_data["abs_pixel_coord"].shape)
        input_traj_maps = torch.from_numpy(np.ascontiguousarray(
            transformed_traj_maps)).\
            permute(2, 0, 1).view(bs, T, new_H, new_W)

        # NEW AUGMENTATION: INVERT TIME
        if random.random() > 0.5:
            abs_pixel_coord = abs_pixel_coord.flip(dims=(0,))
            input_traj_maps = input_traj_maps.flip(dims=(1,))
            if "frame_ids" in batch_data:
                batch_data["frame_ids"] = batch_data["frame_ids"].flip(dims=(0,))

        batch_data["tensor_image"] = image
        batch_data["abs_pixel_coord"] = abs_pixel_coord
        batch_data["input_traj_maps"] = input_traj_maps

        return batch_data

    def __getitem__(self, i):
        batch_path = os.path.join(self.path_to_folder, self.ids[i])

        batch_data, batch_id = load_batch_cache(batch_path)

        # rescale coordinates wrt pixel coordinates
        batch_data["abs_pixel_coord"] /= self.args.down_factor

        if self.data_augmentation:
            batch_data = self.augment_traj_and_images(batch_data)

        if self.jdv2_manifest is not None:
            cache_dir = os.path.join(jdv2_cache_root(self.args), self.set_name)
            batch_data['jdv2_cache'] = load_cache_record(
                os.path.join(cache_dir, f'{i:06d}.pt'),
                allow_future_supervision=False)
            if (self.set_name == 'train' and
                    self.args.phase in {'train', 'train_test'}):
                batch_data['jdv2_teacher_cache'] = load_cache_record(
                    os.path.join(cache_dir, f'{i:06d}.teacher.pt'),
                    allow_future_supervision=True)

        return batch_data, batch_id


def get_dataloader(args, set_name):
    """
    Create a data loader for a specific set/data split
    """
    assert set_name in ['train', 'valid', 'test']

    shuffle = args.shuffle_train_batches if set_name == 'train' else \
        args.shuffle_test_batches

    # ETH/UCY repositories often mirror ``valid`` and ``test`` for the held-
    # out scene.  V4 must not use that test scene for checkpoint selection, so
    # create one deterministic, disjoint internal split from the synchronized
    # training cache.  The split is over cached scene windows and its exact
    # member names are persisted by the trainer for reproducibility.
    use_internal = (
        getattr(args, 'model_selection_split', 'dataset_valid') ==
        'internal_train' and set_name in {'train', 'valid'})
    source_set = 'train' if use_internal else set_name
    dataset = dataset_set_name(args, set_name=source_set)
    if use_internal:
        indices = list(range(len(dataset)))
        strategy = getattr(
            args, 'internal_validation_strategy', 'window_random')
        heldout_source = None
        if strategy == 'source_block':
            train_indices, valid_indices, heldout_source = \
                _source_block_partition(dataset)
        else:
            splitter = random.Random(int(args.internal_validation_seed))
            splitter.shuffle(indices)
            valid_count = max(
                1, int(math.ceil(
                    len(indices) *
                    float(args.internal_validation_fraction))))
            valid_indices = sorted(indices[:valid_count])
            train_indices = sorted(indices[valid_count:])
        if not train_indices:
            raise ValueError('Internal validation split left no training data')
        chosen = train_indices if set_name == 'train' else valid_indices
        dataset.data_augmentation = (
            bool(args.data_augmentation) if set_name == 'train' else False)
        dataset = Subset(dataset, chosen)
        # Public metadata lets the trainer audit and persist the exact split.
        dataset.source_ids = [dataset.dataset.ids[index] for index in chosen]
        dataset.source_set_name = 'train'
        dataset.logical_set_name = set_name
        dataset.internal_validation_strategy = strategy
        dataset.internal_validation_source = heldout_source
        print(
            f"V4 protocol: {set_name} uses {len(chosen)}/{len(indices)} "
            'disjoint windows from the train cache '
            f"(strategy={strategy}, seed={args.internal_validation_seed}, "
            f"heldout_source={heldout_source}).")

    generator = torch.Generator()
    generator.manual_seed(int(getattr(args, 'seed', 0)))
    loader = DataLoader(dataset, batch_size=1, shuffle=shuffle,
                        num_workers=args.num_workers, generator=generator)

    return loader
