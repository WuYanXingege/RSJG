"""Compact SDD cache and main-process GPU trajectory-map materialization.

The cache omits the dense ``[agent,time,height,width]`` Gaussian tensor.  A
worker still samples the original Albumentations replay and applies it to the
scene image/keypoints.  The training process reconstructs the *original*
Gaussian maps on the target device and replays the same spatial operations.
"""

from __future__ import annotations

import os
import pickle
import random
import hashlib
import math
from pathlib import Path

import numpy as np
import torch

from .legacy_sdd_input_pipeline import (
    augmentation_transform,
    load_batch_cache_inprocess,
    replay_traj_maps_preallocated,
)


MODE = "compact_gpu_heatmaps_v1"
SCHEMA = "ebjd-sdd-compact-heatmap-cache-v1"
_INTERNAL_PREFIX = "_paper_compact_"


def _fixed_seed(coordinates):
    base = os.environ.get("EBJD_SDD_FIXED_AUGMENTATION_SEED")
    if base is None:
        return None
    payload = np.ascontiguousarray(coordinates).view(np.uint8)
    digest = hashlib.sha256(payload).digest()
    return (int(base) + int.from_bytes(digest[:8], "little")) % (2 ** 32)


def _applied_transforms(replay):
    for item in replay.get("transforms", []):
        children = item.get("transforms")
        if children is not None:
            yield from _applied_transforms(item)
        elif item.get("applied", False):
            yield item


def _requires_exact_cpu_replay(replay):
    resampling = {"Perspective", "Affine", "ShiftScaleRotate", "Rotate"}
    return any(
        item["__class_fullname__"].split(".")[-1] in resampling
        for item in _applied_transforms(replay))


def gaussian_maps_gpu(centers, height, width, chunk_size=256):
    """Reproduce the legacy un-normalized Gaussian maps in FP32."""
    if centers.ndim != 3 or centers.shape[-1] != 2:
        raise ValueError("centers must have shape [T,B,2]")
    centers = centers.float()
    steps, agents = centers.shape[:2]
    flat = centers.permute(1, 0, 2).reshape(-1, 2)
    ys = torch.arange(height, device=centers.device)
    xs = torch.arange(width, device=centers.device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    variance = float(min(width, height) / 64.0) ** 2
    output = torch.empty(
        (flat.shape[0], height, width), device=centers.device,
        dtype=torch.float32)
    for start in range(0, flat.shape[0], chunk_size):
        stop = min(start + chunk_size, flat.shape[0])
        point = flat[start:stop]
        distance = ((xx[None] - point[:, 0, None, None]).square() +
                    (yy[None] - point[:, 1, None, None]).square())
        values = ((1.0 / (2.0 * math.pi * variance)) *
                  torch.exp(-distance / (2.0 * variance)))
        values = values / values.flatten(1).amax(1).clamp_min(
            torch.finfo(values.dtype).tiny)[:, None, None]
        output[start:stop] = values
    return output.view(agents, steps, height, width)


def _warp_perspective(images, matrix, output_hw):
    """OpenCV-style forward pixel homography via CUDA grid_sample."""
    height, width = images.shape[-2:]
    out_height, out_width = map(int, output_hw)
    device = images.device
    matrix = torch.as_tensor(matrix, device=device, dtype=torch.float64)
    inverse = torch.linalg.inv(matrix)
    ys = torch.arange(out_height, device=device, dtype=torch.float64)
    xs = torch.arange(out_width, device=device, dtype=torch.float64)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    homogeneous = torch.stack(
        (xx, yy, torch.ones_like(xx)), dim=-1).reshape(-1, 3).T
    source = inverse @ homogeneous
    source = source[:2] / source[2:].clamp_min(1e-12)
    # OpenCV INTER_LINEAR uses a 5-bit interpolation table.  Quantizing the
    # inverse-map coordinates to 1/32 reproduces that established raster
    # convention while grid_sample performs the batched CUDA interpolation.
    source = torch.round(source * 32.0) / 32.0
    source_x, source_y = source[0], source[1]
    x0 = torch.floor(source_x).long()
    y0 = torch.floor(source_y).long()
    fx = (source_x - x0).float()
    fy = (source_y - y0).float()
    flat = images.reshape(images.shape[0], -1)
    result = torch.zeros(
        (images.shape[0], source_x.numel()), device=device,
        dtype=images.dtype)
    for x, y, weight in (
            (x0, y0, (1 - fx) * (1 - fy)),
            (x0 + 1, y0, fx * (1 - fy)),
            (x0, y0 + 1, (1 - fx) * fy),
            (x0 + 1, y0 + 1, fx * fy)):
        valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        index = y.clamp(0, height - 1) * width + x.clamp(0, width - 1)
        result.add_(flat[:, index] * (weight * valid).to(images.dtype))
    return result.view(images.shape[0], out_height, out_width)


def replay_maps_gpu(maps, replay):
    """Replay the legacy spatial transform sequence on ``[B,T,H,W]`` maps."""
    agents, steps = maps.shape[:2]
    values = maps.reshape(agents * steps, *maps.shape[-2:])
    for transform in _applied_transforms(replay):
        name = transform["__class_fullname__"].split(".")[-1]
        params = transform.get("params") or {}
        if name == "HorizontalFlip":
            values = values.flip(-1)
        elif name == "VerticalFlip":
            values = values.flip(-2)
        elif name == "Transpose":
            values = values.transpose(-2, -1)
        elif name == "RandomRotate90":
            values = torch.rot90(
                values, int(params["factor"]), dims=(-2, -1))
        elif name == "Perspective":
            height, width = values.shape[-2:]
            matrix = np.asarray(params["matrix"], dtype=np.float64)
            if bool(transform.get("keep_size", True)):
                scale = np.diag([
                    width / float(params["max_width"]),
                    height / float(params["max_height"]), 1.0])
                matrix = scale @ matrix
            values = _warp_perspective(values, matrix, (height, width))
        elif name in {"Affine", "ShiftScaleRotate", "Rotate"}:
            matrix = np.asarray(params["matrix"], dtype=np.float64)
            output_shape = params.get("output_shape", values.shape[-2:])
            values = _warp_perspective(values, matrix, output_shape)
        else:
            raise RuntimeError(f"unsupported applied SDD transform: {name}")
    return values.reshape(agents, steps, *values.shape[-2:])


def augment_compact(self, batch_data):
    image = batch_data["tensor_image"].permute(1, 2, 0).numpy().astype(
        np.float32, copy=False)
    original = np.asarray(batch_data["abs_pixel_coord"]).copy()
    height, width = image.shape[:2]
    keypoint_coords = original.copy()
    keypoint_coords[:, :, 0] = np.clip(
        keypoint_coords[:, :, 0], 0, width - 1e-3)
    keypoint_coords[:, :, 1] = np.clip(
        keypoint_coords[:, :, 1], 0, height - 1e-3)
    transform = getattr(self, "_paper_compact_transform", None)
    if transform is None:
        transform = augmentation_transform()
        self._paper_compact_transform = transform
    fixed_seed = _fixed_seed(keypoint_coords)
    if fixed_seed is not None:
        transform.set_random_seed(fixed_seed)
    transformed = transform(
        image=image,
        keypoints=list(map(tuple, keypoint_coords.reshape(-1, 2))))
    coordinates = torch.tensor(transformed["keypoints"]).view(original.shape)
    exact_maps = None
    if _requires_exact_cpu_replay(transformed["replay"]):
        legacy_path = batch_data[f"{_INTERNAL_PREFIX}legacy_path"]
        legacy_batch, _ = load_batch_cache_inprocess(legacy_path)
        legacy_maps = legacy_batch["input_traj_maps"]
        agents, steps, map_height, map_width = legacy_maps.shape
        channels = legacy_maps.view(
            agents * steps, map_height, map_width).permute(1, 2, 0).numpy()
        replayed = replay_traj_maps_preallocated(
            transformed["replay"], channels)
        exact_maps = torch.from_numpy(np.ascontiguousarray(replayed)).permute(
            2, 0, 1).view(agents, steps, *replayed.shape[:2])
    reverse_time = (
        random.Random(fixed_seed ^ 0x5A17).random() > 0.5
        if fixed_seed is not None else random.random() > 0.5)
    if reverse_time:
        coordinates = coordinates.flip(0)
        if exact_maps is not None:
            exact_maps = exact_maps.flip(1)
        if "frame_ids" in batch_data:
            batch_data["frame_ids"] = batch_data["frame_ids"].flip(0)
    batch_data["tensor_image"] = torch.from_numpy(np.ascontiguousarray(
        transformed["image"])).permute(2, 0, 1)
    batch_data["abs_pixel_coord"] = coordinates
    batch_data[f"{_INTERNAL_PREFIX}centers"] = torch.from_numpy(original)
    batch_data[f"{_INTERNAL_PREFIX}replay"] = pickle.dumps(
        transformed["replay"], protocol=pickle.HIGHEST_PROTOCOL)
    batch_data[f"{_INTERNAL_PREFIX}reverse"] = reverse_time
    if exact_maps is not None:
        batch_data["input_traj_maps"] = exact_maps
        batch_data[f"{_INTERNAL_PREFIX}exact_maps"] = True
    return batch_data


def augment_legacy_fixed(self, batch_data):
    """Benchmark-only legacy raster path with batch-identity-fixed replay."""
    image = batch_data["tensor_image"].permute(1, 2, 0).numpy().astype(
        np.float32, copy=False)
    coordinates = np.asarray(batch_data["abs_pixel_coord"])
    maps = batch_data["input_traj_maps"]
    agents, steps, height, width = maps.shape
    channels = maps.view(agents * steps, height, width).permute(
        1, 2, 0).numpy().astype(np.float32, copy=False)
    coordinates[:, :, 0] = np.clip(coordinates[:, :, 0], 0, width - 1e-3)
    coordinates[:, :, 1] = np.clip(coordinates[:, :, 1], 0, height - 1e-3)
    transform = augmentation_transform()
    fixed_seed = _fixed_seed(coordinates)
    if fixed_seed is None:
        raise RuntimeError("fixed legacy augmentation requires a seed")
    transform.set_random_seed(fixed_seed)
    transformed = transform(
        image=image,
        keypoints=list(map(tuple, coordinates.reshape(-1, 2))))
    transformed_maps = replay_traj_maps_preallocated(
        transformed["replay"], channels)
    batch_data["tensor_image"] = torch.from_numpy(np.ascontiguousarray(
        transformed["image"])).permute(2, 0, 1)
    batch_data["abs_pixel_coord"] = torch.tensor(
        transformed["keypoints"]).view(coordinates.shape)
    new_height, new_width = transformed_maps.shape[:2]
    batch_data["input_traj_maps"] = torch.from_numpy(np.ascontiguousarray(
        transformed_maps)).permute(2, 0, 1).view(
            agents, steps, new_height, new_width)
    if random.Random(fixed_seed ^ 0x5A17).random() > 0.5:
        batch_data["abs_pixel_coord"] = batch_data["abs_pixel_coord"].flip(0)
        batch_data["input_traj_maps"] = batch_data["input_traj_maps"].flip(1)
        if "frame_ids" in batch_data:
            batch_data["frame_ids"] = batch_data["frame_ids"].flip(0)
    return batch_data


def install_fixed_legacy_augmentation(data_loader_module):
    if "EBJD_SDD_FIXED_AUGMENTATION_SEED" not in os.environ:
        raise RuntimeError("fixed augmentation seed environment is absent")
    data_loader_module.dataset_set_name.augment_traj_and_images = (
        augment_legacy_fixed)


def install(data_loader_module, model_class, compact_cache_root):
    """Install the compact SDD loader and device materialization hook."""
    compact_cache_root = Path(compact_cache_root).resolve()
    old_init = data_loader_module.dataset_set_name.__init__
    old_prepare = model_class.prepare_inputs

    def compact_init(self, args, set_name, logical_role=None):
        old_init(self, args, set_name, logical_role=logical_role)
        self._paper_legacy_path_to_folder = self.path_to_folder
        folder = compact_cache_root / f"{set_name}_batches"
        if not folder.is_dir():
            raise FileNotFoundError(f"compact cache split missing: {folder}")
        self.path_to_folder = os.fspath(folder)
        self.ids = sorted(
            name for name in os.listdir(folder)
            if name.endswith((".pkl", ".pkl.zst")))

    def compact_getitem(self, index):
        if self.jdv2_manifest is not None:
            raise RuntimeError("compact SDD heatmaps are independent-GDTS only")
        batch_path = os.path.join(self.path_to_folder, self.ids[index])
        batch_data, batch_id = load_batch_cache_inprocess(batch_path)
        batch_data["abs_pixel_coord"] /= self.args.down_factor
        batch_data[f"{_INTERNAL_PREFIX}legacy_path"] = os.path.join(
            self._paper_legacy_path_to_folder, self.ids[index])
        if self.data_augmentation:
            batch_data = self.augment_traj_and_images(batch_data)
        if f"{_INTERNAL_PREFIX}centers" not in batch_data:
            centers = batch_data["abs_pixel_coord"]
            if torch.is_tensor(centers):
                centers = centers.clone()
            else:
                centers = np.asarray(centers).copy()
            batch_data[f"{_INTERNAL_PREFIX}centers"] = centers
            batch_data[f"{_INTERNAL_PREFIX}replay"] = b""
            batch_data[f"{_INTERNAL_PREFIX}reverse"] = False
        return batch_data, batch_id

    def compact_prepare(self, batch_data, batch_id):
        centers = batch_data.pop(f"{_INTERNAL_PREFIX}centers").squeeze(0)
        replay_value = batch_data.pop(f"{_INTERNAL_PREFIX}replay")
        reverse_value = batch_data.pop(f"{_INTERNAL_PREFIX}reverse")
        batch_data.pop("_paper_compact_schema", None)
        batch_data.pop(f"{_INTERNAL_PREFIX}legacy_path", None)
        original_hw = batch_data.pop("_paper_compact_original_hw")
        if isinstance(original_hw, (list, tuple)):
            height, width = (
                int(value.item()) if torch.is_tensor(value) else int(value)
                for value in original_hw)
        else:
            height, width = map(int, original_hw.squeeze(0).tolist())
        centers = centers.to(self.device, dtype=torch.float32,
                             non_blocking=True)
        exact_maps = batch_data.pop(f"{_INTERNAL_PREFIX}exact_maps", None)
        with torch.no_grad():
            maps = (
                batch_data["input_traj_maps"].squeeze(0).to(
                    self.device, dtype=torch.float32, non_blocking=True)
                if exact_maps is not None else
                gaussian_maps_gpu(centers, height, width))
            replay_bytes = replay_value[0] if isinstance(
                replay_value, (list, tuple)) else replay_value
            if replay_bytes and exact_maps is None:
                maps = replay_maps_gpu(maps, pickle.loads(replay_bytes))
            reverse = bool(reverse_value.item()) if torch.is_tensor(
                reverse_value) else bool(reverse_value[0]) if isinstance(
                    reverse_value, (list, tuple)) else bool(reverse_value)
            if reverse and exact_maps is None:
                maps = maps.flip(1)
            maps = maps.contiguous()
        expected_hw = tuple(batch_data["tensor_image"].shape[-2:])
        if tuple(maps.shape[-2:]) != expected_hw:
            raise RuntimeError(
                f"trajectory-map/image shape mismatch: "
                f"maps={tuple(maps.shape[-2:])}, image={expected_hw}")
        batch_data["input_traj_maps"] = maps.unsqueeze(0)
        return old_prepare(self, batch_data, batch_id)

    data_loader_module.load_batch_cache = load_batch_cache_inprocess
    data_loader_module.dataset_set_name.__init__ = compact_init
    data_loader_module.dataset_set_name.__getitem__ = compact_getitem
    data_loader_module.dataset_set_name.augment_traj_and_images = augment_compact
    model_class.prepare_inputs = compact_prepare
    data_loader_module._paper_sdd_input_pipeline = MODE

