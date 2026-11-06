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
import json
import tempfile
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from .legacy_sdd_input_pipeline import (
    augmentation_transform,
    load_batch_cache_inprocess,
    replay_traj_maps_preallocated,
)


MODE = "compact_gpu_heatmaps_v1"
AFFINE_MODE = "compact_gpu_heatmaps_v2_affine"
SCHEMA = "ebjd-sdd-compact-heatmap-cache-v1"
_INTERNAL_PREFIX = "_paper_compact_"
_PIPELINE_STATS = {
    "total_batches": 0,
    "gpu_fast_batches": 0,
    "cpu_fallback_batches": 0,
    "worker_prepare_seconds": 0.0,
    "worker_prepare_seconds_gpu_fast": 0.0,
    "worker_prepare_seconds_cpu_fallback": 0.0,
    "applied_operations": {},
    "fallback_reasons": {},
    "map_sizes": {},
}


def pipeline_statistics_snapshot():
    return deepcopy(_PIPELINE_STATS)


def pipeline_statistics_delta(before, after):
    result = {}
    for key in (
            "total_batches", "gpu_fast_batches", "cpu_fallback_batches",
            "worker_prepare_seconds", "worker_prepare_seconds_gpu_fast",
            "worker_prepare_seconds_cpu_fallback"):
        result[key] = after[key] - before[key]
    for key in ("applied_operations", "fallback_reasons", "map_sizes"):
        names = set(before[key]) | set(after[key])
        result[key] = {
            name: after[key].get(name, 0) - before[key].get(name, 0)
            for name in sorted(names)
            if after[key].get(name, 0) != before[key].get(name, 0)
        }
    return result


def _atomic_json(payload, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=destination.name + ".", suffix=".tmp",
        dir=destination.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def append_epoch_statistics(destination, epoch, statistics):
    destination = Path(destination)
    payload = {
        "schema": "ebjd-sdd-input-pipeline-epoch-statistics-v1",
        "epochs": [],
    }
    if destination.is_file():
        payload = json.loads(destination.read_text(encoding="utf-8"))
    payload["epochs"] = [
        row for row in payload.get("epochs", [])
        if int(row["epoch"]) != int(epoch)
    ]
    payload["epochs"].append({"epoch": int(epoch), **statistics})
    payload["epochs"].sort(key=lambda row: int(row["epoch"]))
    _atomic_json(payload, destination)


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


def _requires_exact_cpu_replay(replay, mode=MODE):
    return bool(_cpu_fallback_reasons(replay, mode))


def _cpu_fallback_reasons(replay, mode=MODE):
    unsupported = (
        {"Perspective"} if mode == AFFINE_MODE else
        {"Perspective", "Affine", "ShiftScaleRotate", "Rotate"})
    return sorted({
        item["__class_fullname__"].split(".")[-1]
        for item in _applied_transforms(replay)
        if item["__class_fullname__"].split(".")[-1] in unsupported
    })


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


def _warp_perspective(images, matrix, output_hw, channel_chunk_size=128):
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
    result = torch.zeros(
        (images.shape[0], source_x.numel()), device=device,
        dtype=images.dtype)
    sampling_plan = []
    for x, y, weight in (
            (x0, y0, (1 - fx) * (1 - fy)),
            (x0 + 1, y0, fx * (1 - fy)),
            (x0, y0 + 1, (1 - fx) * fy),
            (x0 + 1, y0 + 1, fx * fy)):
        valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        index = y.clamp(0, height - 1) * width + x.clamp(0, width - 1)
        sampling_plan.append(
            (index, (weight * valid).to(images.dtype)))
    flat = images.reshape(images.shape[0], -1)
    for start in range(0, images.shape[0], channel_chunk_size):
        stop = min(start + channel_chunk_size, images.shape[0])
        output = result[start:stop]
        source_values = flat[start:stop]
        for index, weight in sampling_plan:
            output.add_(source_values[:, index] * weight)
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
    worker_started = time.perf_counter()
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
    applied_operations = [
        item["__class_fullname__"].split(".")[-1]
        for item in _applied_transforms(transformed["replay"])]
    fallback_reasons = _cpu_fallback_reasons(
        transformed["replay"], self._paper_compact_mode)
    exact_maps = None
    if _requires_exact_cpu_replay(
            transformed["replay"], self._paper_compact_mode):
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
    route = "cpu_fallback" if exact_maps is not None else "gpu_fast"
    batch_data[f"{_INTERNAL_PREFIX}statistics"] = pickle.dumps({
        "route": route,
        "worker_prepare_seconds": time.perf_counter() - worker_started,
        "applied_operations": applied_operations,
        "fallback_reasons": (
            fallback_reasons if exact_maps is not None else []),
        "map_size": f"{height}x{width}",
    }, protocol=pickle.HIGHEST_PROTOCOL)
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


def install(data_loader_module, model_class, compact_cache_root, mode=MODE):
    """Install the compact SDD loader and device materialization hook."""
    if mode not in {MODE, AFFINE_MODE}:
        raise ValueError(f"unsupported compact heatmap mode: {mode}")
    compact_cache_root = Path(compact_cache_root).resolve()
    old_init = data_loader_module.dataset_set_name.__init__
    old_prepare = model_class.prepare_inputs
    old_data_loader = data_loader_module.DataLoader

    def worker_init(worker_id):
        del worker_id
        # The worker RNG has already been initialized by PyTorch. Restrict
        # only nested compute threads; do not resample or overwrite RNG state.
        torch.set_num_threads(1)
        try:
            import cv2
            cv2.setNumThreads(1)
        except ImportError:
            pass

    def compact_data_loader(*args, **kwargs):
        workers = int(kwargs.get("num_workers", 0))
        if workers > 0:
            kwargs.setdefault("prefetch_factor", 2)
            kwargs.setdefault("worker_init_fn", worker_init)
            # Keep the legacy epoch-boundary worker lifecycle. Enabling
            # persistent workers would change its augmentation RNG sequence.
            kwargs.setdefault("persistent_workers", False)
        return old_data_loader(*args, **kwargs)

    def compact_init(self, args, set_name, logical_role=None):
        old_init(self, args, set_name, logical_role=logical_role)
        self._paper_compact_mode = mode
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
            batch_data[f"{_INTERNAL_PREFIX}statistics"] = pickle.dumps({
                "route": "gpu_fast",
                "worker_prepare_seconds": 0.0,
                "applied_operations": [],
                "fallback_reasons": [],
                "map_size": "%dx%d" % tuple(
                    batch_data["_paper_compact_original_hw"]),
            }, protocol=pickle.HIGHEST_PROTOCOL)
        return batch_data, batch_id

    def compact_prepare(self, batch_data, batch_id):
        centers = batch_data.pop(f"{_INTERNAL_PREFIX}centers").squeeze(0)
        replay_value = batch_data.pop(f"{_INTERNAL_PREFIX}replay")
        reverse_value = batch_data.pop(f"{_INTERNAL_PREFIX}reverse")
        statistics_value = batch_data.pop(f"{_INTERNAL_PREFIX}statistics")
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
        statistics_bytes = (
            statistics_value[0] if isinstance(statistics_value, (list, tuple))
            else statistics_value)
        statistics = pickle.loads(statistics_bytes)
        route = statistics["route"]
        _PIPELINE_STATS["total_batches"] += 1
        _PIPELINE_STATS[f"{route}_batches"] += 1
        seconds = float(statistics["worker_prepare_seconds"])
        _PIPELINE_STATS["worker_prepare_seconds"] += seconds
        _PIPELINE_STATS[f"worker_prepare_seconds_{route}"] += seconds
        for name in statistics["applied_operations"]:
            _PIPELINE_STATS["applied_operations"][name] = (
                _PIPELINE_STATS["applied_operations"].get(name, 0) + 1)
        for name in statistics["fallback_reasons"]:
            _PIPELINE_STATS["fallback_reasons"][name] = (
                _PIPELINE_STATS["fallback_reasons"].get(name, 0) + 1)
        size = statistics["map_size"]
        _PIPELINE_STATS["map_sizes"][size] = (
            _PIPELINE_STATS["map_sizes"].get(size, 0) + 1)
        return old_prepare(self, batch_data, batch_id)

    data_loader_module.load_batch_cache = load_batch_cache_inprocess
    data_loader_module.DataLoader = compact_data_loader
    data_loader_module.dataset_set_name.__init__ = compact_init
    data_loader_module.dataset_set_name.__getitem__ = compact_getitem
    data_loader_module.dataset_set_name.augment_traj_and_images = augment_compact
    model_class.prepare_inputs = compact_prepare
    data_loader_module._paper_sdd_input_pipeline = mode

