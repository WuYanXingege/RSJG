"""Execution-only input optimizations for the read-only legacy SDD loader.

The model, cached values, augmentation distribution, and sample order are
unchanged.  Installation is explicit and fail-closed so other datasets keep
the pinned legacy path.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import io
import os
import pickle
import random
from pathlib import Path

import numpy as np
import torch


PREALLOC_MODE = "prealloc_replay_v1"
MODE = "inprocess_zstd_prealloc_v1"
_ZSTD_CONTENTSIZE_UNKNOWN = (1 << 64) - 1
_ZSTD_CONTENTSIZE_ERROR = (1 << 64) - 2
_ZSTD_LIBRARY = None


class _ZstdInBuffer(ctypes.Structure):
    _fields_ = [
        ("src", ctypes.c_void_p),
        ("size", ctypes.c_size_t),
        ("pos", ctypes.c_size_t),
    ]


class _ZstdOutBuffer(ctypes.Structure):
    _fields_ = [
        ("dst", ctypes.c_void_p),
        ("size", ctypes.c_size_t),
        ("pos", ctypes.c_size_t),
    ]


def _zstd_library():
    global _ZSTD_LIBRARY
    if _ZSTD_LIBRARY is not None:
        return _ZSTD_LIBRARY
    name = ctypes.util.find_library("zstd")
    if not name:
        raise RuntimeError("libzstd is unavailable for in-process cache reads")
    library = ctypes.CDLL(name)
    library.ZSTD_getFrameContentSize.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    library.ZSTD_getFrameContentSize.restype = ctypes.c_ulonglong
    library.ZSTD_decompress.argtypes = [
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t]
    library.ZSTD_decompress.restype = ctypes.c_size_t
    library.ZSTD_decompressBound.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    library.ZSTD_decompressBound.restype = ctypes.c_ulonglong
    library.ZSTD_isError.argtypes = [ctypes.c_size_t]
    library.ZSTD_isError.restype = ctypes.c_uint
    library.ZSTD_getErrorName.argtypes = [ctypes.c_size_t]
    library.ZSTD_getErrorName.restype = ctypes.c_char_p
    library.ZSTD_createDStream.argtypes = []
    library.ZSTD_createDStream.restype = ctypes.c_void_p
    library.ZSTD_freeDStream.argtypes = [ctypes.c_void_p]
    library.ZSTD_freeDStream.restype = ctypes.c_size_t
    library.ZSTD_initDStream.argtypes = [ctypes.c_void_p]
    library.ZSTD_initDStream.restype = ctypes.c_size_t
    library.ZSTD_decompressStream.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_ZstdOutBuffer),
        ctypes.POINTER(_ZstdInBuffer),
    ]
    library.ZSTD_decompressStream.restype = ctypes.c_size_t
    library.ZSTD_DStreamInSize.argtypes = []
    library.ZSTD_DStreamInSize.restype = ctypes.c_size_t
    _ZSTD_LIBRARY = library
    return library


def _check_zstd(code: int, operation: str) -> int:
    library = _zstd_library()
    if library.ZSTD_isError(code):
        message = library.ZSTD_getErrorName(code).decode(errors="replace")
        raise RuntimeError(f"{operation} failed: {message}")
    return int(code)


class ZstdStreamReader(io.RawIOBase):
    """Bounded-memory, in-process zstd reader suitable for pickle.load."""

    def __init__(self, path):
        super().__init__()
        self._library = _zstd_library()
        self._handle = open(path, "rb")
        self._stream = self._library.ZSTD_createDStream()
        if not self._stream:
            self._handle.close()
            raise MemoryError("ZSTD_createDStream returned null")
        _check_zstd(
            self._library.ZSTD_initDStream(self._stream),
            "ZSTD_initDStream")
        self._input_chunk_size = int(self._library.ZSTD_DStreamInSize())
        self._input_storage = None
        self._input = _ZstdInBuffer(None, 0, 0)
        self._source_eof = False
        self._frame_hint = 1

    def readable(self):
        return True

    def _refill(self):
        payload = self._handle.read(self._input_chunk_size)
        if not payload:
            self._source_eof = True
            self._input_storage = None
            self._input = _ZstdInBuffer(None, 0, 0)
            return
        self._input_storage = ctypes.create_string_buffer(payload, len(payload))
        self._input = _ZstdInBuffer(
            ctypes.cast(self._input_storage, ctypes.c_void_p), len(payload), 0)

    def readinto(self, buffer):
        if self.closed:
            raise ValueError("read from closed zstd stream")
        view = memoryview(buffer).cast("B")
        if not view:
            return 0
        output_storage = (ctypes.c_ubyte * len(view)).from_buffer(view)
        produced = 0
        while produced < len(view):
            if self._input.pos >= self._input.size and not self._source_eof:
                self._refill()
            if self._source_eof and self._input.pos >= self._input.size:
                if self._frame_hint != 0:
                    raise EOFError("truncated zstd frame")
                break
            output = _ZstdOutBuffer(
                ctypes.cast(
                    ctypes.byref(output_storage, produced), ctypes.c_void_p),
                len(view) - produced,
                0,
            )
            before_input = int(self._input.pos)
            self._frame_hint = _check_zstd(
                self._library.ZSTD_decompressStream(
                    self._stream, ctypes.byref(output),
                    ctypes.byref(self._input)),
                "ZSTD_decompressStream")
            produced += int(output.pos)
            if (int(output.pos) == 0 and
                    int(self._input.pos) == before_input):
                raise RuntimeError("zstd stream made no decompression progress")
        return produced

    def close(self):
        if not self.closed:
            if self._stream:
                _check_zstd(
                    self._library.ZSTD_freeDStream(self._stream),
                    "ZSTD_freeDStream")
                self._stream = None
            self._handle.close()
        super().close()


def decompress_zstd_frame(payload: bytes) -> bytes:
    """Losslessly decode one known-size zstd frame without a subprocess."""
    library = _zstd_library()
    source = ctypes.create_string_buffer(payload, len(payload))
    size = int(library.ZSTD_getFrameContentSize(source, len(payload)))
    if size == _ZSTD_CONTENTSIZE_ERROR:
        raise RuntimeError("SDD cache frame is invalid")
    capacity = size
    if size == _ZSTD_CONTENTSIZE_UNKNOWN:
        capacity = int(library.ZSTD_decompressBound(source, len(payload)))
        if capacity in {0, _ZSTD_CONTENTSIZE_ERROR}:
            raise RuntimeError(
                "SDD cache frame has no valid decompression bound")
    destination = ctypes.create_string_buffer(capacity)
    result = int(library.ZSTD_decompress(
        destination, capacity, source, len(payload)))
    if library.ZSTD_isError(result):
        message = library.ZSTD_getErrorName(result).decode(errors="replace")
        raise RuntimeError(f"zstd decompression failed: {message}")
    if size != _ZSTD_CONTENTSIZE_UNKNOWN and result != size:
        raise RuntimeError(
            f"zstd decompressed size mismatch: expected={size}, actual={result}")
    return destination.raw[:result]


def load_batch_cache_inprocess(path):
    """Deserialize the exact legacy pickle with an in-process zstd decoder."""
    path = os.fspath(path)
    if path.endswith(".pkl"):
        with open(path, "rb") as handle:
            return pickle.load(handle)
    if not path.endswith(".pkl.zst"):
        raise ValueError(f"Unsupported batch cache file: {path}")
    with ZstdStreamReader(path) as raw:
        with io.BufferedReader(raw, buffer_size=1024 * 1024) as stream:
            return pickle.load(stream)


def augmentation_transform():
    import albumentations as A
    import cv2

    return A.ReplayCompose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Transpose(p=0.5),
        A.RandomRotate90(p=1.0),
        A.OneOf([
            A.Perspective(
                scale=0.05, pad_mode=cv2.BORDER_CONSTANT, p=1.0),
            A.Affine(
                shear=(-10, 10), mode=cv2.BORDER_CONSTANT, p=1.0),
        ], p=0.2),
        A.OneOf([
            A.ShiftScaleRotate(
                shift_limit_x=0.01, shift_limit_y=0, scale_limit=0,
                rotate_limit=0, border_mode=cv2.BORDER_CONSTANT, p=1.0),
            A.ShiftScaleRotate(
                shift_limit_x=0, shift_limit_y=0.01, scale_limit=0,
                rotate_limit=0, border_mode=cv2.BORDER_CONSTANT, p=1.0),
            A.Affine(
                translate_percent=(0, 0.01),
                mode=cv2.BORDER_CONSTANT, p=1.0),
        ], p=0.2),
        A.Rotate(limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.4),
    ], keypoint_params=A.KeypointParams(
        format="xy", remove_invisible=False))


def replay_traj_maps_preallocated(replay, input_traj_maps, chunk_size=256):
    """Replay one sampled transform without list+concatenate full-size copies."""
    import albumentations as A

    output = None
    channels = input_traj_maps.shape[-1]
    for start in range(0, channels, chunk_size):
        stop = min(start + chunk_size, channels)
        chunk = np.ascontiguousarray(input_traj_maps[..., start:stop])
        transformed = A.ReplayCompose.replay(replay, image=chunk)["image"]
        if output is None:
            output = np.empty(
                (*transformed.shape[:2], channels), dtype=transformed.dtype)
        output[..., start:stop] = transformed
    if output is None:
        raise RuntimeError("trajectory-map tensor has no channels")
    return output


def optimized_augment_traj_and_images(self, batch_data):
    image = batch_data["tensor_image"]
    coordinates = batch_data["abs_pixel_coord"]
    trajectory_maps = batch_data["input_traj_maps"]

    image = image.permute(1, 2, 0).numpy()
    if image.dtype != np.float32:
        image = image.astype(np.float32)
    else:
        image = np.ascontiguousarray(image)
    batch_size, steps, old_height, old_width = trajectory_maps.shape
    trajectory_maps = trajectory_maps.view(
        batch_size * steps, old_height, old_width).permute(1, 2, 0).numpy()
    if trajectory_maps.dtype != np.float32:
        trajectory_maps = trajectory_maps.astype(np.float32)

    coordinates[:, :, 0] = np.clip(
        coordinates[:, :, 0], 0, old_width - 1e-3)
    coordinates[:, :, 1] = np.clip(
        coordinates[:, :, 1], 0, old_height - 1e-3)
    keypoints = list(map(tuple, coordinates.reshape(-1, 2)))

    transform = getattr(self, "_paper_replay_transform", None)
    if transform is None:
        transform = augmentation_transform()
        self._paper_replay_transform = transform
    transformed = transform(image=image, keypoints=keypoints)
    transformed_maps = replay_traj_maps_preallocated(
        transformed["replay"], trajectory_maps)

    image = torch.from_numpy(np.ascontiguousarray(
        transformed["image"])).permute(2, 0, 1)
    _, new_height, new_width = image.shape
    # Match the legacy constructor exactly: SDD coordinates enter as a
    # float64 NumPy array, while Albumentations keypoints become float32 here.
    coordinates = torch.tensor(transformed["keypoints"]).view(
        batch_data["abs_pixel_coord"].shape)
    trajectory_maps = torch.from_numpy(transformed_maps).permute(
        2, 0, 1).view(batch_size, steps, new_height, new_width)

    if random.random() > 0.5:
        coordinates = coordinates.flip(dims=(0,))
        trajectory_maps = trajectory_maps.flip(dims=(1,))
        if "frame_ids" in batch_data:
            batch_data["frame_ids"] = batch_data["frame_ids"].flip(dims=(0,))

    batch_data["tensor_image"] = image
    batch_data["abs_pixel_coord"] = coordinates
    batch_data["input_traj_maps"] = trajectory_maps
    return batch_data


def install(data_loader_module, batch_cache_module, mode: str = MODE) -> None:
    """Install the SDD-only optimized read and augmentation implementation."""
    if mode not in {PREALLOC_MODE, MODE}:
        raise ValueError(f"unsupported SDD input pipeline mode: {mode}")
    if getattr(data_loader_module, "_paper_sdd_input_pipeline", None) == mode:
        return
    if mode == MODE:
        batch_cache_module.load_batch_cache = load_batch_cache_inprocess
        data_loader_module.load_batch_cache = load_batch_cache_inprocess
    data_loader_module.dataset_set_name.augment_traj_and_images = (
        optimized_augment_traj_and_images)
    data_loader_module._paper_sdd_input_pipeline = mode
