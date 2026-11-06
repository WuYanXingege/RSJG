"""Identity-checked complete-scene shard input and padding for EBJD."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset


@dataclass
class SceneBatch:
    observed: torch.Tensor
    future: torch.Tensor
    semantic_maps: torch.Tensor
    valid: torch.Tensor
    scene_ids: list[str]
    agent_ids: list[list[str]]
    frame_ids: list[list[int]]
    timestamps: list[list[float]]
    source_sequences: list[str]
    metadata: dict
    # Multiplicative conversion from the model's internal coordinate unit to
    # the paper metric unit.  ETH/UCY manifests default to one.  JMM SDD
    # stores metres internally and supplies px/m independently per scene.
    paper_metric_scale: torch.Tensor | None = None

    def to(
        self, device: torch.device | str, *, non_blocking: bool = False,
    ) -> "SceneBatch":
        return SceneBatch(
            self.observed.to(device, non_blocking=non_blocking),
            self.future.to(device, non_blocking=non_blocking),
            self.semantic_maps.to(device, non_blocking=non_blocking),
            self.valid.to(device, non_blocking=non_blocking), self.scene_ids,
            self.agent_ids, self.frame_ids, self.timestamps,
            self.source_sequences, self.metadata,
            None if self.paper_metric_scale is None else
            self.paper_metric_scale.to(device, non_blocking=non_blocking))

    def pin_memory(self) -> "SceneBatch":
        """Pin only tensor storage; identities and metadata remain shared."""
        return SceneBatch(
            self.observed.pin_memory(), self.future.pin_memory(),
            self.semantic_maps.pin_memory(), self.valid.pin_memory(),
            self.scene_ids, self.agent_ids, self.frame_ids, self.timestamps,
            self.source_sequences, self.metadata,
            None if self.paper_metric_scale is None else
            self.paper_metric_scale.pin_memory())

    def slice_scenes(self, start: int, stop: int) -> "SceneBatch":
        """Slice only the scene dimension while retaining the original N padding."""
        batch = int(self.observed.shape[0])
        if not (0 <= start < stop <= batch):
            raise IndexError(
                f"scene slice [{start}:{stop}] is outside batch size {batch}")
        return SceneBatch(
            self.observed[start:stop], self.future[start:stop],
            self.semantic_maps[start:stop], self.valid[start:stop],
            list(self.scene_ids[start:stop]), list(self.agent_ids[start:stop]),
            list(self.frame_ids[start:stop]), list(self.timestamps[start:stop]),
            list(self.source_sequences[start:stop]), self.metadata,
            None if self.paper_metric_scale is None else
            self.paper_metric_scale[start:stop])

    def compact_scenes(self, start: int, stop: int) -> "SceneBatch":
        """Pack valid agents of contiguous scenes without changing their order.

        The caller must draw randomness in the original logical ``[B,Npad]``
        layout before invoking this method.  This operation only removes invalid
        padding and then re-pads the selected scenes to their local maximum.
        It intentionally supports non-prefix validity masks so that compaction
        does not rely on an undocumented collate invariant.
        """
        batch = int(self.observed.shape[0])
        if not (0 <= start < stop <= batch):
            raise IndexError(
                f"scene slice [{start}:{stop}] is outside batch size {batch}")
        positions = [
            self.valid[index].nonzero(as_tuple=False).flatten()
            for index in range(start, stop)
        ]
        if any(int(index.numel()) == 0 for index in positions):
            raise ValueError("each compacted scene must contain a valid agent")
        maximum = max(int(index.numel()) for index in positions)
        group = stop - start

        def pack(source: torch.Tensor) -> torch.Tensor:
            packed = source.new_zeros((group, maximum, *source.shape[2:]))
            for local, (scene, index) in enumerate(zip(
                range(start, stop), positions, strict=True,
            )):
                packed[local, :index.numel()] = source[scene].index_select(
                    0, index.to(source.device))
            return packed

        valid = torch.zeros(
            group, maximum, dtype=torch.bool, device=self.valid.device)
        for local, index in enumerate(positions):
            valid[local, :index.numel()] = True
        agent_ids = []
        for scene, index in zip(range(start, stop), positions, strict=True):
            identities = list(self.agent_ids[scene])
            if len(identities) != int(index.numel()):
                raise ValueError(
                    "agent identity count does not match the valid-mask count")
            agent_ids.append(identities)
        return SceneBatch(
            pack(self.observed), pack(self.future), pack(self.semantic_maps), valid,
            list(self.scene_ids[start:stop]), agent_ids,
            list(self.frame_ids[start:stop]), list(self.timestamps[start:stop]),
            list(self.source_sequences[start:stop]), self.metadata,
            None if self.paper_metric_scale is None else
            self.paper_metric_scale[start:stop])

    def paper_coordinates(
        self, predictions: torch.Tensor, targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert model-space trajectories to the declared paper unit."""
        if self.paper_metric_scale is None:
            return predictions, targets
        scale = self.paper_metric_scale.to(
            device=predictions.device, dtype=predictions.dtype)
        if scale.ndim != 1 or scale.shape[0] != predictions.shape[0]:
            raise ValueError("paper metric scale must have shape [B]")
        if not torch.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError("paper metric scale must be finite and positive")
        return (
            predictions * scale[:, None, None, None, None],
            targets * scale[:, None, None, None],
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SceneManifestDataset(Dataset):
    """Bounded shard reader with mandatory unit/crop/identity metadata."""

    def __init__(self, manifest_path: str | Path, verify_hashes: bool = True) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("schema") != "ebjd-scene-manifest-v1":
            raise ValueError("unsupported or missing scene manifest schema")
        metadata = self.manifest.get("metadata", {})
        required = {
            "unit": "m", "dt_seconds": 0.4, "obs_length": 8,
            "pred_length": 12, "crop_width_m": 32.0, "crop_pixels": 256,
            "semantic_channels": 6, "align_corners": True,
            "out_of_bounds_fill": 0,
        }
        for key, expected in required.items():
            if metadata.get(key) != expected:
                raise ValueError(f"manifest metadata {key} must be {expected!r}")
        self.metadata = metadata
        self.shards = []
        self.index: list[tuple[int, int]] = []
        identities: set[str] = set()
        for shard_index, descriptor in enumerate(self.manifest.get("shards", [])):
            path = (self.manifest_path.parent / descriptor["path"]).resolve()
            if verify_hashes and _sha256(path) != descriptor["sha256"]:
                raise ValueError(f"shard SHA256 mismatch: {path}")
            with np.load(path, allow_pickle=True) as payload:
                shard_scene_ids = [str(value) for value in payload["scene_ids"]]
                if len(shard_scene_ids) != int(descriptor["scenes"]):
                    raise ValueError(f"shard scene count mismatch: {path}")
                if identities.intersection(shard_scene_ids):
                    raise ValueError(f"duplicate scene identity in shards: {path}")
                identities.update(shard_scene_ids)
            self.shards.append(path)
            self.index.extend((shard_index, local) for local in range(int(descriptor["scenes"])))
        if not self.index:
            raise ValueError("manifest contains no scenes")
        statistics = self.manifest.get("statistics", {})
        if int(statistics.get("scenes", -1)) != len(self.index):
            raise ValueError("manifest scene count differs from shard descriptors")
        if len(set(self.manifest.get("window_keys", []))) != len(self.index):
            raise ValueError("manifest window keys are missing or duplicated")
        self._cached_index = -1
        self._cached_payload = None

    def __len__(self) -> int:
        return len(self.index)

    def _payload(self, shard_index: int):
        if shard_index != self._cached_index:
            # Materialize each compressed member exactly once per shard.  Repeated
            # ``NpzFile[key]`` calls would decompress the complete object array for
            # every scene in that shard, which is bounded but unnecessarily
            # quadratic in the number of scenes per shard.
            with np.load(self.shards[shard_index], allow_pickle=True) as archive:
                self._cached_payload = {
                    name: archive[name] for name in archive.files}
            self._cached_index = shard_index
        return self._cached_payload

    def __getitem__(self, index: int) -> dict:
        shard_index, local = self.index[index]
        payload = self._payload(shard_index)
        observed = torch.as_tensor(np.asarray(payload["observed"][local]), dtype=torch.float32)
        future = torch.as_tensor(np.asarray(payload["future"][local]), dtype=torch.float32)
        maps = torch.as_tensor(np.asarray(payload["semantic_maps"][local]), dtype=torch.uint8)
        agent_ids = [str(item) for item in payload["agent_ids"][local]]
        frame_ids = [int(item) for item in payload["frame_ids"][local]]
        timestamps = [float(item) for item in payload["timestamps"][local]]
        if observed.ndim != 3 or observed.shape[-2:] != (8, 2):
            raise ValueError("observed must be finite [N,8,2]")
        if future.shape != (observed.shape[0], 12, 2):
            raise ValueError("future agent count/dimensions differ from observed")
        if maps.shape != (observed.shape[0], 6, 256, 256):
            raise ValueError("semantic maps must be [N,6,256,256]")
        if len(agent_ids) != observed.shape[0] or len(frame_ids) != 20 or len(timestamps) != 20:
            raise ValueError("scene identity lengths are inconsistent")
        if len(set(agent_ids)) != len(agent_ids) or len(set(frame_ids)) != 20:
            raise ValueError("agent IDs and frame IDs must be unique within a scene")
        if frame_ids != sorted(frame_ids):
            raise ValueError("frame IDs must be strictly increasing")
        if not torch.isfinite(observed).all() or not torch.isfinite(future).all() or not torch.isfinite(maps).all():
            raise ValueError("scene contains NaN or Inf")
        if not torch.all((maps == 0) | (maps == 1)) or (maps.sum(1) > 1).any():
            raise ValueError("categorical semantic crop must be one-hot or all-zero out of bounds")
        if not np.allclose(np.diff(timestamps), 0.4, atol=1e-7):
            raise ValueError("timestamps are not a synchronized 0.4 s horizon")
        return {
            "observed": observed, "future": future, "semantic_maps": maps,
            "scene_id": str(payload["scene_ids"][local]), "agent_ids": agent_ids,
            "frame_ids": frame_ids, "timestamps": timestamps,
            "source_sequence": str(payload["source_sequences"][local]),
            "paper_metric_scale": (
                1.0 / float(payload["scale_m_per_px"][local])
                if "scale_m_per_px" in payload else 1.0),
            "metadata": self.metadata,
        }


# Backward-compatible name, but the input is now a manifest rather than an ad-hoc NPZ.
NPZSceneDataset = SceneManifestDataset


def _collate_scenes(
    samples: list[dict], *, preserve_map_storage_dtype: bool,
) -> SceneBatch:
    if not samples:
        raise ValueError("cannot collate an empty batch")
    metadata = samples[0]["metadata"]
    if any(item["metadata"] != metadata for item in samples):
        raise ValueError("metadata differs within a scene batch")
    maximum = max(item["observed"].shape[0] for item in samples)
    batch, channels, height, width = (
        len(samples), *samples[0]["semantic_maps"].shape[1:])
    observed = torch.zeros(batch, maximum, 8, 2)
    future = torch.zeros(batch, maximum, 12, 2)
    map_dtype = (
        samples[0]["semantic_maps"].dtype
        if preserve_map_storage_dtype else torch.float32)
    maps = torch.zeros(
        batch, maximum, channels, height, width, dtype=map_dtype)
    valid = torch.zeros(batch, maximum, dtype=torch.bool)
    paper_metric_scale = torch.as_tensor([
        float(item.get("paper_metric_scale", 1.0)) for item in samples
    ], dtype=torch.float32)
    if not torch.isfinite(paper_metric_scale).all() or (paper_metric_scale <= 0).any():
        raise ValueError("paper metric scales must be finite and positive")
    for index, item in enumerate(samples):
        agents = item["observed"].shape[0]
        if item["semantic_maps"].shape[1:] != (channels, height, width):
            raise ValueError("map crop shapes differ within a batch")
        observed[index, :agents] = item["observed"]
        future[index, :agents] = item["future"]
        maps[index, :agents] = item["semantic_maps"]
        valid[index, :agents] = True
    return SceneBatch(
        observed, future, maps, valid,
        [item["scene_id"] for item in samples],
        [item["agent_ids"] for item in samples],
        [item["frame_ids"] for item in samples],
        [item["timestamps"] for item in samples],
        [item["source_sequence"] for item in samples], metadata,
        paper_metric_scale)


def collate_scenes(samples: list[dict]) -> SceneBatch:
    """Legacy collation: materialize semantic maps as CPU float32."""
    return _collate_scenes(samples, preserve_map_storage_dtype=False)


def collate_scenes_compact_maps(samples: list[dict]) -> SceneBatch:
    """Keep categorical semantic maps uint8 until their first GPU operation.

    Shards already store six-channel categorical crops as uint8.  Preserving
    that representation avoids a four-fold expansion in DataLoader workers
    and during host-to-device transfer.  The encoder still receives float maps
    after the existing training augmentation, so model semantics are unchanged.
    """
    return _collate_scenes(samples, preserve_map_storage_dtype=True)


class SyntheticSceneDataset(Dataset):
    """Deterministic smoke-only data; never presented as a benchmark result."""

    def __init__(self, scenes: int = 8, agents: int = 3, pixels: int = 32, seed: int = 1) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.items = []
        metadata = {
            "unit": "m", "dt_seconds": 0.4, "obs_length": 8,
            "pred_length": 12, "crop_width_m": 32.0, "crop_pixels": pixels,
            "semantic_channels": 6, "align_corners": True,
            "out_of_bounds_fill": 0, "synthetic": True,
        }
        for scene in range(scenes):
            count = max(1, agents - scene % 2)
            start = torch.randn(count, 1, 2, generator=generator)
            velocity = 0.3 * torch.randn(count, 1, 2, generator=generator)
            times = torch.arange(-7, 13).view(1, 20, 1) * 0.4
            full = start + times * velocity + 0.02 * torch.randn(count, 20, 2, generator=generator)
            self.items.append({
                "observed": full[:, :8], "future": full[:, 8:],
                "semantic_maps": torch.zeros(count, 6, pixels, pixels),
                "scene_id": f"synthetic-{scene}",
                "agent_ids": [f"agent-{index}" for index in range(count)],
                "frame_ids": list(range(20)),
                "timestamps": [index * 0.4 for index in range(20)],
                "source_sequence": "synthetic", "metadata": metadata,
                "paper_metric_scale": 1.0,
            })

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        return self.items[index]


@dataclass(frozen=True)
class SceneRigidPlan:
    """CPU-sampled scene transform shared by coordinates, maps and proposals."""

    centers: torch.Tensor  # [B,2], deliberately computed on CPU once
    rotations: torch.Tensor  # [B,2,2], forward row-vector transform

    def to(self, device: torch.device | str) -> "SceneRigidPlan":
        return SceneRigidPlan(
            self.centers.to(device), self.rotations.to(device))


def sample_scene_rigid_plan(
    batch: SceneBatch, generator: torch.Generator | None = None,
) -> SceneRigidPlan:
    """Draw the legacy rotation/reflection without changing its RNG stream."""
    if batch.observed.device.type != "cpu":
        raise ValueError("formal scene augmentation plans must be sampled on CPU")
    batch_size = int(batch.observed.shape[0])
    angles = 2 * torch.pi * torch.rand(batch_size, generator=generator)
    reflected = torch.rand(batch_size, generator=generator) < 0.5
    centers = torch.stack([
        batch.observed[scene, batch.valid[scene], -1].mean(0)
        for scene in range(batch_size)
    ])
    rotations = []
    for scene in range(batch_size):
        cosine, sine = angles[scene].cos(), angles[scene].sin()
        rotation = torch.stack((
            torch.stack((cosine, -sine)),
            torch.stack((sine, cosine)),
        ))
        if reflected[scene]:
            rotation = rotation @ torch.diag(torch.tensor([-1.0, 1.0]))
        rotations.append(rotation)
    return SceneRigidPlan(centers=centers, rotations=torch.stack(rotations))


def apply_scene_rigid_plan(
    batch: SceneBatch, points: torch.Tensor, plan: SceneRigidPlan,
) -> tuple[SceneBatch, torch.Tensor]:
    """Apply one explicit plan on CPU or GPU without sampling randomness."""
    batch_size = int(batch.observed.shape[0])
    if points.shape[0] != batch_size or points.shape[-2:] != (
        batch.observed.shape[1], 2,
    ):
        raise ValueError("points must have shape [B,...,N,2] with batch padding")
    if plan.centers.shape != (batch_size, 2) or plan.rotations.shape != (
        batch_size, 2, 2,
    ):
        raise ValueError("scene rigid plan has incompatible dimensions")
    if points.device != batch.observed.device:
        raise ValueError("points and scene batch must be on the same device")
    device_plan = plan.to(batch.observed.device)
    observed, future = batch.observed.clone(), batch.future.clone()
    # Categorical crops remain compact uint8 through collation and transfer,
    # then become float32 immediately before the same bilinear grid_sample.
    maps = batch.semantic_maps.float().clone()
    transformed_points = points.clone()
    for scene in range(batch_size):
        center = device_plan.centers[scene].to(observed)
        rotation = device_plan.rotations[scene].to(observed)
        observed[scene] = (observed[scene] - center) @ rotation.T + center
        future[scene] = (future[scene] - center) @ rotation.T + center
        transformed_points[scene] = (
            transformed_points[scene] - center) @ rotation.T + center
        inverse = rotation.T
        theta = torch.zeros(
            maps.shape[1], 2, 3, dtype=maps.dtype, device=maps.device)
        theta[:, :, :2] = inverse.to(theta)
        grid = F.affine_grid(theta, maps[scene].shape, align_corners=False)
        maps[scene] = F.grid_sample(
            maps[scene], grid, mode="bilinear", padding_mode="zeros",
            align_corners=False)
    augmented = SceneBatch(
        observed, future, maps, batch.valid.clone(), list(batch.scene_ids),
        batch.agent_ids, batch.frame_ids, batch.timestamps,
        batch.source_sequences, batch.metadata,
        None if batch.paper_metric_scale is None else
        batch.paper_metric_scale.clone())
    return augmented, transformed_points


def augment_batch_with_points(
    batch: SceneBatch,
    points: torch.Tensor,
    generator: torch.Generator | None = None,
) -> tuple[SceneBatch, torch.Tensor]:
    """Augment a scene batch and aligned ``[B,...,N,2]`` world points.

    This is the proposal-safe form of the formal augmentation: the exact same
    sampled rigid transform is applied to observations, labels, map crops and
    external point conditions without drawing any additional randomness.
    """
    plan = sample_scene_rigid_plan(batch, generator)
    return apply_scene_rigid_plan(batch, points, plan)


def augment_batch_with_points_on_device(
    batch: SceneBatch,
    points: torch.Tensor,
    device: torch.device | str,
    generator: torch.Generator | None = None,
    *,
    non_blocking: bool = False,
) -> tuple[SceneBatch, torch.Tensor]:
    """Transfer compact inputs, then run the existing rigid augmentation.

    Only raw-input preparation is under ``no_grad``.  MSCE, the coupling head,
    GCJD and CTF are invoked later by the caller and retain their full graph.
    """
    plan = sample_scene_rigid_plan(batch, generator)
    with torch.no_grad():
        device_batch = batch.to(device, non_blocking=non_blocking)
        device_points = points.to(device, non_blocking=non_blocking)
        return apply_scene_rigid_plan(device_batch, device_points, plan)


def augment_batch(batch: SceneBatch, generator: torch.Generator | None = None) -> SceneBatch:
    """Rotate/reflect every scene consistently, including all agent map crops."""
    placeholder = torch.zeros(
        batch.observed.shape[0], batch.observed.shape[1], 2,
        dtype=batch.observed.dtype, device=batch.observed.device)
    augmented, _ = augment_batch_with_points(batch, placeholder, generator)
    return augmented


def augment_batch_with_gt_endpoint_condition(
    batch: SceneBatch, generator: torch.Generator | None = None,
) -> tuple[SceneBatch, torch.Tensor]:
    """Apply formal augmentation and return its valid, padded GT endpoint.

    This helper is training-only: callers must not use it from inference
    because it intentionally reads the future label. Padding is zeroed after
    augmentation so it cannot become an accidental endpoint condition.
    """
    endpoints = batch.future[..., -1, :].detach()[:, None]
    endpoints = torch.where(
        batch.valid[:, None, :, None], endpoints, torch.zeros_like(endpoints))
    augmented, _ = augment_batch_with_points(batch, endpoints, generator)
    condition = augmented.future[..., -1, :].detach()
    condition = torch.where(
        augmented.valid[..., None], condition, torch.zeros_like(condition))
    return augmented, condition
