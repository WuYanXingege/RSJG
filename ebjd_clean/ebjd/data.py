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
    native_goals: torch.Tensor | None = None
    augmentation_sha256: str | None = None

    def to(self, device: torch.device | str) -> "SceneBatch":
        return SceneBatch(
            self.observed.to(device), self.future.to(device),
            self.semantic_maps.to(device), self.valid.to(device), self.scene_ids,
            self.agent_ids, self.frame_ids, self.timestamps,
            self.source_sequences, self.metadata,
            None if self.native_goals is None else self.native_goals.to(device),
            self.augmentation_sha256)

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
            None if self.native_goals is None else self.native_goals[start:stop],
            self.augmentation_sha256)

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
            None if self.native_goals is None else pack(self.native_goals),
            self.augmentation_sha256)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class NativeGoalBank:
    """Strict identity-bound frozen native-GDTS goal candidate bank."""

    SCHEMA = "ebjd-native-goal-bank-v1"

    def __init__(
        self, manifest_path: str | Path, source_manifest: str | Path,
        verify_hashes: bool = True,
    ) -> None:
        self.path = Path(manifest_path).resolve()
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("schema") != self.SCHEMA or payload.get("status") != "COMPLETED":
            raise ValueError("native goal bank is not a completed supported bank")
        source = Path(source_manifest).resolve()
        if payload.get("source_manifest_sha256") != _sha256(source):
            raise ValueError("native goal bank source manifest SHA256 mismatch")
        if payload.get("unit") != "m" or int(payload.get("candidates", -1)) != 20:
            raise ValueError("native goal bank must contain K=20 world-metre candidates")
        self.payload = payload
        self.items: dict[str, dict] = {}
        for descriptor in payload.get("files", []):
            path = (self.path.parent / descriptor["path"]).resolve()
            if verify_hashes and _sha256(path) != descriptor["sha256"]:
                raise ValueError(f"native goal cache shard SHA256 mismatch: {path}")
            with np.load(path, allow_pickle=True) as archive:
                count = len(archive["window_keys"])
                for index in range(count):
                    key = str(archive["window_keys"][index])
                    if key in self.items:
                        raise ValueError(f"duplicate native goal window: {key}")
                    self.items[key] = {
                        "scene_id": str(archive["scene_ids"][index]),
                        "source_sequence": str(archive["source_sequences"][index]),
                        "frame_ids": [int(x) for x in archive["frame_ids"][index]],
                        "agent_ids": [str(x) for x in archive["agent_ids"][index]],
                        "observed_sha256": str(archive["observed_sha256"][index]),
                        "candidates": np.asarray(
                            archive["candidates"][index], dtype=np.float32),
                    }
        if len(self.items) != int(payload.get("scene_count", -1)):
            raise ValueError("native goal bank scene count mismatch")

    def get(
        self, window_key: str, scene_id: str, source_sequence: str,
        frame_ids: list[int], agent_ids: list[str], observed: torch.Tensor,
    ) -> torch.Tensor:
        if window_key not in self.items:
            raise KeyError(f"native goal bank misses {window_key}")
        item = self.items[window_key]
        checks = {
            "scene_id": item["scene_id"] == scene_id,
            "source_sequence": item["source_sequence"] == source_sequence,
            "frame_ids": item["frame_ids"] == frame_ids,
            "agent_ids": item["agent_ids"] == agent_ids,
            "observed_sha256": item["observed_sha256"] == hashlib.sha256(
                observed.contiguous().numpy().tobytes()).hexdigest(),
        }
        if not all(checks.values()):
            raise ValueError(f"native goal identity mismatch for {window_key}: {checks}")
        candidates = torch.from_numpy(item["candidates"].copy())
        if candidates.shape != (len(agent_ids), 20, 2) or not torch.isfinite(candidates).all():
            raise ValueError(f"native goals have invalid shape/content for {window_key}")
        return candidates


class SceneManifestDataset(Dataset):
    """Bounded shard reader with mandatory unit/crop/identity metadata."""

    def __init__(
        self, manifest_path: str | Path, verify_hashes: bool = True,
        native_goal_bank: str | Path | None = None,
    ) -> None:
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
        self.native_goal_bank = (
            NativeGoalBank(native_goal_bank, self.manifest_path, verify_hashes)
            if native_goal_bank is not None else None)

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
        scene_id = str(payload["scene_ids"][local])
        source_sequence = str(payload["source_sequences"][local])
        window_key = str(payload["window_keys"][local])
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
        item = {
            "observed": observed, "future": future, "semantic_maps": maps,
            "scene_id": scene_id, "agent_ids": agent_ids,
            "frame_ids": frame_ids, "timestamps": timestamps,
            "source_sequence": source_sequence, "window_key": window_key,
            "metadata": self.metadata,
        }
        if self.native_goal_bank is not None:
            item["native_goals"] = self.native_goal_bank.get(
                window_key, scene_id, source_sequence, frame_ids, agent_ids, observed)
        return item


# Backward-compatible name, but the input is now a manifest rather than an ad-hoc NPZ.
NPZSceneDataset = SceneManifestDataset


def collate_scenes(samples: list[dict]) -> SceneBatch:
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
    maps = torch.zeros(batch, maximum, channels, height, width)
    valid = torch.zeros(batch, maximum, dtype=torch.bool)
    has_native = all("native_goals" in item for item in samples)
    if has_native != any("native_goals" in item for item in samples):
        raise ValueError("native goals are present for only part of a batch")
    native_goals = torch.zeros(batch, maximum, 20, 2) if has_native else None
    for index, item in enumerate(samples):
        agents = item["observed"].shape[0]
        if item["semantic_maps"].shape[1:] != (channels, height, width):
            raise ValueError("map crop shapes differ within a batch")
        observed[index, :agents] = item["observed"]
        future[index, :agents] = item["future"]
        maps[index, :agents] = item["semantic_maps"]
        valid[index, :agents] = True
        if native_goals is not None:
            native_goals[index, :agents] = item["native_goals"]
    return SceneBatch(
        observed, future, maps, valid,
        [item["scene_id"] for item in samples],
        [item["agent_ids"] for item in samples],
        [item["frame_ids"] for item in samples],
        [item["timestamps"] for item in samples],
        [item["source_sequence"] for item in samples], metadata,
        native_goals, None)


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
            })

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        return self.items[index]


def augment_batch(batch: SceneBatch, generator: torch.Generator | None = None) -> SceneBatch:
    """Rotate/reflect every scene consistently, including all agent map crops."""
    batch_size = batch.observed.shape[0]
    angles = 2 * torch.pi * torch.rand(batch_size, generator=generator)
    reflected = torch.rand(batch_size, generator=generator) < 0.5
    observed, future, maps = batch.observed.clone(), batch.future.clone(), batch.semantic_maps.clone()
    native_goals = None if batch.native_goals is None else batch.native_goals.clone()
    for scene in range(batch_size):
        center = observed[scene, batch.valid[scene], -1].mean(0)
        cosine, sine = angles[scene].cos(), angles[scene].sin()
        rotation = torch.stack((
            torch.stack((cosine, -sine)), torch.stack((sine, cosine))))
        if reflected[scene]:
            rotation = rotation @ torch.diag(torch.tensor([-1.0, 1.0]))
        observed[scene] = (observed[scene] - center) @ rotation.T + center
        future[scene] = (future[scene] - center) @ rotation.T + center
        if native_goals is not None:
            native_goals[scene] = (
                native_goals[scene] - center.view(1, 1, 2)) @ rotation.T + center
        inverse = rotation.T
        theta = torch.zeros(maps.shape[1], 2, 3, dtype=maps.dtype)
        theta[:, :, :2] = inverse
        grid = F.affine_grid(theta, maps[scene].shape, align_corners=False)
        maps[scene] = F.grid_sample(
            maps[scene], grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    augmentation = hashlib.sha256()
    augmentation.update(angles.contiguous().numpy().tobytes())
    augmentation.update(reflected.contiguous().numpy().tobytes())
    augmentation.update("\0".join(batch.scene_ids).encode("utf-8"))
    return SceneBatch(
        observed, future, maps, batch.valid.clone(), list(batch.scene_ids),
        batch.agent_ids, batch.frame_ids, batch.timestamps,
        batch.source_sequences, batch.metadata, native_goals,
        augmentation.hexdigest())
