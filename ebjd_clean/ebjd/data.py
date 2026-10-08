"""Scene-complete NPZ input and padding for EBJD."""

from __future__ import annotations

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

    def to(self, device: torch.device | str) -> "SceneBatch":
        return SceneBatch(
            self.observed.to(device), self.future.to(device),
            self.semantic_maps.to(device), self.valid.to(device), self.scene_ids)


class NPZSceneDataset(Dataset):
    """Read one complete synchronized scene per entry without truncating agents.

    Required arrays are ``observed`` and ``future``. They may be dense arrays or
    object arrays of respectively [N,8,2] and [N,12,2]. ``semantic_maps`` is
    [N,6,H,W] or [N,14,H,W]. Coordinates are expected in metres.
    """

    def __init__(self, path: str | Path) -> None:
        payload = np.load(Path(path), allow_pickle=True)
        self.observed = payload["observed"]
        self.future = payload["future"]
        self.maps = payload["semantic_maps"]
        self.scene_ids = payload.get("scene_ids", np.arange(len(self.observed)).astype(str))
        if not (len(self.observed) == len(self.future) == len(self.maps)):
            raise ValueError("NPZ scene arrays have different lengths")

    def __len__(self) -> int:
        return len(self.observed)

    def __getitem__(self, index: int) -> dict:
        observed = torch.as_tensor(self.observed[index], dtype=torch.float32)
        future = torch.as_tensor(self.future[index], dtype=torch.float32)
        maps = torch.as_tensor(self.maps[index], dtype=torch.float32)
        if observed.shape[-2:] != (8, 2) or future.shape[-2:] != (12, 2):
            raise ValueError("NPZ must use observed 8 and future 12 metre coordinates")
        if maps.shape[0] != observed.shape[0] or maps.shape[1] not in (6, 14):
            raise ValueError("each scene needs one 6/14-channel crop per agent")
        return {
            "observed": observed, "future": future, "semantic_maps": maps,
            "scene_id": str(self.scene_ids[index]),
        }


def collate_scenes(samples: list[dict]) -> SceneBatch:
    if not samples:
        raise ValueError("cannot collate an empty batch")
    maximum = max(item["observed"].shape[0] for item in samples)
    batch = len(samples)
    channels, height, width = samples[0]["semantic_maps"].shape[1:]
    observed = torch.zeros(batch, maximum, 8, 2)
    future = torch.zeros(batch, maximum, 12, 2)
    maps = torch.zeros(batch, maximum, channels, height, width)
    valid = torch.zeros(batch, maximum, dtype=torch.bool)
    for index, item in enumerate(samples):
        agents = item["observed"].shape[0]
        if item["semantic_maps"].shape[1:] != (channels, height, width):
            raise ValueError("map crop shapes must be equal within a batch")
        observed[index, :agents] = item["observed"]
        future[index, :agents] = item["future"]
        maps[index, :agents] = item["semantic_maps"]
        valid[index, :agents] = True
    return SceneBatch(observed, future, maps, valid, [item["scene_id"] for item in samples])


class SyntheticSceneDataset(Dataset):
    """Deterministic smoke-only data; never presented as a benchmark result."""

    def __init__(self, scenes: int = 8, agents: int = 3, pixels: int = 32, seed: int = 1) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.items = []
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
    for scene in range(batch_size):
        center = observed[scene, batch.valid[scene], -1].mean(0)
        cosine, sine = angles[scene].cos(), angles[scene].sin()
        rotation = torch.stack((
            torch.stack((cosine, -sine)), torch.stack((sine, cosine))))
        if reflected[scene]:
            rotation = rotation @ torch.diag(torch.tensor([-1.0, 1.0]))
        observed[scene] = (observed[scene] - center) @ rotation.T + center
        future[scene] = (future[scene] - center) @ rotation.T + center
        # affine_grid maps output coordinates to input; inverse equals transpose.
        inverse = rotation.T
        theta = torch.zeros(maps.shape[1], 2, 3, dtype=maps.dtype)
        theta[:, :, :2] = inverse
        grid = F.affine_grid(theta, maps[scene].shape, align_corners=False)
        maps[scene] = F.grid_sample(
            maps[scene], grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return SceneBatch(observed, future, maps, batch.valid.clone(), list(batch.scene_ids))
