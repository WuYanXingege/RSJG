"""Standalone frozen native endpoint provider.

The provider intentionally contains no GDTS trajectory encoder or denoiser.  It
strict-loads an extracted matched-GDTS Goal U-Net, applies the native TTST
sampler, and converts its ordered candidates to world metres through an
explicit coordinate callback.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as TF


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.ReLU(True))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.block(value)


class Encoder(nn.Module):
    def __init__(self, channels: tuple[int, ...]) -> None:
        super().__init__()
        self.enc_blocks = nn.ModuleList([
            DoubleConv(channels[index], channels[index + 1])
            for index in range(len(channels) - 1)])
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, value: torch.Tensor) -> list[torch.Tensor]:
        features = []
        for block in self.enc_blocks:
            value = block(value)
            features.append(value)
            value = self.pool(value)
        return features


class UpConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(mode="bilinear", scale_factor=2, align_corners=False),
            nn.Conv2d(in_channels, out_channels, 3, padding=1))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.up(value)


class Decoder(nn.Module):
    def __init__(self, channels: tuple[int, ...]) -> None:
        super().__init__()
        self.upconvs = nn.ModuleList([
            UpConv(channels[index], channels[index + 1])
            for index in range(len(channels) - 1)])
        self.dec_blocks = nn.ModuleList([
            DoubleConv(2 * channels[index + 1], channels[index + 1])
            for index in range(len(channels) - 1)])

    def forward(self, value: torch.Tensor,
                features: list[torch.Tensor]) -> torch.Tensor:
        for up, block, skip in zip(
                self.upconvs, self.dec_blocks, features, strict=True):
            value = up(value)
            if skip.shape[-2:] != value.shape[-2:]:
                skip = TF.CenterCrop(list(value.shape[-2:]))(skip)
            value = block(torch.cat((value, skip), dim=1))
        return value


class GoalUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        enc = (14, 32, 32, 64, 64, 64)
        dec = (64, 64, 64, 32, 32)
        self.encoder = Encoder(enc)
        self.decoder = Decoder(dec)
        self.head = nn.Module()
        self.head.out_layer = nn.Conv2d(dec[-1], 12, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        height, width = value.shape[-2:]
        features = self.encoder(value)
        decoded = self.decoder(features[-1], features[-2::-1])
        return F.interpolate(self.head.out_layer(decoded), (height, width))


def _sample(probability: torch.Tensor, count: int = 1000) -> torch.Tensor:
    flat = probability.reshape(probability.shape[0] * probability.shape[1], -1)
    threshold = flat.max(dim=1, keepdim=True).values
    filtered = flat * (flat >= threshold * 0.05).to(flat.dtype)
    total = filtered.sum(dim=1, keepdim=True)
    normalized = filtered / total.clamp_min(torch.finfo(flat.dtype).tiny)
    fallback = torch.zeros_like(flat)
    fallback.scatter_(1, flat.argmax(dim=1, keepdim=True), 1.0)
    normalized = torch.where(total > 0, normalized, fallback)
    indices = torch.multinomial(normalized, count, replacement=True)
    indices = indices.reshape(probability.shape[0], probability.shape[1], -1)
    result = indices.unsqueeze(-1).repeat(1, 1, 1, 2).float()
    result[..., 0] = result[..., 0].remainder(probability.shape[3])
    result[..., 1] = torch.floor(result[..., 1] / probability.shape[3])
    return result


def _kmeans(points: torch.Tensor, clusters: int) -> torch.Tensor:
    centers = points[np.random.choice(len(points), clusters, replace=False)].clone().float()
    for _ in range(1000):
        assignment = ((points[:, None].float() - centers[None]) ** 2).sum(-1).argmin(1)
        previous = centers.clone()
        for index in range(clusters):
            selected = points[assignment == index]
            if not len(selected):
                selected = points[torch.randint(len(points), (1,))]
            centers[index] = selected.float().mean(0)
        if torch.sqrt(((centers - previous) ** 2).sum(1)).sum().square() < 0.001:
            break
    return centers


def native_ttst(probability: torch.Tensor, goals: int = 20) -> torch.Tensor:
    samples = _sample(probability).permute(2, 0, 1, 3)
    centers = torch.stack([
        _kmeans(samples[:, person, 0], goals)
        for person in range(samples.shape[1])])
    return centers.permute(1, 0, 2)  # [K,N,2], native ordered centers


class NativeGoalProvider(nn.Module):
    """Matched-weight Goal U-Net plus native TTST; no GDTS trajectory model."""

    def __init__(self, state: dict[str, torch.Tensor], manifest: dict) -> None:
        super().__init__()
        self.goal_module = GoalUNet()
        self.goal_module.load_state_dict(state, strict=True)
        self.artifact_manifest = manifest
        self.eval().requires_grad_(False)

    @classmethod
    def from_artifact(cls, manifest_path: str | Path) -> "NativeGoalProvider":
        path = Path(manifest_path).resolve()
        manifest = json.loads(path.read_text(encoding="utf-8"))
        weights = (path.parent / manifest["weights"]).resolve()
        if file_sha256(weights) != manifest["weights_sha256"]:
            raise ValueError("goal-provider artifact SHA256 mismatch")
        payload = torch.load(weights, map_location="cpu", weights_only=True)
        if set(payload) != {"state_dict"}:
            raise ValueError("goal-provider artifact members differ from schema")
        provider = cls(payload["state_dict"], manifest)
        if sum(parameter.numel() for parameter in provider.parameters()) != 613772:
            raise ValueError("goal-provider parameter count mismatch")
        return provider

    @torch.inference_mode()
    def predict_prepared(
        self,
        semantic_image: torch.Tensor,
        observed_maps: torch.Tensor,
        pixel_to_world: Callable[[torch.Tensor], torch.Tensor] | None = None,
        *,
        down_factor: float = 8.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        agents = observed_maps.shape[0]
        image = semantic_image.unsqueeze(0).repeat(agents, 1, 1, 1)
        logits = self.goal_module(torch.cat((image, observed_maps[:, :8]), dim=1))
        candidates = native_ttst(torch.sigmoid(logits[:, -1:]), 20)
        if pixel_to_world is not None:
            candidates = pixel_to_world(candidates * down_factor).float()
        return logits, candidates

    def predict(self, observed: torch.Tensor, scene_inputs: dict,
                *, seed: int | None = None) -> torch.Tensor:
        """Unified public API over explicitly prepared native scene inputs."""
        del observed  # identity is carried for API consistency; maps are precomputed.
        if seed is not None:
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        _, candidates = self.predict_prepared(
            scene_inputs["semantic_image"], scene_inputs["observed_maps"],
            scene_inputs.get("pixel_to_world"),
            down_factor=float(scene_inputs.get("down_factor", 8.0)))
        return candidates
