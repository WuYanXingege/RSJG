"""Standalone native endpoint provider used by the dependency audit.

This module deliberately has no import from a GDTS/EBJD source tree.  It
accepts the already prepared semantic image and observed-trajectory maps,
strict-loads only the goal U-Net state, and reproduces native TTST.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as TF


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.ReLU(True))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.block(value)


class Encoder(nn.Module):
    def __init__(self, channels) -> None:
        super().__init__()
        self.enc_blocks = nn.ModuleList([
            DoubleConv(channels[i], channels[i + 1])
            for i in range(len(channels) - 1)])
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, value: torch.Tensor) -> list[torch.Tensor]:
        features = []
        for block in self.enc_blocks:
            value = block(value); features.append(value); value = self.pool(value)
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
    def __init__(self, channels) -> None:
        super().__init__()
        self.upconvs = nn.ModuleList([
            UpConv(channels[i], channels[i + 1])
            for i in range(len(channels) - 1)])
        self.dec_blocks = nn.ModuleList([
            DoubleConv(2 * channels[i + 1], channels[i + 1])
            for i in range(len(channels) - 1)])

    def forward(self, value: torch.Tensor, features) -> torch.Tensor:
        for up, block, skip in zip(self.upconvs, self.dec_blocks, features):
            value = up(value)
            if skip.shape[-2:] != value.shape[-2:]:
                skip = TF.CenterCrop(list(value.shape[-2:]))(skip)
            value = block(torch.cat((value, skip), dim=1))
        return value


class GoalUNet(nn.Module):
    def __init__(self, enc=(14, 32, 32, 64, 64, 64),
                 dec=(64, 64, 64, 32, 32), out_channels=12) -> None:
        super().__init__()
        self.encoder = Encoder(enc)
        self.decoder = Decoder(dec)
        self.head = nn.Module()
        self.head.out_layer = nn.Conv2d(dec[-1], out_channels, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        height, width = value.shape[-2:]
        features = self.encoder(value)
        value = self.decoder(features[-1], features[-2::-1])
        return F.interpolate(self.head.out_layer(value), (height, width))


def _sample(probability: torch.Tensor, num_samples: int = 1000) -> torch.Tensor:
    flat = probability.reshape(probability.shape[0] * probability.shape[1], -1)
    threshold = flat.max(dim=1, keepdim=True).values
    filtered = flat * (flat >= threshold * 0.05).to(flat.dtype)
    row_sum = filtered.sum(dim=1, keepdim=True)
    normalized = filtered / row_sum.clamp_min(torch.finfo(flat.dtype).tiny)
    fallback = torch.zeros_like(flat)
    fallback.scatter_(1, flat.argmax(dim=1, keepdim=True), 1.0)
    normalized = torch.where(row_sum > 0, normalized, fallback)
    indices = torch.multinomial(normalized, num_samples, replacement=True)
    indices = indices.reshape(probability.shape[0], probability.shape[1], -1)
    result = indices.unsqueeze(-1).repeat(1, 1, 1, 2).float()
    result[..., 0] = result[..., 0].remainder(probability.shape[3])
    result[..., 1] = torch.floor(result[..., 1] / probability.shape[3])
    return result


def _kmeans(points: torch.Tensor, clusters: int) -> torch.Tensor:
    indices = np.random.choice(len(points), clusters, replace=False)
    centers = points[indices].clone().float()
    for _ in range(1000):
        distances = ((points[:, None].float() - centers[None]) ** 2).sum(-1)
        assignment = distances.argmin(dim=1)
        previous = centers.clone()
        for index in range(clusters):
            selected = points[assignment == index]
            if selected.shape[0] == 0:
                selected = points[torch.randint(len(points), (1,))]
            centers[index] = selected.float().mean(dim=0)
        shift = torch.sqrt(((centers - previous) ** 2).sum(1)).sum()
        if shift ** 2 < 0.001:
            break
    return centers


def native_ttst(probability: torch.Tensor, num_goals: int = 20) -> torch.Tensor:
    samples = _sample(probability).permute(2, 0, 1, 3)
    centers = torch.stack([
        _kmeans(samples[:, person, 0], num_goals)
        for person in range(samples.shape[1])])
    # [K,N,2]. The formal exporter keeps these first 20 cluster centers;
    # native code appends argmax as item 21, which is therefore omitted.
    return centers.permute(1, 0, 2)


class NativeGoalProvider(nn.Module):
    def __init__(self, state: dict[str, torch.Tensor]) -> None:
        super().__init__()
        enc = (14, 32, 32, 64, 64, 64)
        dec = (64, 64, 64, 32, 32)
        self.goal_module = GoalUNet(enc, dec, 12)
        self.goal_module.load_state_dict(state, strict=True)
        self.goal_module.eval()

    @torch.inference_mode()
    def forward(self, semantic_image: torch.Tensor,
                observed_maps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        agents = observed_maps.shape[0]
        image = semantic_image.unsqueeze(0).repeat(agents, 1, 1, 1)
        logits = self.goal_module(torch.cat((image, observed_maps[:, :8]), dim=1))
        goals = native_ttst(torch.sigmoid(logits[:, -1:]), 20)
        return logits, goals
