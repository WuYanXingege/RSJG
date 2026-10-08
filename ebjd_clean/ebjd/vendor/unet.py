"""Small trainable U-Net matching the channel contract in the EBJD design."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat((self.up(x), skip), dim=1))


class TrainableMapUNet(nn.Module):
    """Five DoubleConv encoders, four up blocks, F32 and 12 logits."""

    def __init__(self, input_channels: int = 14) -> None:
        super().__init__()
        self.encoders = nn.ModuleList([
            DoubleConv(input_channels, 32),
            DoubleConv(32, 32),
            DoubleConv(32, 64),
            DoubleConv(64, 64),
            DoubleConv(64, 64),
        ])
        self.pool = nn.MaxPool2d(2)
        self.up_blocks = nn.ModuleList([
            UpBlock(64, 64, 64),
            UpBlock(64, 64, 64),
            UpBlock(64, 32, 32),
            UpBlock(32, 32, 32),
        ])
        self.future_logits = nn.Conv2d(32, 12, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skips: list[torch.Tensor] = []
        for index, block in enumerate(self.encoders):
            x = block(x)
            if index < len(self.encoders) - 1:
                skips.append(x)
                x = self.pool(x)
        for block, skip in zip(self.up_blocks, reversed(skips), strict=True):
            x = block(x, skip)
        return x, self.future_logits(x)
