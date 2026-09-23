"""Compact supervised burn-scar utility segmentation model."""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class BurnScarUtilityNet(nn.Module):
    """Small U-Net used to produce a 0-1 burn-scar utility map."""

    def __init__(self, base_channels: int = 16) -> None:
        super().__init__()
        self.encoder_1 = ConvBlock(3, base_channels)
        self.encoder_2 = ConvBlock(base_channels, base_channels * 2)
        self.bottleneck = ConvBlock(base_channels * 2, base_channels * 4)
        self.pool = nn.MaxPool2d(2)
        self.up_2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.decoder_2 = ConvBlock(base_channels * 4, base_channels * 2)
        self.up_1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.decoder_1 = ConvBlock(base_channels * 2, base_channels)
        self.output = nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        level_1 = self.encoder_1(image)
        level_2 = self.encoder_2(self.pool(level_1))
        latent = self.bottleneck(self.pool(level_2))
        decoded_2 = self.decoder_2(torch.cat([self.up_2(latent), level_2], dim=1))
        decoded_1 = self.decoder_1(torch.cat([self.up_1(decoded_2), level_1], dim=1))
        return self.output(decoded_1)


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    intersection = (probabilities * targets).sum(dim=(1, 2, 3))
    denominator = probabilities.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return 1.0 - ((2.0 * intersection + epsilon) / (denominator + epsilon)).mean()
