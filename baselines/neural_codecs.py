"""baselines.neural_codecs

Plain-English purpose: Reference codecs and baseline model wrappers used for fair comparison.

This file is part of the public Space Semantic Compression research demo.
It is documented at module level so researchers, supervisors, and non-specialist
readers can understand where the file fits before reading implementation details.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class AutoencoderCodec(nn.Module):
    def __init__(self, latent_channels: int = 64) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(32, latent_channels, 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(latent_channels, latent_channels, 4, 2, 1),
            nn.SiLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, latent_channels, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(latent_channels, 32, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        return self.decoder(z), z


class VariationalAutoencoderCodec(nn.Module):
    def __init__(self, latent_channels: int = 64) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(32, latent_channels, 4, 2, 1),
            nn.SiLU(),
        )
        self.mu = nn.Conv2d(latent_channels, latent_channels, 3, padding=1)
        self.logvar = nn.Conv2d(latent_channels, latent_channels, 3, padding=1)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 32, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.stem(x)
        mu = self.mu(h)
        logvar = self.logvar(h).clamp(-8.0, 8.0)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) if self.training else torch.zeros_like(std)
        z = mu + eps * std
        return self.decoder(z), z, mu, logvar

    def kl_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


class LightweightCNNCodec(nn.Module):
    def __init__(self, width: int = 32) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(width, width, 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(width, width, 4, 2, 1),
            nn.SiLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(width, width, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(width, width, 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(width, 3, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        return self.decoder(z), z


def reconstruction_loss(x: torch.Tensor, xhat: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(xhat, x) + 0.25 * F.mse_loss(xhat, x)
