"""token_selection.learned_mode_selector

Plain-English purpose: Optional trainable token-priority model conditioned on mission mode.

This module is the first step toward improving the model itself with mode data.
It does not replace the validated rule-based selector. Instead, it provides a
small PyTorch model that can learn token priorities from VQ-VAE tokens,
semantic utility, local entropy/detail features, and a mode label.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


MODE_TO_INDEX = {"mission_utility": 0, "reconstruction_balanced": 1}
INDEX_TO_MODE = {value: key for key, value in MODE_TO_INDEX.items()}


@dataclass(frozen=True)
class ModeConditionedSelectorConfig:
    codebook_size: int = 8192
    token_embedding_dim: int = 16
    mode_embedding_dim: int = 4
    hidden_channels: int = 32


class ModeConditionedTokenScorer(nn.Module):
    """Predict token priority scores while conditioning on the requested operating mode."""

    def __init__(self, config: ModeConditionedSelectorConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModeConditionedSelectorConfig()
        self.token_embedding = nn.Embedding(self.config.codebook_size, self.config.token_embedding_dim)
        self.mode_embedding = nn.Embedding(len(MODE_TO_INDEX), self.config.mode_embedding_dim)
        input_channels = self.config.token_embedding_dim + self.config.mode_embedding_dim + 3
        self.scorer = nn.Sequential(
            nn.Conv2d(input_channels, self.config.hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.config.hidden_channels, self.config.hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.config.hidden_channels, 1, kernel_size=1),
        )

    def forward(
        self,
        token_ids: torch.Tensor,
        utility_map: torch.Tensor,
        entropy_map: torch.Tensor,
        detail_map: torch.Tensor,
        mode: str | torch.Tensor,
    ) -> torch.Tensor:
        token_ids = self._token_grid(token_ids)
        batch, height, width = token_ids.shape
        device = token_ids.device

        token_features = self.token_embedding(token_ids.clamp_min(0) % self.config.codebook_size)
        token_features = token_features.permute(0, 3, 1, 2).contiguous()
        utility = self._feature_grid(utility_map, batch, height, width, device)
        entropy = self._feature_grid(entropy_map, batch, height, width, device)
        detail = self._feature_grid(detail_map, batch, height, width, device)
        mode_index = self._mode_index(mode, batch, device)
        mode_features = self.mode_embedding(mode_index).view(batch, -1, 1, 1).expand(-1, -1, height, width)

        features = torch.cat([token_features, utility, entropy, detail, mode_features], dim=1)
        return self.scorer(features).squeeze(1)

    def score(
        self,
        token_ids: torch.Tensor,
        utility_map: torch.Tensor,
        entropy_map: torch.Tensor,
        detail_map: torch.Tensor,
        mode: str | torch.Tensor,
    ) -> torch.Tensor:
        return torch.sigmoid(self.forward(token_ids, utility_map, entropy_map, detail_map, mode))

    def select(
        self,
        token_ids: torch.Tensor,
        utility_map: torch.Tensor,
        entropy_map: torch.Tensor,
        detail_map: torch.Tensor,
        mode: str | torch.Tensor,
        keep_ratio: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.score(token_ids, utility_map, entropy_map, detail_map, mode)
        keep_mask = topk_mask(scores, keep_ratio)
        return keep_mask, scores

    def _token_grid(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim == 2:
            token_ids = token_ids.unsqueeze(0)
        if token_ids.ndim == 4 and token_ids.shape[1] == 1:
            token_ids = token_ids[:, 0]
        if token_ids.ndim != 3:
            raise ValueError("token_ids must have shape [H,W], [B,H,W], or [B,1,H,W].")
        return token_ids.long()

    def _feature_grid(self, values: torch.Tensor, batch: int, height: int, width: int, device: torch.device) -> torch.Tensor:
        values = values.to(device=device, dtype=torch.float32)
        if values.ndim == 2:
            values = values.unsqueeze(0).unsqueeze(0)
        elif values.ndim == 3:
            values = values.unsqueeze(1)
        elif values.ndim != 4:
            raise ValueError("feature maps must have shape [H,W], [B,H,W], or [B,1,H,W].")
        if values.shape[-2:] != (height, width):
            values = torch.nn.functional.interpolate(values, size=(height, width), mode="bilinear", align_corners=False)
        if values.shape[0] == 1 and batch > 1:
            values = values.expand(batch, -1, -1, -1)
        if values.shape[0] != batch:
            raise ValueError("feature batch size must match token batch size.")
        return values

    def _mode_index(self, mode: str | torch.Tensor, batch: int, device: torch.device) -> torch.Tensor:
        if isinstance(mode, str):
            if mode not in MODE_TO_INDEX:
                raise ValueError(f"Unsupported token selection mode: {mode}")
            return torch.full((batch,), MODE_TO_INDEX[mode], device=device, dtype=torch.long)
        mode = mode.to(device=device, dtype=torch.long).reshape(-1)
        if mode.numel() == 1 and batch > 1:
            mode = mode.expand(batch)
        if mode.numel() != batch:
            raise ValueError("mode tensor must contain one value per batch item.")
        return mode


def topk_mask(scores: torch.Tensor, keep_ratio: float) -> torch.Tensor:
    if scores.ndim == 2:
        scores = scores.unsqueeze(0)
    if scores.ndim != 3:
        raise ValueError("scores must have shape [H,W] or [B,H,W].")
    batch, height, width = scores.shape
    total = height * width
    keep_count = max(1, int(round(total * float(np.clip(keep_ratio, 0.01, 1.0)))))
    flat_scores = scores.reshape(batch, total)
    indices = torch.topk(flat_scores, k=keep_count, dim=1).indices
    mask = torch.zeros_like(flat_scores, dtype=torch.bool)
    mask.scatter_(1, indices, True)
    return mask.reshape(batch, height, width)


def local_token_entropy_numpy(tokens: torch.Tensor, shape: tuple[int, int]) -> np.ndarray:
    values = tokens.detach().cpu().numpy().reshape(shape).astype("int64")
    entropy = np.zeros(shape, dtype="float32")
    padded = np.pad(values, 1, mode="edge")
    for row in range(shape[0]):
        for col in range(shape[1]):
            window = padded[row : row + 3, col : col + 3].reshape(-1)
            _, counts = np.unique(window, return_counts=True)
            probs = counts.astype("float64") / counts.sum()
            entropy[row, col] = float(-np.sum(probs * np.log2(probs + 1e-12)))
    return normalize_numpy(entropy)


def normalize_numpy(values: np.ndarray) -> np.ndarray:
    values = values.astype("float32")
    lo = float(values.min()) if values.size else 0.0
    hi = float(values.max()) if values.size else 0.0
    if hi - lo < 1e-8:
        return np.zeros_like(values, dtype="float32")
    return (values - lo) / (hi - lo)
