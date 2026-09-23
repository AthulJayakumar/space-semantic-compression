"""Tests for decoder-sensitive token-impact supervision."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from token_selection.impact_target import decoder_replacement_impact, robust_normalize


class TinyDecoderModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vq = nn.Module()
        self.vq.embedding = nn.Embedding(3, 1)
        with torch.no_grad():
            self.vq.embedding.weight[:, 0] = torch.tensor([0.0, 1.0, 2.0])
        self.post_vq = nn.Identity()
        self.decoder = nn.Identity()


def test_replacement_impact_prioritizes_token_whose_fallback_increases_loss():
    model = TinyDecoderModel()
    tokens = torch.tensor([[[1, 2], [1, 2]]])
    target = torch.full((1, 1, 2, 2), 2.0)
    mission_mask = torch.ones((1, 1, 2, 2))
    impact, loss = decoder_replacement_impact(
        model, tokens, target, mission_mask, mission_weight=1.0, context_pixels=0
    )
    assert loss > 0.0
    assert impact[0, 1] > impact[0, 0]
    assert impact[1, 1] > impact[1, 0]


def test_robust_normalize_clips_outliers_and_preserves_shape():
    values = np.asarray([[0.0, 1.0], [2.0, 100.0]], dtype="float32")
    normalized = robust_normalize(values, upper_quantile=0.75)
    assert normalized.shape == values.shape
    assert float(normalized.min()) == 0.0
    assert float(normalized.max()) == 1.0
