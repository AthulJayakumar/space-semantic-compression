"""Rank tokens by their codebook distance from the decoder fallback code."""

from __future__ import annotations

import numpy as np
import torch


def decoder_impact_scores(tokens: torch.Tensor, embedding: torch.Tensor) -> np.ndarray:
    """Cheap latent-space proxy for the effect of replacing a token by fallback."""
    if tokens.ndim != 3 or tokens.shape[0] != 1:
        raise ValueError("Decoder-impact ranking accepts one encoded image")
    codes = tokens.detach().reshape(-1).long()
    values, counts = torch.unique(codes, return_counts=True)
    fallback = values[counts.argmax()]
    with torch.no_grad():
        impact = torch.linalg.vector_norm(embedding[codes] - embedding[fallback], dim=1)
    scores = impact.reshape(tokens.shape[-2:]).cpu().numpy().astype("float32")
    low, high = float(scores.min()), float(scores.max())
    return (scores - low) / (high - low) if high - low > 1e-8 else np.zeros_like(scores)


def blended_decoder_aware_scores(utility_scores: np.ndarray, impact_scores: np.ndarray) -> np.ndarray:
    if utility_scores.shape != impact_scores.shape:
        raise ValueError("Utility and decoder-impact scores must share a token grid")
    return 0.5 * utility_scores.astype("float32") + 0.5 * impact_scores.astype("float32")
