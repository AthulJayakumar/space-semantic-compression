"""Decoder-side latent interpolation using only transmitted codes and mask."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt


def decode_spatially_filled_tokens(
    vqvae: torch.nn.Module,
    received_tokens: torch.Tensor,
    keep_mask: np.ndarray,
    iterations: int = 4,
) -> torch.Tensor:
    """Diffuse neighboring received code embeddings into missing positions."""
    if received_tokens.ndim != 3 or received_tokens.shape[0] != 1:
        raise ValueError("Expected one [1,H,W] received token grid")
    if keep_mask.shape != tuple(received_tokens.shape[-2:]) or not keep_mask.any():
        raise ValueError("Keep mask must match the token grid and contain a received code")
    if iterations < 1:
        raise ValueError("At least one interpolation iteration is required")
    codebook = vqvae.vq.embedding.weight
    embeddings = F.embedding(received_tokens.to(codebook.device), codebook)
    values = embeddings.permute(0, 3, 1, 2).contiguous()
    known = torch.as_tensor(keep_mask, device=values.device, dtype=values.dtype)[None, None]
    for _ in range(iterations):
        neighbor_sum = F.avg_pool2d(values * known, 3, stride=1, padding=1) * 9.0
        neighbor_count = F.avg_pool2d(known, 3, stride=1, padding=1) * 9.0
        fillable = (known == 0) & (neighbor_count > 0)
        values = torch.where(fillable, neighbor_sum / neighbor_count.clamp_min(1.0), values)
        known = torch.where(fillable, torch.ones_like(known), known)
    return vqvae.decoder(vqvae.post_vq(values))


def decode_nearest_filled_tokens(
    vqvae: torch.nn.Module, received_tokens: torch.Tensor, keep_mask: np.ndarray
) -> torch.Tensor:
    """Copy the nearest transmitted code into each missing token position."""
    if received_tokens.ndim != 3 or received_tokens.shape[0] != 1:
        raise ValueError("Expected one [1,H,W] received token grid")
    if keep_mask.shape != tuple(received_tokens.shape[-2:]) or not keep_mask.any():
        raise ValueError("Keep mask must match the token grid and contain a received code")
    nearest = distance_transform_edt(~keep_mask, return_distances=False, return_indices=True)
    indices = tuple(torch.as_tensor(axis, device=received_tokens.device) for axis in nearest)
    filled = received_tokens[0][indices].unsqueeze(0)
    return vqvae.decode(filled)
