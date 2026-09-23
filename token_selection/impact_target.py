"""Decoder-sensitive supervision targets for learned token selection."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def decoder_replacement_impact(
    model,
    token_ids: torch.Tensor,
    target_image: torch.Tensor,
    mission_mask: torch.Tensor,
    *,
    mission_weight: float = 0.8,
    context_pixels: int = 16,
    integration_steps: int = 4,
) -> tuple[np.ndarray, float]:
    """Attribute decoder loss reduction relative to the wire fallback-token grid.

    Integrated gradients follow the latent path from the all-fallback baseline
    to the full token embedding grid. This captures nonlinear decoder effects
    that a gradient evaluated only at the full reconstruction would miss.
    """
    tokens = token_ids.to(next(model.parameters()).device).long()
    target = target_image.to(tokens.device, dtype=torch.float32)
    mask = mission_mask.to(tokens.device, dtype=torch.float32)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.shape[-2:] != target.shape[-2:]:
        mask = F.interpolate(mask, size=target.shape[-2:], mode="nearest")
    if context_pixels > 0:
        kernel = 2 * int(context_pixels) + 1
        mask = F.max_pool2d(mask, kernel_size=kernel, stride=1, padding=context_pixels)

    flat_tokens = tokens.reshape(-1).detach().cpu().numpy()
    values, counts = np.unique(flat_tokens, return_counts=True)
    fallback_id = int(values[np.argmax(counts)])
    full_embeddings = model.vq.embedding(tokens.reshape(-1)).view(
        tokens.shape[0], tokens.shape[1], tokens.shape[2], -1
    )
    full_embeddings = full_embeddings.permute(0, 3, 1, 2).contiguous().detach()
    fallback = model.vq.embedding.weight[fallback_id].detach().view(1, -1, 1, 1)
    baseline = fallback.expand_as(full_embeddings)
    latent_delta = full_embeddings - baseline
    pixel_weights = (1.0 - mission_weight) + mission_weight * mask
    gradient_sum = torch.zeros_like(full_embeddings)
    full_loss = 0.0
    steps = max(2, int(integration_steps))
    with torch.enable_grad():
        for step in range(1, steps + 1):
            alpha = step / steps
            point = (baseline + alpha * latent_delta).detach().requires_grad_(True)
            reconstruction = model.decoder(model.post_vq(point))
            loss = ((reconstruction - target).square() * pixel_weights).mean()
            gradient_sum += torch.autograd.grad(loss, point, retain_graph=False, create_graph=False)[0]
            if step == steps:
                full_loss = float(loss.detach().cpu())

    attribution = latent_delta * (gradient_sum / steps)
    # Negative loss attribution means the token helps relative to fallback.
    impact = (-attribution.sum(dim=1)).clamp_min(0.0)
    impact = robust_normalize(impact[0].detach().cpu().numpy())
    return impact, full_loss


def robust_normalize(values: np.ndarray, upper_quantile: float = 0.995) -> np.ndarray:
    """Normalize a non-negative map while limiting isolated gradient outliers."""
    array = np.asarray(values, dtype="float32")
    array = np.maximum(array, 0.0)
    if not array.size:
        return array
    upper = float(np.quantile(array, upper_quantile))
    if upper < 1e-12:
        return np.zeros_like(array, dtype="float32")
    return np.clip(array / upper, 0.0, 1.0).astype("float32")
