"""Differentiable fixed-budget token selection through frozen decoder and task model."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from semantic_ai.burn_scar_model import dice_loss
from token_selection.learned_mode_selector import topk_mask


def straight_through_topk(logits: torch.Tensor, keep_ratio: float, temperature: float) -> torch.Tensor:
    """Use an exact forward token count and a smooth backward approximation."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    hard = topk_mask(logits, keep_ratio).to(logits.dtype)
    count = int(hard[0].sum().item())
    threshold = torch.topk(logits.detach().flatten(1), count, dim=1).values[:, -1]
    soft = torch.sigmoid((logits - threshold[:, None, None]) / temperature)
    return hard + soft - soft.detach()


def decode_gated_tokens(
    vqvae: torch.nn.Module, token_ids: torch.Tensor, gate: torch.Tensor
) -> torch.Tensor:
    """Decode selected code embeddings, replacing unselected tokens by the modal code."""
    if token_ids.shape != gate.shape or token_ids.ndim != 3:
        raise ValueError("token IDs and gates must share [B,H,W] shape")
    batch, height, width = token_ids.shape
    codebook = vqvae.vq.embedding.weight
    embeddings = F.embedding(token_ids, codebook)
    flat = token_ids.flatten(1)
    fallbacks = torch.stack([
        torch.bincount(ids, minlength=codebook.shape[0]).argmax() for ids in flat
    ])
    fallback_embedding = F.embedding(fallbacks, codebook).view(batch, 1, 1, -1)
    latent = fallback_embedding + gate.unsqueeze(-1) * (embeddings - fallback_embedding)
    latent = latent.permute(0, 3, 1, 2).contiguous()
    return vqvae.decoder(vqvae.post_vq(latent))


def task_aligned_loss(
    detector: torch.nn.Module,
    reconstruction: torch.Tensor,
    original: torch.Tensor,
    burn_mask: torch.Tensor,
    reconstruction_weight: float = 0.05,
) -> tuple[torch.Tensor, dict[str, float]]:
    rgb = (reconstruction + 1.0) * 0.5
    detector_input = F.interpolate(rgb, size=(256, 256), mode="bilinear", align_corners=False)
    mask = F.interpolate(burn_mask, size=(256, 256), mode="nearest")
    logits = detector(detector_input)
    bce = F.binary_cross_entropy_with_logits(logits, mask, pos_weight=torch.tensor(5.0, device=logits.device))
    dice = dice_loss(logits, mask)
    reconstruction_l1 = F.l1_loss(reconstruction, original)
    loss = bce + dice + reconstruction_weight * reconstruction_l1
    return loss, {
        "bce": float(bce.detach()),
        "dice_loss": float(dice.detach()),
        "reconstruction_l1": float(reconstruction_l1.detach()),
    }
