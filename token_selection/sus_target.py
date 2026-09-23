"""Differentiable mission-utility targets for token-selector training."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from token_selection.impact_target import robust_normalize


SUS_WEIGHTS = (0.4, 0.3, 0.2, 0.1)


def sus_aligned_replacement_impact(
    model,
    detector,
    token_ids: torch.Tensor,
    target_image: torch.Tensor,
    ground_truth_mask: torch.Tensor,
    *,
    detector_threshold: float = 0.5,
    detector_input_size: int = 256,
    integration_steps: int = 4,
    reconstruction_weight: float = 0.15,
    region_temperature: float = 0.08,
) -> tuple[np.ndarray, dict[str, float]]:
    """Attribute each token's reduction of a differentiable SUS surrogate.

    The formal evaluation-time SUS remains unchanged. This surrogate follows
    the same component weights while replacing connected-component counting
    and hard thresholding with differentiable occupancy approximations.
    """
    device = next(model.parameters()).device
    tokens = token_ids.to(device).long()
    target = target_image.to(device=device, dtype=torch.float32)
    truth = _mask_tensor(ground_truth_mask, target.shape[-2:], device)
    detector = detector.to(device).eval()
    detector.requires_grad_(False)

    with torch.no_grad():
        reference_probability = _detector_probability(detector, target, detector_input_size)
        truth_small = F.interpolate(truth, size=reference_probability.shape[-2:], mode="nearest")
        threshold = max(float(detector_threshold), float(torch.quantile(reference_probability, 0.75)))
        reference_region = (reference_probability >= threshold).to(reference_probability.dtype)

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
    gradient_sum = torch.zeros_like(full_embeddings)
    full_components: dict[str, float] = {}
    steps = max(2, int(integration_steps))

    with torch.enable_grad():
        for step in range(1, steps + 1):
            alpha = step / steps
            point = (baseline + alpha * latent_delta).detach().requires_grad_(True)
            reconstruction = model.decoder(model.post_vq(point))
            probability = _detector_probability(detector, reconstruction, detector_input_size)
            loss, components = differentiable_sus_loss(
                probability,
                reference_probability,
                reference_region,
                truth_small,
                reconstruction,
                target,
                threshold=threshold,
                region_temperature=region_temperature,
                reconstruction_weight=reconstruction_weight,
            )
            gradient_sum += torch.autograd.grad(loss, point, retain_graph=False, create_graph=False)[0]
            if step == steps:
                full_components = {key: float(value.detach().cpu()) for key, value in components.items()}

    attribution = latent_delta * (gradient_sum / steps)
    impact = (-attribution.sum(dim=1)).clamp_min(0.0)
    return robust_normalize(impact[0].detach().cpu().numpy()), full_components


def differentiable_sus_loss(
    probability: torch.Tensor,
    reference_probability: torch.Tensor,
    reference_region: torch.Tensor,
    truth_mask: torch.Tensor,
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    *,
    threshold: float,
    region_temperature: float = 0.08,
    reconstruction_weight: float = 0.15,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return loss and transparent proxy components on a zero-to-one scale."""
    epsilon = 1e-6
    detector_retention = torch.clamp(
        probability.sum() / reference_probability.sum().clamp_min(epsilon), max=1.0
    )
    soft_region = torch.sigmoid((probability - threshold) / max(region_temperature, 1e-3))
    object_retention = _soft_dice(soft_region, reference_region)

    relevance_support = truth_mask
    if float(relevance_support.sum().detach().cpu()) < 1.0:
        relevance_support = reference_region
    relevance_retention = torch.clamp(
        (probability * relevance_support).sum()
        / (reference_probability * relevance_support).sum().clamp_min(epsilon),
        max=1.0,
    )
    region_preservation = torch.clamp(
        (soft_region * reference_region).sum() / reference_region.sum().clamp_min(epsilon), max=1.0
    )
    sus_proxy = (
        SUS_WEIGHTS[0] * detector_retention
        + SUS_WEIGHTS[1] * object_retention
        + SUS_WEIGHTS[2] * relevance_retention
        + SUS_WEIGHTS[3] * region_preservation
    )
    pixel_weights = 0.25 + 0.75 * F.interpolate(truth_mask, size=target.shape[-2:], mode="nearest")
    reconstruction_loss = ((reconstruction - target).square() * pixel_weights).mean()
    weight = float(np.clip(reconstruction_weight, 0.0, 1.0))
    total_loss = (1.0 - weight) * (1.0 - sus_proxy) + weight * reconstruction_loss
    return total_loss, {
        "surrogate_sus": sus_proxy,
        "detector_retention_proxy": detector_retention,
        "object_retention_proxy": object_retention,
        "relevance_retention_proxy": relevance_retention,
        "region_preservation_proxy": region_preservation,
        "reconstruction_loss": reconstruction_loss,
        "total_loss": total_loss,
    }


def _detector_probability(detector, image: torch.Tensor, input_size: int) -> torch.Tensor:
    resized = F.interpolate(image, size=(input_size, input_size), mode="bilinear", align_corners=False)
    return torch.sigmoid(detector(resized))


def _mask_tensor(mask: torch.Tensor, size: tuple[int, int], device: torch.device) -> torch.Tensor:
    values = mask.to(device=device, dtype=torch.float32)
    if values.ndim == 2:
        values = values.unsqueeze(0).unsqueeze(0)
    elif values.ndim == 3:
        values = values.unsqueeze(1)
    if values.shape[-2:] != size:
        values = F.interpolate(values, size=size, mode="nearest")
    return values


def _soft_dice(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    epsilon = 1e-6
    intersection = (prediction * target).sum()
    return (2.0 * intersection + epsilon) / (prediction.sum() + target.sum() + epsilon)
