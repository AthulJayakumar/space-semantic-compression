"""Training and calibration utilities for the burn-scar utility detector."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def focal_tversky_hard_negative_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    focal_gamma: float = 2.0,
    positive_alpha: float = 0.70,
    false_positive_weight: float = 0.30,
    false_negative_weight: float = 0.70,
    hard_negative_fraction: float = 0.10,
    hard_negative_weight: float = 0.20,
) -> torch.Tensor:
    """Combine focal classification, Tversky overlap, and hard negatives."""
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probability = torch.sigmoid(logits)
    correct_probability = probability * targets + (1.0 - probability) * (1.0 - targets)
    alpha = positive_alpha * targets + (1.0 - positive_alpha) * (1.0 - targets)
    focal = (alpha * (1.0 - correct_probability).pow(focal_gamma) * bce).mean()

    dimensions = (1, 2, 3)
    true_positive = (probability * targets).sum(dim=dimensions)
    false_positive = (probability * (1.0 - targets)).sum(dim=dimensions)
    false_negative = ((1.0 - probability) * targets).sum(dim=dimensions)
    epsilon = 1e-6
    tversky = (true_positive + epsilon) / (
        true_positive
        + false_positive_weight * false_positive
        + false_negative_weight * false_negative
        + epsilon
    )
    tversky_loss = 1.0 - tversky.mean()

    negative_losses = (bce * (targets < 0.5)).flatten(1)
    hard_count = max(1, int(round(negative_losses.shape[1] * hard_negative_fraction)))
    hard_negative = torch.topk(negative_losses, k=hard_count, dim=1).values.mean()
    return focal + tversky_loss + hard_negative_weight * hard_negative


def expected_calibration_error(
    probabilities: np.ndarray,
    targets: np.ndarray,
    bins: int = 15,
) -> float:
    """Calculate binary expected calibration error over equally spaced bins."""
    prediction = np.asarray(probabilities, dtype="float64").reshape(-1)
    truth = np.asarray(targets, dtype="float64").reshape(-1)
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for index in range(bins):
        lower, upper = boundaries[index], boundaries[index + 1]
        selected = (prediction >= lower) & (prediction < upper if index < bins - 1 else prediction <= upper)
        if not selected.any():
            continue
        confidence = float(prediction[selected].mean())
        accuracy = float(truth[selected].mean())
        error += float(selected.mean()) * abs(confidence - accuracy)
    return error


def select_temperature(
    logits: np.ndarray,
    targets: np.ndarray,
    candidates: np.ndarray | None = None,
) -> tuple[float, float]:
    """Select a scalar temperature by validation binary cross-entropy."""
    temperatures = candidates if candidates is not None else np.linspace(0.5, 3.0, 51)
    logit_tensor = torch.from_numpy(np.asarray(logits, dtype="float32"))
    target_tensor = torch.from_numpy(np.asarray(targets, dtype="float32"))
    scored = []
    for temperature in temperatures:
        loss = F.binary_cross_entropy_with_logits(logit_tensor / float(temperature), target_tensor)
        scored.append((float(loss), float(temperature)))
    loss, temperature = min(scored)
    return temperature, loss


def probabilities_from_logits(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    values = np.asarray(logits, dtype="float64") / max(float(temperature), 1e-6)
    return (1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))).astype("float32")
