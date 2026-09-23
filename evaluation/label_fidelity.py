"""Independent burn-scar mask overlap after image reconstruction."""

from __future__ import annotations

import numpy as np


def binary_overlap(predicted: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    if predicted.shape != truth.shape:
        raise ValueError("Prediction and ground-truth masks must have the same shape")
    predicted = np.asarray(predicted, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    intersection = int(np.count_nonzero(predicted & truth))
    predicted_count = int(np.count_nonzero(predicted))
    truth_count = int(np.count_nonzero(truth))
    union = predicted_count + truth_count - intersection
    dice = 2 * intersection / (predicted_count + truth_count) if predicted_count + truth_count else 1.0
    iou = intersection / union if union else 1.0
    return float(dice), float(iou)
