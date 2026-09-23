"""Tests for calibrated burn-scar detector training utilities."""

from __future__ import annotations

import numpy as np
import torch

from semantic_ai.burn_scar_training import (
    expected_calibration_error,
    focal_tversky_hard_negative_loss,
    probabilities_from_logits,
    select_temperature,
)


def test_focal_tversky_loss_is_finite_and_differentiable():
    logits = torch.zeros((2, 1, 8, 8), requires_grad=True)
    targets = torch.zeros_like(logits)
    targets[:, :, 2:6, 2:6] = 1.0
    loss = focal_tversky_hard_negative_loss(logits, targets)
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_perfect_probabilities_have_near_zero_calibration_error():
    probabilities = np.asarray([0.0, 1.0, 0.0, 1.0], dtype="float32")
    targets = probabilities.copy()
    assert expected_calibration_error(probabilities, targets) < 1e-8


def test_temperature_selection_and_probability_conversion_are_bounded():
    logits = np.asarray([-4.0, 4.0], dtype="float32")
    targets = np.asarray([0.0, 1.0], dtype="float32")
    temperature, loss = select_temperature(logits, targets, np.asarray([0.5, 1.0, 2.0]))
    probabilities = probabilities_from_logits(logits, temperature)
    assert temperature in {0.5, 1.0, 2.0}
    assert loss >= 0.0
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
