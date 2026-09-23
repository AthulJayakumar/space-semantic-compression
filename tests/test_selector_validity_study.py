"""Tests for independent selector and SUS validity diagnostics."""

from __future__ import annotations

import numpy as np

from scripts.run_selector_validity_study import downsample_mask, fire_size_class, segmentation_metrics


def test_downsample_mask_preserves_fractional_region_mass():
    mask = np.zeros((4, 4), dtype="float32")
    mask[:2, :2] = 1.0
    reduced = downsample_mask(mask, (2, 2))
    assert reduced.shape == (2, 2)
    assert float(reduced.sum()) == 1.0


def test_segmentation_metrics_are_one_for_exact_mask():
    truth = np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype="float32")
    metrics = segmentation_metrics(truth, truth, threshold=0.5)
    assert np.isclose(metrics["dice"], 1.0)
    assert np.isclose(metrics["iou"], 1.0)


def test_fire_size_classes_have_stable_boundaries():
    assert fire_size_class(0.049) == "small_lt_5pct"
    assert fire_size_class(0.05) == "medium_5_25pct"
    assert fire_size_class(0.25) == "large_ge_25pct"
