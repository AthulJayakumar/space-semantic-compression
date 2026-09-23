"""Tests for matched-byte calibration and leakage-safe splitting."""

from __future__ import annotations

import numpy as np
from PIL import Image

from evaluation.dataset_validity import deterministic_group_split, estimate_cloud_fraction
from evaluation.matched_rate import EncodedCandidate, MatchedRateBenchmark, bootstrap_mean_ci


def test_group_split_is_stable_and_disjoint_by_construction():
    assert deterministic_group_split("10SFF") == deterministic_group_split("10SFF")
    assert deterministic_group_split("10SFF") in {"train", "validation", "test"}


def test_cloud_proxy_detects_bright_low_saturation_pixels():
    image = Image.new("RGB", (16, 16), (240, 240, 240))
    assert estimate_cloud_fraction(image) == 1.0


def test_calibration_never_exceeds_budget(tmp_path):
    runner = object.__new__(MatchedRateBenchmark)

    def encode(parameter: int) -> EncodedCandidate:
        payload = bytes(parameter * 10)
        return EncodedCandidate(payload, Image.new("RGB", (2, 2)), float(parameter))

    selected = runner._calibrate(46, 1, 10, encode, increasing=True)
    assert selected.size == 40


def test_bootstrap_ci_contains_sample_mean():
    values = np.asarray([1.0, 2.0, 3.0, 4.0])
    low, high = bootstrap_mean_ci(values, samples=250)
    assert low <= values.mean() <= high


def test_detector_context_expands_high_utility_neighborhood():
    values = np.zeros((5, 5), dtype="float32")
    values[2, 2] = 1.0
    expanded = MatchedRateBenchmark._max_filter(values, radius=1)
    assert expanded.sum() == 9.0
    assert expanded[1:4, 1:4].all()


def test_detector_only_composition_does_not_reintroduce_heuristic_map(tmp_path):
    runner = object.__new__(MatchedRateBenchmark)
    runner.utility_map_mode = "detector_only"
    runner.semantic_blend_weight = 0.25
    runner.context_radius = 1
    detector = np.asarray([[0.0, 0.5], [1.0, 0.25]], dtype="float32")
    semantic = np.ones((2, 2), dtype="float32")
    composed = runner.compose_utility_map(detector, semantic)
    np.testing.assert_allclose(composed, detector)
