"""Tests for SUS-aligned selector supervision and data separation."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from scripts.train_sus_aligned_selector import geographic_internal_split
from token_selection.sus_target import differentiable_sus_loss


def test_sus_surrogate_prefers_reference_detector_response():
    reference = torch.tensor([[[[0.9, 0.8], [0.1, 0.1]]]])
    region = (reference >= 0.5).float()
    truth = region.clone()
    target = torch.ones((1, 3, 2, 2)) * 0.5
    matching_loss, matching = differentiable_sus_loss(
        reference,
        reference,
        region,
        truth,
        target,
        target,
        threshold=0.5,
        reconstruction_weight=0.0,
    )
    degraded = reference * 0.1
    degraded_loss, degraded_components = differentiable_sus_loss(
        degraded,
        reference,
        region,
        truth,
        target,
        target,
        threshold=0.5,
        reconstruction_weight=0.0,
    )
    assert degraded_loss > matching_loss
    assert matching["surrogate_sus"] > degraded_components["surrogate_sus"]


def test_internal_selector_split_uses_disjoint_geographic_groups():
    items = [
        SimpleNamespace(sample_id=f"sample-{index}", geographic_group=f"group-{index // 2}")
        for index in range(20)
    ]
    train, validation = geographic_internal_split(items, validation_fraction=0.2, seed=42)
    train_groups = {item.geographic_group for item in train}
    validation_groups = {item.geographic_group for item in validation}
    assert train_groups.isdisjoint(validation_groups)
    assert len(train) + len(validation) == len(items)
