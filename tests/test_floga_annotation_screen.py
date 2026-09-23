"""Check conservative footprint screening and Sentinel acquisition identity."""

from __future__ import annotations

import pytest

from scripts.screen_floga_annotations import (
    bbox_center_radius, separation_lower_bound_km, sentinel_acquisition_key,
)


def test_sentinel_key_ignores_processing_baseline_but_keeps_scene_identity() -> None:
    old = "S2A_MSIL2A_20170916T092031_N0205_R093_T34SEG_20170916T092843"
    new = "S2A_MSIL2A_20170916T092031_N0500_R093_T34SEG_20230201T000000"
    other_tile = "S2A_MSIL2A_20170916T092031_N0500_R093_T34SEH_20230201T000000"
    assert sentinel_acquisition_key(old) == sentinel_acquisition_key(new)
    assert sentinel_acquisition_key(old) != sentinel_acquisition_key(other_tile)
    assert sentinel_acquisition_key("no product") is None


def test_bbox_distance_lower_bound_is_conservative() -> None:
    a = bbox_center_radius((20.0, 35.0, 20.1, 35.1))
    b = bbox_center_radius((20.05, 35.05, 20.15, 35.15))
    assert separation_lower_bound_km(a, b) == pytest.approx(0)
    far = bbox_center_radius((25.0, 40.0, 25.1, 40.1))
    assert separation_lower_bound_km(a, far) > 100
    with pytest.raises(ValueError):
        bbox_center_radius((25.0, 40.0, 20.0, 35.0))
