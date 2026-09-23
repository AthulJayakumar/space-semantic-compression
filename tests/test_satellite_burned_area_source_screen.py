"""Check event parsing and geographic-distance screening primitives."""

from __future__ import annotations

import pytest

from scripts.screen_satellite_burned_area_source import event_code, haversine_km


def test_event_code_is_taken_from_folder_prefix() -> None:
    assert event_code("EMSR214_05LELAVANDOU_02GRADING_MAP_v1_vector") == "EMSR214"
    with pytest.raises(ValueError):
        event_code("LELAVANDOU_02GRADING_MAP")


def test_haversine_distance_is_symmetric_and_zero_at_same_point() -> None:
    assert haversine_km(6.0, 43.0, 6.0, 43.0) == pytest.approx(0)
    forward = haversine_km(6.0, 43.0, 7.0, 44.0)
    assert forward == pytest.approx(haversine_km(7.0, 44.0, 6.0, 43.0))
    assert 130 < forward < 140
