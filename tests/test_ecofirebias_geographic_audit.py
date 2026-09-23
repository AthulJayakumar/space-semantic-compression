"""Location and train-only label-calibration checks."""

from __future__ import annotations

import numpy as np
from pyproj import Transformer

from scripts.audit_ecofirebias_replication import source_locations
from scripts.verify_ecofirebias_dnbr_encoding import best_byte_threshold, select_train_pairs


def test_source_locations_transform_projected_center() -> None:
    east, north = Transformer.from_crs("EPSG:4326", "EPSG:32630", always_xy=True).transform(-3.0, 40.0)
    row = {
        "epsg": "32630", "projected_center_east_m": str(east),
        "projected_center_north_m": str(north), "sample_id": "sample",
    }
    lon, lat = source_locations([row])[0]
    assert abs(lon + 3.0) < 1e-6
    assert abs(lat - 40.0) < 1e-6


def test_train_selection_keeps_complete_pairs() -> None:
    rows = [
        {"split": split, "event_id": event, "kind": kind, "continent": "Africa"}
        for split, event in (("train", "a"), ("test", "b"), ("train", "c"))
        for kind in ("burn", "neg")
    ]
    chosen = select_train_pairs(rows, pairs_per_continent=1)
    assert len(chosen) == 2
    assert {row["event_id"] for row in chosen} == {"a"}


def test_byte_threshold_recovers_known_fraction() -> None:
    arrays = [np.array([[10, 20], [30, 40]], dtype=np.uint8), np.array([[10, 20], [30, 40]], dtype=np.uint8)]
    result = best_byte_threshold(arrays, np.array([0.5, 0.5]))
    assert result["mean_absolute_fraction_error"] == 0.0
    assert result["pixel_rule"] == "byte > 20"
