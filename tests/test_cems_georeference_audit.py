"""Tests for reading event provenance from CEMS GeoTIFF metadata."""

from __future__ import annotations

import numpy as np
import tifffile

from scripts.audit_cems_georeference import read_cems_metadata


def test_read_cems_metadata_recovers_event_and_projected_center(tmp_path):
    path = tmp_path / "tile.tif"
    xml = (
        "<GDALMetadata><Item name='code'>EMSR435</Item>"
        "<Item name='country'>Ukraine</Item><Item name='crs'>32636</Item>"
        "<Item name='tile_id'>33074</Item></GDALMetadata>"
    )
    tifffile.imwrite(
        path,
        np.zeros((4, 6), dtype="uint16"),
        extratags=[
            (42112, "s", len(xml) + 1, xml, False),
            (33550, "d", 3, (30.0, 30.0, 0.0), False),
            (33922, "d", 6, (0.0, 0.0, 0.0, 300000.0, 5700000.0, 0.0), False),
        ],
    )
    result = read_cems_metadata(path)
    assert result["event_group"] == "EMSR435"
    assert result["country"] == "Ukraine"
    assert result["epsg"] == "32636"
    assert result["projected_center_east_m"] == 300090.0
    assert result["projected_center_north_m"] == 5699940.0
