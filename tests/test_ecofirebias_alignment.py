"""Check pixel-grid mapping and optional validity-mask manifest support."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from evaluation.matched_rate import read_benchmark_manifest
from scripts.audit_ecofirebias_alignment import map_post_pixels_to_dnbr


class FakePage:
    def __init__(self, x: float) -> None:
        self.shape = (4, 4)
        self.geotiff_tags = {
            "ProjectedCSTypeGeoKey": 32630,
            "ModelTiepoint": (0, 0, 0, x, 1000, 0),
            "ModelPixelScale": (10, 10, 0),
        }


def test_grid_mapping_identity() -> None:
    rows, cols, covered = map_post_pixels_to_dnbr(FakePage(500), FakePage(500))
    expected_rows, expected_cols = np.indices((4, 4))
    assert np.array_equal(rows, expected_rows)
    assert np.array_equal(cols, expected_cols)
    assert covered.all()


def test_grid_mapping_marks_uncovered_edge() -> None:
    rows, cols, covered = map_post_pixels_to_dnbr(FakePage(500), FakePage(510))
    assert not covered[:, 0].any()
    assert covered[:, 1:].all()
    assert np.array_equal(cols[:, 1:], np.broadcast_to(np.arange(3), (4, 3)))


def test_manifest_reads_optional_valid_mask(tmp_path: Path) -> None:
    path = tmp_path / "manifest.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("sample_id", "image_path", "mask_path", "valid_mask_path", "split"))
        writer.writeheader()
        writer.writerow({"sample_id": "a", "image_path": "a.png", "mask_path": "mask.png", "valid_mask_path": "valid.png", "split": "development"})
    items = read_benchmark_manifest(path, split="development")
    assert len(items) == 1
    assert items[0].valid_mask_path == Path("valid.png")
