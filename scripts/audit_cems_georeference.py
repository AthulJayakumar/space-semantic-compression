"""Recover event and projected-location metadata embedded in CEMS GeoTIFFs."""

from __future__ import annotations

import csv
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import tifffile


ROOT = Path(__file__).resolve().parents[1]


def read_cems_metadata(path: Path) -> dict[str, object]:
    with tifffile.TiffFile(path) as image:
        page = image.pages[0]
        xml_text = page.tags["GDAL_METADATA"].value
        tiepoint = page.tags["ModelTiepointTag"].value
        scale = page.tags["ModelPixelScaleTag"].value
        height, width = page.shape[:2]
    metadata = {
        node.attrib["name"]: node.text or ""
        for node in ET.fromstring(xml_text).findall("Item")
    }
    for required in ("code", "country", "crs", "tile_id"):
        if not metadata.get(required):
            raise ValueError(f"Missing {required} in {path}")
    return {
        "event_group": metadata["code"],
        "country": metadata["country"],
        "epsg": metadata["crs"],
        "source_tile_id": metadata["tile_id"],
        "projected_center_east_m": float(tiepoint[3] + (width / 2 - tiepoint[0]) * scale[0]),
        "projected_center_north_m": float(tiepoint[4] - (height / 2 - tiepoint[1]) * scale[1]),
        "pixel_size_x_m": float(scale[0]),
        "pixel_size_y_m": float(scale[1]),
        "width": width,
        "height": height,
    }


def audit(root: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    manifest = root / "datasets/research_wildfire/cems_hls/cems_hls_manifest.csv"
    with manifest.open(newline="", encoding="utf-8") as handle:
        source = list(csv.DictReader(handle))
    rows = []
    for entry in source:
        image_path = (root / entry["image_path"]).resolve()
        mask_path = (root / entry["mask_path"]).resolve()
        if not image_path.is_file() or not mask_path.is_file():
            raise FileNotFoundError(f"CEMS image/mask missing: {image_path}")
        rows.append({
            "sample_id": entry["source_id"],
            "split": entry["split"],
            **read_cems_metadata(image_path),
        })
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate CEMS sample IDs")
    event_sets = {
        split: {row["event_group"] for row in rows if row["split"] == split}
        for split in ("train", "validation", "test")
    }
    split_overlap = {
        f"{left}_{right}": sorted(event_sets[left] & event_sets[right])
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    }
    impactmesh_path = root / "results/external_lockbox_impactmesh_100/impactmesh_100_lockbox_manifest.csv"
    with impactmesh_path.open(newline="", encoding="utf-8") as handle:
        impactmesh = list(csv.DictReader(handle))
    impactmesh_codes = {row["event_group"].split("_", 1)[0] for row in impactmesh}
    cross_source_overlap = sorted(event_sets["train"] & impactmesh_codes)
    overlap_tiles = [row for row in impactmesh if row["event_group"].split("_", 1)[0] in cross_source_overlap]
    all_cems_codes = set().union(*event_sets.values())
    all_source_overlap = sorted(all_cems_codes & impactmesh_codes)
    all_source_tiles = [row for row in impactmesh if row["event_group"].split("_", 1)[0] in all_source_overlap]
    summary = {
        "tiles": len(rows),
        "events": len({row["event_group"] for row in rows}),
        "tiles_by_split": dict(Counter(str(row["split"]) for row in rows)),
        "events_by_split": {split: len(events) for split, events in event_sets.items()},
        "countries_by_split": {
            split: sorted({str(row["country"]) for row in rows if row["split"] == split})
            for split in event_sets
        },
        "event_overlap_between_splits": split_overlap,
        "event_separated_source_split": all(not events for events in split_overlap.values()),
        "impactmesh_evaluated_event_codes": len(impactmesh_codes),
        "impactmesh_codes_overlapping_cems_train": cross_source_overlap,
        "impactmesh_overlapping_tiles": len(overlap_tiles),
        "impactmesh_event_clean_tiles_previously_evaluated": len(impactmesh) - len(overlap_tiles),
        "impactmesh_codes_overlapping_full_cems_source": all_source_overlap,
        "impactmesh_tiles_overlapping_full_cems_source": len(all_source_tiles),
        "geolocation_status": "Projected GeoTIFF centers and EPSG codes recovered; geographic longitude/latitude conversion not required for event audit",
    }
    return rows, summary


def main() -> None:
    rows, summary = audit(ROOT)
    output = ROOT / "results/future_satellite_cohort_audit"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "cems_georeference.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "cems_event_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
