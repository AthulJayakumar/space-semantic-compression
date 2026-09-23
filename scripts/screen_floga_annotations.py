"""Audit FLOGA v1 polygon metadata without reading imagery or model outputs.

Requires pyshp (the ``shapefile`` module). Annotation files are from the
pinned public FLOGA-annotations repository revision in REVISION below.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from pathlib import Path

from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.screen_satellite_burned_area_source import haversine_km  # noqa: E402

AUDIT = ROOT / "results/future_satellite_cohort_audit"
ANNOTATIONS = ROOT / "datasets/floga_metadata/v1"
REVISION = "f695e39964bae5292e907bc86d61c019c0e271c1"
BUFFER_KM = 100.0


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sentinel_acquisition_key(value: str) -> str | None:
    match = re.search(r"(S2[AB])_MSIL2A_(\d{8}T\d{6})_.*?_(T\d{2}[A-Z]{3})(?:_|$)", value)
    return "|".join(match.groups()) if match else None


def bbox_center_radius(bbox: tuple[float, float, float, float]) -> tuple[float, float, float]:
    west, south, east, north = bbox
    if not (-180 <= west <= east <= 180 and -90 <= south <= north <= 90):
        raise ValueError(f"Invalid geographic bounding box: {bbox}")
    lon, lat = (west + east) / 2, (south + north) / 2
    radius = max(haversine_km(lon, lat, x, y) for x in (west, east) for y in (south, north))
    return lon, lat, radius


def separation_lower_bound_km(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return max(0.0, haversine_km(a[0], a[1], b[0], b[1]) - a[2] - b[2])


def known_eco_samples() -> set[str]:
    selected = set(json.loads((AUDIT / "ecofirebias_training_expansion/TRAINING_PLAN_FROZEN.json").read_text(encoding="utf-8"))["sample_ids"])
    selected.update(json.loads((AUDIT / "DEVELOPMENT_GATE_FROZEN.json").read_text(encoding="utf-8"))["development_ids"])
    selected.update(json.loads((AUDIT / "ecofirebias_official_validation/VALIDATION_PLAN_FROZEN.json").read_text(encoding="utf-8"))["sample_ids"])
    selected.update(row["example_id"] for row in json.loads((AUDIT / "dnbr_encoding_train_only.json").read_text(encoding="utf-8"))["samples"])
    selected.update(row["sample_id"] for row in read_csv(ROOT / "results/external_lockbox_ecofirebias_120/ecofirebias_120_lockbox_manifest.csv"))
    return selected


def audit() -> dict[str, object]:
    import shapefile

    all_eco = {row["example_id"]: row for row in read_csv(ROOT / "datasets/wildfire_global_v1/metadata.csv")}
    selected_ids = known_eco_samples()
    if selected_ids - all_eco.keys():
        raise ValueError("Known EcoFireBias sample IDs are absent from source metadata")
    eco_centers = [
        (float(all_eco[sample_id]["patch_lon"]), float(all_eco[sample_id]["patch_lat"]), 2.0)
        for sample_id in selected_ids
    ]
    eco_products = {
        key for sample_id in selected_ids
        for field in ("pre_item_id", "post_item_id")
        if (key := sentinel_acquisition_key(all_eco[sample_id][field]))
    }
    cems_centers = []
    transformers: dict[str, Transformer] = {}
    for row in read_csv(AUDIT / "cems_georeference.csv"):
        epsg = row["epsg"]
        transformers.setdefault(epsg, Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True))
        transform = transformers[epsg]
        east, north = float(row["projected_center_east_m"]), float(row["projected_center_north_m"])
        center_lon, center_lat = transform.transform(east, north)
        half_width = abs(float(row["pixel_size_x_m"])) * int(row["width"]) / 2
        half_height = abs(float(row["pixel_size_y_m"])) * int(row["height"]) / 2
        radius = max(
            haversine_km(center_lon, center_lat, *transform.transform(x, y))
            for x in (east - half_width, east + half_width)
            for y in (north - half_height, north + half_height)
        )
        cems_centers.append((center_lon, center_lat, radius))
    events: dict[str, dict[str, object]] = {}
    file_hashes = {}
    polygon_count = 0
    for year in range(2017, 2022):
        stem = f"fb_{year}_final_images_4326"
        files = [ANNOTATIONS / f"{stem}.{extension}" for extension in ("shp", "shx", "dbf", "prj")]
        for path in files:
            file_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        reader = shapefile.Reader(str(ANNOTATIONS / f"{stem}.shp"))
        for record, shape in zip(reader.iterRecords(), reader.iterShapes(), strict=True):
            polygon_count += 1
            event_id = f"{year}:{record['ID']}"
            bounds = shape.bbox
            if event_id not in events:
                events[event_id] = {
                    "year": year, "source_event_id": record["ID"], "polygons": 0,
                    "bbox": list(bounds), "acquisitions": set(),
                }
            event = events[event_id]
            event["polygons"] += 1
            event["bbox"] = [min(event["bbox"][0], bounds[0]), min(event["bbox"][1], bounds[1]),
                             max(event["bbox"][2], bounds[2]), max(event["bbox"][3], bounds[3])]
            for field in ("SEN_2A_s", "SEN_2A_e"):
                if key := sentinel_acquisition_key(record[field]):
                    event["acquisitions"].add(key)
    screened = []
    for event_id, event in sorted(events.items()):
        center = bbox_center_radius(tuple(event["bbox"]))
        nearest_cems = min(separation_lower_bound_km(center, old) for old in cems_centers)
        nearest_eco = min(separation_lower_bound_km(center, old) for old in eco_centers)
        direct_acquisition_overlap = sorted(event["acquisitions"] & eco_products)
        screened.append({
            "event_id": event_id, "year": event["year"], "polygons": event["polygons"],
            "bbox_wgs84": event["bbox"], "event_bbox_radius_km": round(center[2], 2),
            "nearest_known_cems_footprint_lower_bound_km": round(nearest_cems, 2),
            "nearest_consulted_eco_chip_lower_bound_km": round(nearest_eco, 2),
            "exact_consulted_eco_acquisition_overlap": direct_acquisition_overlap,
            "provisional_100km_geometry_screen": nearest_cems > BUFFER_KM and nearest_eco > BUFFER_KM and not direct_acquisition_overlap,
        })
    output = {
        "status": "annotation_metadata_only_not_eligible_for_scoring",
        "source_repository": "https://github.com/Orion-AI-Lab/FLOGA-annotations",
        "annotation_version": "v1",
        "revision": REVISION,
        "annotation_sha256": file_hashes,
        "polygons": polygon_count,
        "year_event_ids": len(screened),
        "known_consulted_eco_samples": len(selected_ids),
        "within_100km_known_cems_footprint_lower_bound": sum(row["nearest_known_cems_footprint_lower_bound_km"] <= BUFFER_KM for row in screened),
        "within_100km_consulted_eco_chip_lower_bound": sum(row["nearest_consulted_eco_chip_lower_bound_km"] <= BUFFER_KM for row in screened),
        "exact_consulted_eco_acquisition_overlap_events": sum(bool(row["exact_consulted_eco_acquisition_overlap"]) for row in screened),
        "provisional_100km_geometry_screen_events": sum(row["provisional_100km_geometry_screen"] for row in screened),
        "model_outputs_accessed": False,
        "imagery_or_pixel_masks_accessed": False,
        "limitations": [
            "This is v1 annotation geography, not proof of v2 GeoTIFF image availability or equivalence.",
            "Bounding-box lower bounds are conservative but do not cover unknown ancestor training imagery.",
            "Only known CEMS footprints and selected/consulted EcoFireBias chips were screened; other source lineages remain incomplete.",
            "Image channel order, radiometric conversion, pixelwise label alignment, and selective archive access remain unverified.",
        ],
        "events": screened,
    }
    target = AUDIT / "external_source_screen/FLOGA_V1_ANNOTATION_SCREEN.json"
    target.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    return output


if __name__ == "__main__":
    result = audit()
    print(json.dumps({key: value for key, value in result.items() if key not in ("events", "annotation_sha256")}, indent=2))
