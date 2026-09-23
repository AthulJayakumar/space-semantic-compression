"""Screen public burned-area metadata against known training locations only."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path

from pyproj import Transformer


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "results/future_satellite_cohort_audit"
SOURCE = AUDIT / "external_source_screen/satellite_burned_area_zenodo_6597139_metadata.csv"
EXPECTED_MD5 = "8a93ab8cd09a4c9b7a3096bb49379ba4"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def event_code(folder: str) -> str:
    match = re.match(r"^(EMSR\d+)_", folder)
    if not match:
        raise ValueError(f"Missing Copernicus event code: {folder}")
    return match.group(1)


def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    lat1, lat2 = math.radians(lat1), math.radians(lat2)
    dlat = lat2 - lat1
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371.0088 * 2 * math.asin(math.sqrt(a))


def screen() -> dict[str, object]:
    if hashlib.md5(SOURCE.read_bytes()).hexdigest() != EXPECTED_MD5:
        raise ValueError("Source metadata does not match Zenodo's published MD5")
    scenes = read_csv(SOURCE)
    if len(scenes) != 73 or len({row["folder"] for row in scenes}) != 73:
        raise ValueError("Expected 73 unique acquisitions")
    cems = read_csv(AUDIT / "cems_georeference.csv")
    source_codes = {row["event_group"] for row in cems}
    transformers: dict[str, Transformer] = {}
    cems_centers = []
    for row in cems:
        epsg = row["epsg"]
        transformers.setdefault(epsg, Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True))
        lon, lat = transformers[epsg].transform(float(row["projected_center_east_m"]), float(row["projected_center_north_m"]))
        cems_centers.append((lon, lat))
    plan = json.loads((AUDIT / "ecofirebias_training_expansion/TRAINING_PLAN_FROZEN.json").read_text(encoding="utf-8"))
    eco_metadata = {row["example_id"]: row for row in read_csv(ROOT / "datasets/wildfire_global_v1/metadata.csv")}
    eco_training_centers = [
        (float(eco_metadata[sample_id]["patch_lon"]), float(eco_metadata[sample_id]["patch_lat"]))
        for sample_id in plan["sample_ids"]
    ]
    rows = []
    for scene in scenes:
        code = event_code(scene["folder"])
        lon, lat = float(scene["longitude"]), float(scene["latitude"])
        bounds = tuple(float(scene[key]) for key in ("top_left_long", "top_left_lat", "bottom_right_long", "bottom_right_lat"))
        if not (-180 <= lon <= 180 and -90 <= lat <= 90 and bounds[0] <= lon <= bounds[2] and bounds[3] <= lat <= bounds[1]):
            raise ValueError(f"Invalid scene center/bounds: {scene['folder']}")
        nearest_cems = min(haversine_km(lon, lat, x, y) for x, y in cems_centers)
        nearest_eco = min(haversine_km(lon, lat, x, y) for x, y in eco_training_centers)
        rows.append({
            "folder": scene["folder"], "event_code": code, "fold": scene["fold"],
            "longitude": lon, "latitude": lat,
            "known_cems_code_overlap": code in source_codes,
            "nearest_cems_source_tile_center_km": round(nearest_cems, 2),
            "nearest_selected_eco_training_patch_center_km": round(nearest_eco, 2),
            "provisional_center_100km_screen": code not in source_codes and nearest_cems > 100 and nearest_eco > 100,
        })
    output = {
        "status": "metadata_only_not_eligible_for_scoring",
        "source_url": "https://zenodo.org/records/6597139",
        "source_metadata_md5": EXPECTED_MD5,
        "scene_count": len(rows), "event_codes": len({row["event_code"] for row in rows}),
        "exact_cems_event_code_overlap_scenes": sum(row["known_cems_code_overlap"] for row in rows),
        "within_100km_known_cems_source_centers": sum(row["nearest_cems_source_tile_center_km"] <= 100 for row in rows),
        "within_100km_selected_eco_training_centers": sum(row["nearest_selected_eco_training_patch_center_km"] <= 100 for row in rows),
        "provisional_100km_center_screen_scenes": sum(row["provisional_center_100km_screen"] for row in rows),
        "provisional_100km_center_screen_event_codes": len({row["event_code"] for row in rows if row["provisional_center_100km_screen"]}),
        "model_outputs_accessed": False,
        "limitations": [
            "Center-distance is not footprint intersection or proof of separation from all inherited training scenes.",
            "One ancestor training manifest is missing, so checkpoint lineage is incomplete.",
            "The Zenodo record has no clearly stated license; redistribution rights require verification.",
            "No raster or mask files were downloaded; label encoding/alignment and image quality are unverified.",
            "The source appears to use CEMS grading maps, so labels are not independent human burn-scar annotations.",
        ],
        "scenes": rows,
    }
    target = AUDIT / "external_source_screen/SOURCE_METADATA_SCREEN.json"
    target.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    return output


if __name__ == "__main__":
    result = screen()
    print(json.dumps({key: value for key, value in result.items() if key != "scenes"}, indent=2))
