"""Check the frozen EcoFireBias cohort without inspecting images or model outputs."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from pyproj import Geod, Transformer


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/future_satellite_cohort_audit"
GEOGRAPHIC_BUFFER_KM = 100.0


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_locations(georef: list[dict[str, str]]) -> list[tuple[float, float]]:
    transformers: dict[str, Transformer] = {}
    locations = []
    for row in georef:
        epsg = row["epsg"].removeprefix("EPSG:")
        if epsg not in transformers:
            transformers[epsg] = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
        lon, lat = transformers[epsg].transform(
            float(row["projected_center_east_m"]), float(row["projected_center_north_m"])
        )
        if not (-180 <= lon <= 180 and -90 <= lat <= 90):
            raise ValueError(f"Invalid transformed CEMS location: {row['sample_id']}")
        locations.append((lon, lat))
    return locations


def audit(root: Path) -> dict[str, object]:
    metadata_path = root / "datasets/wildfire_global_v1/metadata.csv"
    frozen_path = root / "results/external_lockbox_ecofirebias_replication_120/SELECTION_FROZEN.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if sha256(metadata_path) != frozen["metadata_sha256"]:
        raise ValueError("Frozen selection metadata hash does not match")
    metadata = {row["example_id"]: row for row in read_csv(metadata_path)}
    selected = [metadata[example_id] for example_id in frozen["selected_ids"]]
    if len({row["example_id"] for row in selected}) != len(selected):
        raise ValueError("Frozen selection contains duplicate example IDs")
    by_event: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in selected:
        by_event[row["event_id"]].append(row)
    if any(len(pair) != 2 or {item["kind"] for item in pair} != {"burn", "neg"} for pair in by_event.values()):
        raise ValueError("Frozen selection is not a set of burn/negative event pairs")

    georef = read_csv(root / "results/future_satellite_cohort_audit/cems_georeference.csv")
    cems_countries = {part.strip() for row in georef for part in row["country"].split(" - ")}
    # HLS training covers the US. Exclude it even if CEMS names no US scene.
    cems_countries.update({"United States of America", "United States", "USA"})
    locations = source_locations(georef)
    geod = Geod(ellps="WGS84")
    records = []
    for event_id, pair in by_event.items():
        country = pair[0]["country"]
        if {item["country"] for item in pair} != {country}:
            raise ValueError(f"Inconsistent paired countries: {event_id}")
        lon, lat = float(pair[0]["event_lon"]), float(pair[0]["event_lat"])
        nearest_km = min(abs(geod.inv(lon, lat, x, y)[2]) / 1000 for x, y in locations)
        reasons = []
        if country in cems_countries:
            reasons.append("country_present_in_cems_or_us_hls_training")
        if nearest_km < GEOGRAPHIC_BUFFER_KM:
            reasons.append("within_100km_of_cems_source_tile_center")
        records.append({
            "event_id": event_id,
            "example_ids": [item["example_id"] for item in pair],
            "continent": pair[0]["continent"],
            "country": country,
            "nearest_cems_source_center_km": round(nearest_km, 2),
            "eligible": not reasons,
            "exclusion_reasons": reasons,
            "post_rgb_present": [bool((root / "datasets/wildfire_global_v1" / item["post_image_path"]).is_file()) for item in pair],
            "raw_dnbr_present": [bool((root / "datasets/wildfire_global_v1" / item["dnbr_geotiff_path"]).is_file()) for item in pair],
        })
    eligible = [row for row in records if row["eligible"]]
    return {
        "status": "pre_evaluation_geographic_and_asset_audit",
        "image_pixels_or_model_outputs_accessed": False,
        "frozen_selection_sha256": sha256(frozen_path),
        "metadata_sha256": frozen["metadata_sha256"],
        "source_georeference_sha256": sha256(root / "results/future_satellite_cohort_audit/cems_georeference.csv"),
        "geographic_buffer_km": GEOGRAPHIC_BUFFER_KM,
        "source_locations_checked": len(locations),
        "frozen_events": len(records),
        "eligible_events": len(eligible),
        "eligible_samples": 2 * len(eligible),
        "eligible_continents": dict(Counter(row["continent"] for row in eligible)),
        "local_post_rgb_files": sum(sum(row["post_rgb_present"]) for row in records),
        "local_raw_dnbr_files": sum(sum(row["raw_dnbr_present"]) for row in records),
        "events": records,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    result = audit(ROOT)
    destination = OUT / "ecofirebias_replication_audit.json"
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "events"}, indent=2))


if __name__ == "__main__":
    main()
