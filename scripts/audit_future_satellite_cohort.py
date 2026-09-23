"""Audit unused local satellite cohorts without opening a reserved test set."""

from __future__ import annotations

import csv
import json
import re
import sys
import tarfile
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_cems_georeference import audit as audit_cems  # noqa: E402

HLS_GROUP = re.compile(r"HLS\.S30\.T([0-9A-Z]{5})\.")


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def impactmesh_event(sample_id: str) -> str:
    parts = sample_id.split("_")
    if len(parts) < 2:
        raise ValueError(f"Unparseable ImpactMesh ID: {sample_id}")
    return "_".join(parts[:2])


def audit(root: Path) -> dict[str, object]:
    hls = rows(root / "results/validated_hls_500/validated_scene_manifest.csv")
    selector = rows(root / "results/token_sus_selector/internal_geographic_split.csv")
    mixed = rows(root / "results/mixed_satellite_training/training_manifest.csv")
    cems = rows(root / "datasets/research_wildfire/cems_hls/cems_hls_manifest.csv")
    cems_geo, cems_summary = audit_cems(root)
    used_hls_ids = {row["sample_id"] for row in mixed if row["dataset"] == "ibm_nasa_hls_burn_scars"}
    held_groups = {row["geographic_group"] for row in hls if row["split"] != "train"}
    held_groups |= {row["geographic_group"] for row in selector if row["split"] == "selector_validation"}
    archive_path = root / "datasets/research_wildfire/hls_burn_scars/hls_burn_scars.tar.gz"
    archive_images: dict[str, str] = {}
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile() or not member.name.endswith("_merged.tif"):
                continue
            sample_id = Path(member.name).stem
            match = HLS_GROUP.search(sample_id)
            if match is None or sample_id in archive_images:
                raise ValueError(f"Unparseable or duplicate HLS image: {member.name}")
            archive_images[sample_id] = match.group(1)
    unused_hls = {sample_id: group for sample_id, group in archive_images.items() if sample_id not in used_hls_ids}
    unused_hls_fresh_groups = {
        group for group in unused_hls.values()
        if group not in held_groups and group not in {archive_images[sample_id] for sample_id in used_hls_ids if sample_id in archive_images}
    }

    holdout_path = root / "datasets/impactmesh_fire_v1/split/impactmesh_fire_test_holdout.txt"
    impactmesh_ids = [line for line in holdout_path.read_text(encoding="utf-8").splitlines() if line]
    impactmesh_seen = rows(root / "results/external_lockbox_impactmesh_100/impactmesh_100_lockbox_manifest.csv")
    impactmesh_events = {impactmesh_event(sample_id) for sample_id in impactmesh_ids}
    impactmesh_seen_events = {row["event_group"] for row in impactmesh_seen}

    eco = rows(root / "datasets/wildfire_global_v1/metadata.csv")
    eco_test = [row for row in eco if row["split"] == "test"]
    eco_by_id = {row["example_id"]: row for row in eco_test}
    eco_old = rows(root / "results/external_lockbox_ecofirebias_120/ecofirebias_120_lockbox_manifest.csv")
    eco_old_events = {row["event_group"] for row in eco_old}
    eco_reserved = json.loads((root / "results/external_lockbox_ecofirebias_replication_120/SELECTION_FROZEN.json").read_text(encoding="utf-8"))
    reserved_ids = set(eco_reserved["selected_ids"])
    if reserved_ids - eco_by_id.keys():
        raise ValueError("Reserved EcoFireBias IDs are missing from local metadata")
    reserved_rows = [eco_by_id[sample_id] for sample_id in reserved_ids]
    reserved_events = {row["event_id"] for row in reserved_rows}
    if eco_old_events & reserved_events:
        raise ValueError("Evaluated and reserved EcoFireBias events overlap")
    eco_test_events = {row["event_id"] for row in eco_test}
    reserved_non_north_america = [row for row in reserved_rows if row["continent"] != "North America"]
    local_dnbr = {path.stem for path in (root / "datasets/wildfire_global_v1/images/dnbr").rglob("*.png")}
    local_post = {path.stem for path in (root / "datasets/wildfire_global_v1/images/post").rglob("*.png")}

    return {
        "hls": {
            "archive_images": len(archive_images),
            "used_training_images_in_archive": len(archive_images.keys() & used_hls_ids),
            "unused_archive_images": len(unused_hls),
            "unused_geographic_groups_outside_all_known_hls_training_and_holdouts": len(unused_hls_fresh_groups),
            "event_identifier_available": False,
            "archive_validation_tiles_reassigned_to_mixed_training": sum(
                row["dataset"] == "ibm_nasa_hls_burn_scars" and row["source_split"] == "validation"
                for row in mixed
            ),
        },
        "cems": {
            "source_splits": dict(Counter(row["split"] for row in cems)),
            "training_images_used_by_mixed_model": sum(row["dataset"] == "cems_hls" for row in mixed),
            "embedded_event_codes": cems_summary["events"],
            "train_test_event_code_overlap": len(cems_summary["event_overlap_between_splits"]["train_test"]),
            "projected_georeference_available": len(cems_geo) == len(cems),
            "checkpoint_lineage_caveat": "Ancestor checkpoints record random fitting on the full CEMS manifest; source-train overlap is a lower bound on model exposure",
        },
        "impactmesh": {
            "local_test_tiles": len(impactmesh_ids),
            "local_test_events": len(impactmesh_events),
            "previously_evaluated_events": len(impactmesh_seen_events),
            "unseen_events_in_local_test_partition": len(impactmesh_events - impactmesh_seen_events),
            "evaluated_tiles_sharing_cems_training_event_code": cems_summary["impactmesh_overlapping_tiles"],
            "shared_cems_training_event_codes": len(cems_summary["impactmesh_codes_overlapping_cems_train"]),
            "evaluated_tiles_sharing_any_cems_source_event_code": cems_summary["impactmesh_tiles_overlapping_full_cems_source"],
            "shared_full_cems_source_event_codes": len(cems_summary["impactmesh_codes_overlapping_full_cems_source"]),
        },
        "ecofirebias": {
            "test_chips": len(eco_test),
            "test_events": len(eco_test_events),
            "previously_evaluated_events": len(eco_old_events),
            "reserved_unevaluated_events": len(reserved_events),
            "remaining_unselected_test_events": len(eco_test_events - eco_old_events - reserved_events),
            "reserved_non_north_america_chips": len(reserved_non_north_america),
            "reserved_non_north_america_events": len({row["event_id"] for row in reserved_non_north_america}),
            "reserved_local_post_images": len(reserved_ids & local_post),
            "reserved_local_dnbr_images": len(reserved_ids & local_dnbr),
            "independent_label_candidate": "Sentinel-2 dNBR map, not yet locally acquired or verified",
        },
        "conclusion": (
            "No locally complete unused event-separated satellite image/mask cohort is ready for a joint SUS-Dice test. "
            "Embedded CEMS GeoTIFF metadata reveal event-code overlap across its source splits and with the previously evaluated ImpactMesh cohort. "
            "The existing frozen EcoFireBias replication selection contains unused events, but its post-fire RGB and dNBR label files must be acquired and validated; cross-source country/location screening is also needed."
        ),
    }


def main() -> None:
    output = ROOT / "results/future_satellite_cohort_audit"
    result = audit(ROOT)
    output.mkdir(parents=True, exist_ok=True)
    (output / "audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
