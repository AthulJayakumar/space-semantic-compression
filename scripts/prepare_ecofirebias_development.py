"""Validate and stage only the predeclared EcoFireBias training development set."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
import tifffile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_ecofirebias_alignment import grid, map_post_pixels_to_dnbr  # noqa: E402

OUT = ROOT / "results/future_satellite_cohort_audit"
SOURCE = ROOT / "datasets/wildfire_global_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stage() -> dict[str, object]:
    gate_path = OUT / "DEVELOPMENT_GATE_FROZEN.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    inventory_path = OUT / "ecofirebias_assets_development.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    if inventory["mode"] != "development" or inventory["dataset_revision"] != gate["dataset_revision"]:
        raise ValueError("Development assets are not from the frozen dataset revision")
    with (SOURCE / "metadata.csv").open(newline="", encoding="utf-8") as handle:
        metadata = {row["example_id"]: row for row in csv.DictReader(handle)}
    entries = {row["example_id"]: row for row in inventory["assets"]}
    if set(entries) != set(gate["development_ids"]):
        raise ValueError("Asset inventory differs from frozen development IDs")

    mask_dir = OUT / "development_masks"
    valid_dir = OUT / "development_valid_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    valid_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    checks = []
    for sample_id in gate["development_ids"]:
        row = metadata[sample_id]
        if row["split"] != "train":
            raise ValueError(f"Non-training sample in development set: {sample_id}")
        assets = entries[sample_id]["assets"]
        paths = {kind: SOURCE / details["path"] for kind, details in assets.items()}
        if any(sha256(paths[kind]) != details["sha256"] for kind, details in assets.items()):
            raise ValueError(f"Downloaded asset hash changed: {sample_id}")
        rgb = np.asarray(Image.open(paths["rgb"]).convert("RGB"))
        with tifffile.TiffFile(paths["post_geotiff"]) as post, tifffile.TiffFile(paths["dnbr"]) as dnbr:
            post_page, dnbr_page = post.pages[0], dnbr.pages[0]
            post_rgb = post_page.asarray()
            label_bytes = dnbr_page.asarray()
            src_rows, src_cols, covered = map_post_pixels_to_dnbr(post_page, dnbr_page)
            exact_grid = grid(post_page) == grid(dnbr_page)
        if rgb.shape != (224, 224, 3) or label_bytes.shape != (224, 224) or label_bytes.dtype != np.uint8:
            raise ValueError(f"Unexpected image or dNBR layout: {sample_id}")
        if not np.array_equal(rgb, post_rgb):
            raise ValueError(f"PNG and georeferenced RGB pixels differ: {sample_id}")
        aligned = np.zeros(label_bytes.shape, dtype=np.uint8)
        aligned[covered] = label_bytes[src_rows[covered], src_cols[covered]]
        mask = aligned > 85
        reported_fraction = float(row["burn_pixel_fraction_027"])
        measured_fraction = float(np.mean(label_bytes > 85))
        if abs(measured_fraction - reported_fraction) > 0.01:
            raise ValueError(f"Quantized dNBR disagrees with training metadata: {sample_id}")
        mask_path = mask_dir / f"{sample_id}.png"
        valid_path = valid_dir / f"{sample_id}.png"
        rendered = (mask.astype(np.uint8) * 255)
        valid_rendered = covered.astype(np.uint8) * 255
        for destination, pixels in ((mask_path, rendered), (valid_path, valid_rendered)):
            if destination.exists():
                if not np.array_equal(np.asarray(Image.open(destination).convert("L")), pixels):
                    raise ValueError(f"Existing development mask differs: {sample_id}")
            else:
                Image.fromarray(pixels).save(destination)
        checks.append({
            "sample_id": sample_id,
            "event_id": row["event_id"],
            "kind": row["kind"],
            "continent": row["continent"],
            "dnbr_valid_fraction_metadata": float(row["dnbr_valid_fraction"]),
            "reported_positive_fraction_027": reported_fraction,
            "quantized_proxy_positive_fraction": measured_fraction,
            "mask_sha256": sha256(mask_path),
            "valid_mask_sha256": sha256(valid_path),
            "rgb_equals_post_geotiff": True,
            "post_dnbr_exact_grid": exact_grid,
            "reprojection_coverage": float(covered.mean()),
        })
        manifest.append({
            "sample_id": sample_id,
            "dataset": "ecofirebias_train_development",
            "image_path": str(paths["rgb"]),
            "mask_path": str(mask_path),
            "valid_mask_path": str(valid_path),
            "split": "development",
            "source_split": "train",
            "geographic_group": row["event_id"],
            "event_group": row["event_id"],
            "label_source": "quantized_dnbr_gt_027_proxy",
            "image_unit": "independent_224x224_sentinel2_post_event_rgb_chip",
            "cloud_fraction": row["post_cloud_cover"],
            "smoke_label_status": "unavailable",
        })
    manifest_path = OUT / "ecofirebias_development_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    report = {
        "status": "staged_train_split_development_only",
        "gate_sha256": sha256(gate_path),
        "asset_inventory_sha256": sha256(inventory_path),
        "manifest_sha256": sha256(manifest_path),
        "images": len(manifest),
        "events": len({row["event_group"] for row in manifest}),
        "kind_counts": dict(Counter(row["kind"] for row in checks)),
        "minimum_metadata_dnbr_valid_fraction": min(row["dnbr_valid_fraction_metadata"] for row in checks),
        "minimum_reprojection_coverage": min(row["reprojection_coverage"] for row in checks),
        "reprojected_images": sum(not row["post_dnbr_exact_grid"] for row in checks),
        "maximum_proxy_fraction_error": max(abs(row["reported_positive_fraction_027"] - row["quantized_proxy_positive_fraction"]) for row in checks),
        "cannot_identify_all_invalid_pixels_from_quantized_bytes": True,
        "model_outputs_accessed": False,
        "checks": checks,
    }
    report_path = OUT / "ecofirebias_development_preparation.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return {key: value for key, value in report.items() if key != "checks"}


def main() -> None:
    print(json.dumps(stage(), indent=2))


if __name__ == "__main__":
    main()
