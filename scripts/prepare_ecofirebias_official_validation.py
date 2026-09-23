"""Align and validate the one-shot EcoFireBias official-validation cohort."""

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

AUDIT = ROOT / "results/future_satellite_cohort_audit"
OUT = AUDIT / "ecofirebias_official_validation"
SOURCE = ROOT / "datasets/wildfire_global_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare() -> dict[str, object]:
    plan_path = OUT / "VALIDATION_PLAN_FROZEN.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    inventory_path = AUDIT / "ecofirebias_assets_official_validation.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    if inventory["mode"] != "official_validation" or inventory["dataset_revision"] != plan["dataset_revision"]:
        raise ValueError("Downloaded assets differ from frozen validation plan")
    if sha256(SOURCE / "metadata.csv") != plan["metadata_sha256"]:
        raise ValueError("Metadata changed after validation selection was frozen")
    with (SOURCE / "metadata.csv").open(newline="", encoding="utf-8") as handle:
        metadata = {row["example_id"]: row for row in csv.DictReader(handle)}
    assets = {row["example_id"]: row["assets"] for row in inventory["assets"]}
    if set(assets) != set(plan["sample_ids"]):
        raise ValueError("Downloaded validation IDs differ from frozen selection")

    mask_dir = OUT / "proxy_masks"
    valid_dir = OUT / "valid_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    valid_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    checks = []
    for sample_id in plan["sample_ids"]:
        row = metadata[sample_id]
        if row["split"] != "val":
            raise ValueError(f"Non-validation sample selected: {sample_id}")
        paths = {kind: SOURCE / record["path"] for kind, record in assets[sample_id].items()}
        if any(sha256(paths[kind]) != record["sha256"] for kind, record in assets[sample_id].items()):
            raise ValueError(f"Asset hash changed: {sample_id}")
        rgb = np.asarray(Image.open(paths["rgb"]).convert("RGB"))
        with tifffile.TiffFile(paths["post_geotiff"]) as post_file, tifffile.TiffFile(paths["dnbr"]) as dnbr_file:
            post, dnbr = post_file.pages[0], dnbr_file.pages[0]
            post_rgb, dnbr_bytes = post.asarray(), dnbr.asarray()
            source_rows, source_cols, covered = map_post_pixels_to_dnbr(post, dnbr)
            exact = grid(post) == grid(dnbr)
        if rgb.shape != (224, 224, 3) or dnbr_bytes.shape != (224, 224) or dnbr_bytes.dtype != np.uint8:
            raise ValueError(f"Unexpected RGB/dNBR layout: {sample_id}")
        if not np.array_equal(rgb, post_rgb):
            raise ValueError(f"RGB PNG and post-fire GeoTIFF pixels differ: {sample_id}")
        reported = float(row["burn_pixel_fraction_027"])
        source_proxy_fraction = float(np.mean(dnbr_bytes > 85))
        if abs(source_proxy_fraction - reported) > 0.01:
            raise ValueError(f"Quantized dNBR fraction differs from pinned metadata: {sample_id}")
        if float(covered.mean()) < 0.95:
            raise ValueError(f"Insufficient reprojected dNBR coverage: {sample_id}")
        aligned = np.zeros((224, 224), dtype=np.uint8)
        aligned[covered] = dnbr_bytes[source_rows[covered], source_cols[covered]]
        mask_path, valid_path = mask_dir / f"{sample_id}.png", valid_dir / f"{sample_id}.png"
        for destination, pixels in ((mask_path, (aligned > 85).astype(np.uint8) * 255), (valid_path, covered.astype(np.uint8) * 255)):
            if destination.exists():
                if not np.array_equal(np.asarray(Image.open(destination).convert("L")), pixels):
                    raise ValueError(f"Existing aligned mask changed: {sample_id}")
            else:
                Image.fromarray(pixels).save(destination)
        manifest.append({
            "sample_id": sample_id,
            "dataset": "ecofirebias_official_validation",
            "image_path": str(paths["rgb"]),
            "mask_path": str(mask_path),
            "valid_mask_path": str(valid_path),
            "split": "official_validation",
            "source_split": "val",
            "geographic_group": row["event_id"],
            "event_group": row["event_id"],
            "label_source": "quantized_dnbr_gt_027_proxy",
            "image_unit": "independent_224x224_sentinel2_post_event_rgb_chip",
            "cloud_fraction": row["post_cloud_cover"],
            "smoke_label_status": "unavailable",
        })
        checks.append({
            "sample_id": sample_id,
            "event_id": row["event_id"],
            "kind": row["kind"],
            "exact_grid": exact,
            "reprojection_coverage": float(covered.mean()),
            "metadata_dnbr_valid_fraction": float(row["dnbr_valid_fraction"]),
            "source_proxy_fraction_error": abs(source_proxy_fraction - reported),
            "proxy_mask_sha256": sha256(mask_path),
            "valid_mask_sha256": sha256(valid_path),
        })
    manifest_path = OUT / "validation_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    summary = {
        "status": "prepared_official_validation_before_model_scoring",
        "plan_sha256": sha256(plan_path),
        "inventory_sha256": sha256(inventory_path),
        "manifest_sha256": sha256(manifest_path),
        "images": len(manifest),
        "events": len({row["event_group"] for row in manifest}),
        "kind_counts": dict(Counter(row["kind"] for row in checks)),
        "cross_grid_images": sum(not row["exact_grid"] for row in checks),
        "minimum_reprojection_coverage": min(row["reprojection_coverage"] for row in checks),
        "minimum_metadata_dnbr_valid_fraction": min(row["metadata_dnbr_valid_fraction"] for row in checks),
        "maximum_source_proxy_fraction_error": max(row["source_proxy_fraction_error"] for row in checks),
        "model_outputs_accessed": False,
        "checks": checks,
    }
    (OUT / "PREPARATION.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return {key: value for key, value in summary.items() if key != "checks"}


def main() -> None:
    print(json.dumps(prepare(), indent=2))


if __name__ == "__main__":
    main()
