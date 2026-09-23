"""Check training-split RGB/dNBR grids before any model comparison."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from pyproj import Geod, Transformer
import tifffile


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/future_satellite_cohort_audit"
SOURCE = ROOT / "datasets/wildfire_global_v1"


def grid(page: tifffile.TiffPage) -> tuple[int, tuple[float, ...], tuple[float, ...]]:
    tags = page.geotiff_tags
    epsg = int(tags["ProjectedCSTypeGeoKey"])
    tiepoint = tuple(float(x) for x in tags["ModelTiepoint"])
    scale = tuple(float(x) for x in tags["ModelPixelScale"])
    return epsg, tiepoint, scale


def map_post_pixels_to_dnbr(post: tifffile.TiffPage, dnbr: tifffile.TiffPage) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dst_epsg, dst_tie, dst_scale = grid(post)
    src_epsg, src_tie, src_scale = grid(dnbr)
    rows, cols = np.indices(post.shape[:2], dtype=np.float64)
    east = dst_tie[3] + (cols + 0.5 - dst_tie[0]) * dst_scale[0]
    north = dst_tie[4] - (rows + 0.5 - dst_tie[1]) * dst_scale[1]
    if dst_epsg != src_epsg:
        east, north = Transformer.from_crs(
            f"EPSG:{dst_epsg}", f"EPSG:{src_epsg}", always_xy=True
        ).transform(east, north)
    src_cols = np.rint((east - src_tie[3]) / src_scale[0] + src_tie[0] - 0.5).astype(np.int32)
    src_rows = np.rint((src_tie[4] - north) / src_scale[1] + src_tie[1] - 0.5).astype(np.int32)
    inside = (
        (src_rows >= 0) & (src_rows < dnbr.shape[0]) &
        (src_cols >= 0) & (src_cols < dnbr.shape[1])
    )
    return src_rows, src_cols, inside


def audit() -> dict[str, object]:
    assets = json.loads((OUT / "ecofirebias_assets_development.json").read_text(encoding="utf-8"))["assets"]
    geod = Geod(ellps="WGS84")
    records = []
    for entry in assets:
        paths = {kind: SOURCE / value["path"] for kind, value in entry["assets"].items()}
        rgb = np.asarray(Image.open(paths["rgb"]).convert("RGB"))
        with tifffile.TiffFile(paths["post_geotiff"]) as post_file, tifffile.TiffFile(paths["dnbr"]) as dnbr_file:
            post, dnbr = post_file.pages[0], dnbr_file.pages[0]
            if not np.array_equal(rgb, post.asarray()):
                raise ValueError(f"PNG and GeoTIFF RGB pixels differ: {entry['example_id']}")
            dst_epsg, dst_tie, dst_scale = grid(post)
            src_epsg, src_tie, src_scale = grid(dnbr)
            src_rows, src_cols, inside = map_post_pixels_to_dnbr(post, dnbr)
            dst_x = dst_tie[3] + (post.shape[1] / 2 - dst_tie[0]) * dst_scale[0]
            dst_y = dst_tie[4] - (post.shape[0] / 2 - dst_tie[1]) * dst_scale[1]
            src_x = src_tie[3] + (dnbr.shape[1] / 2 - src_tie[0]) * src_scale[0]
            src_y = src_tie[4] - (dnbr.shape[0] / 2 - src_tie[1]) * src_scale[1]
            dst_lon, dst_lat = Transformer.from_crs(f"EPSG:{dst_epsg}", "EPSG:4326", always_xy=True).transform(dst_x, dst_y)
            src_lon, src_lat = Transformer.from_crs(f"EPSG:{src_epsg}", "EPSG:4326", always_xy=True).transform(src_x, src_y)
            center_distance_m = abs(geod.inv(dst_lon, dst_lat, src_lon, src_lat)[2])
            records.append({
                "example_id": entry["example_id"],
                "kind": entry["kind"],
                "post_epsg": dst_epsg,
                "dnbr_epsg": src_epsg,
                "exact_grid": (dst_epsg, dst_tie, dst_scale) == (src_epsg, src_tie, src_scale),
                "center_distance_m": round(center_distance_m, 2),
                "valid_reprojection_fraction": float(inside.mean()),
                "rgb_png_equals_post_geotiff": True,
            })
    result = {
        "status": "training_development_alignment_audit_only",
        "model_outputs_or_reserved_images_accessed": False,
        "images": len(records),
        "exact_grid_count": sum(row["exact_grid"] for row in records),
        "cross_utm_zone_count": sum(row["post_epsg"] != row["dnbr_epsg"] for row in records),
        "minimum_reprojection_coverage": min(row["valid_reprojection_fraction"] for row in records),
        "maximum_center_distance_m": max(row["center_distance_m"] for row in records),
        "records": records,
    }
    (OUT / "ecofirebias_development_alignment.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return {key: value for key, value in result.items() if key != "records"}


def main() -> None:
    print(json.dumps(audit(), indent=2))


if __name__ == "__main__":
    main()
