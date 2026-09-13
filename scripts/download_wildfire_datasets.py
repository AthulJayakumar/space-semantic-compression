"""scripts.download_wildfire_datasets

Plain-English purpose: Command-line runners for datasets, benchmarks, reports, and exports.

This file is part of the public Space Semantic Compression research demo.
It is documented at module level so researchers, supervisors, and non-specialist
readers can understand where the file fits before reading implementation details.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def save_hf_image(value: Any, path: Path) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if isinstance(value, Image.Image):
            value.convert("RGB").save(path, quality=95)
            return True
        if isinstance(value, dict):
            if isinstance(value.get("path"), str) and Path(value["path"]).exists():
                Image.open(value["path"]).convert("RGB").save(path, quality=95)
                return True
            if isinstance(value.get("bytes"), bytes):
                import io

                Image.open(io.BytesIO(value["bytes"])).convert("RGB").save(path, quality=95)
                return True
        if isinstance(value, str) and Path(value).exists():
            Image.open(value).convert("RGB").save(path, quality=95)
            return True
    except Exception:
        return False
    return False


def write_yolo_labels(annotations: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    if isinstance(annotations, list):
        for ann in annotations:
            if not isinstance(ann, dict):
                continue
            class_id = int(ann.get("class_id", 0))
            x = float(ann.get("x_center", ann.get("x", 0.5)))
            y = float(ann.get("y_center", ann.get("y", 0.5)))
            w = float(ann.get("width", ann.get("w", 0.1)))
            h = float(ann.get("height", ann.get("h", 0.1)))
            lines.append(f"{class_id} {x:.8f} {y:.8f} {w:.8f} {h:.8f}")
    path.write_text("\n".join(lines), encoding="utf-8")


def first_present(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        return value
    return None


def download_flameye_source(source: str, output_dir: Path, max_images: int | None) -> dict[str, Any]:
    import pandas as pd
    from huggingface_hub import HfApi, hf_hub_download

    repo_id = "Hajorda/flameye-wildfire-detection"
    files = [name for name in HfApi().list_repo_files(repo_id, repo_type="dataset") if name.endswith(".parquet")]
    image_dir = output_dir / "images" / "benchmark"
    label_dir = output_dir / "labels" / "benchmark"
    count = 0
    for filename in files:
        parquet_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
        frame = pd.read_parquet(parquet_path)
        for row in frame.to_dict(orient="records"):
            if str(row.get("source", "")).lower() != source.lower():
                continue
            image_id = str(row.get("image_id") or row.get("id") or f"{source}_{count:06d}")
            if save_hf_image(row.get("image"), image_dir / f"{image_id}.jpg"):
                write_yolo_labels(first_present(row.get("annotations"), row.get("objects"), []), label_dir / f"{image_id}.txt")
                count += 1
            if max_images is not None and count >= max_images:
                break
        if max_images is not None and count >= max_images:
            break
    return {"dataset": source, "source": "Hajorda/flameye-wildfire-detection", "images": count, "path": str(output_dir)}


def download_sentinel_cems(output_dir: Path, max_images: int | None) -> dict[str, Any]:
    import pandas as pd
    from huggingface_hub import HfApi, hf_hub_download

    repo_id = "9334hq/wildfires-cems"
    files = [name for name in HfApi().list_repo_files(repo_id, repo_type="dataset") if name.endswith(".parquet")]
    image_dir = output_dir / "images" / "benchmark"
    count = 0
    for filename in files:
        parquet_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
        frame = pd.read_parquet(parquet_path)
        for row in frame.to_dict(orient="records"):
            image_value = first_present(row.get("image"), row.get("S2L2A"), row.get("s2l2a"))
            if save_hf_image(image_value, image_dir / f"sentinel2_cems_{count:06d}.jpg"):
                count += 1
            if max_images is not None and count >= max_images:
                break
        if max_images is not None and count >= max_images:
            break
    return {"dataset": "sentinel2_cems", "source": "9334hq/wildfires-cems", "images": count, "path": str(output_dir)}


def prepare_flame_from_github(output_dir: Path, max_images: int | None) -> dict[str, Any]:
    repo_url = "https://github.com/AlirezaShamsoshoara/Fire-Detection-UAV-Aerial-Image-Classification-Segmentation-UnmannedAerialVehicle.git"
    raw_dir = output_dir / "_source_repo"
    if not raw_dir.exists():
        subprocess.run(["git", "clone", "--depth", "1", repo_url, str(raw_dir)], check=True)
    image_dir = output_dir / "images" / "benchmark"
    image_dir.mkdir(parents=True, exist_ok=True)
    image_exts = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}
    source_images = sorted(path for path in raw_dir.rglob("*") if path.suffix.lower() in image_exts)
    count = 0
    for image_path in source_images:
        target = image_dir / f"flame_{count:06d}{image_path.suffix.lower()}"
        if not target.exists():
            shutil.copy2(image_path, target)
        count += 1
        if max_images is not None and count >= max_images:
            break
    return {
        "dataset": "flame",
        "source": repo_url,
        "images": count,
        "path": str(output_dir),
        "note": "The full FLAME dataset is hosted by IEEE DataPort; this prepares repository-available imagery automatically.",
    }


def validate_prepared_dataset(path: Path) -> dict[str, Any]:
    image_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
    images = sorted(p for p in path.rglob("*") if p.suffix.lower() in image_exts)
    labels = sorted(p for p in path.rglob("*.txt"))
    readable = 0
    for image_path in images[:100]:
        try:
            with Image.open(image_path) as image:
                image.verify()
            readable += 1
        except Exception:
            pass
    return {
        "path": str(path),
        "images": len(images),
        "labels": len(labels),
        "readable_sample": readable,
        "valid": len(images) > 0 and readable > 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and prepare wildfire datasets for CompressAI validation.")
    parser.add_argument("--output-root", type=Path, default=Path("datasets"))
    parser.add_argument("--max-images", type=int, default=None, help="Optional cap for smoke tests. Omit for full acquisition.")
    parser.add_argument("--skip-sentinel2", action="store_true")
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    manifest.append(download_flameye_source("dfire", args.output_root / "dfire", args.max_images))
    try:
        manifest.append(prepare_flame_from_github(args.output_root / "flame", args.max_images))
    except Exception as exc:
        manifest.append(
            {
                "dataset": "flame",
                "images": 0,
                "path": str(args.output_root / "flame"),
                "error": str(exc),
                "note": "Download the full FLAME IEEE DataPort files into datasets/flame when authenticated access is required.",
            }
        )
    if not args.skip_sentinel2:
        try:
            manifest.append(download_sentinel_cems(args.output_root / "sentinel2_cems", args.max_images))
        except Exception as exc:
            manifest.append({"dataset": "sentinel2_cems", "images": 0, "error": str(exc)})

    integrity = {Path(item["path"]).name: validate_prepared_dataset(Path(item["path"])) for item in manifest if item.get("path")}
    payload = {"downloads": manifest, "integrity": integrity}
    (args.output_root / "download_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
