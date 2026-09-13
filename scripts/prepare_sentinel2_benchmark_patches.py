"""scripts.prepare_sentinel2_benchmark_patches

Plain-English purpose: Command-line runners for datasets, benchmarks, reports, and exports.

This file is part of the public Space Semantic Compression research demo.
It is documented at module level so researchers, supervisors, and non-specialist
readers can understand where the file fits before reading implementation details.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.sentinel2 import Sentinel2Dataset


def center_resize(image: Image.Image, size: int) -> Image.Image:
    image = image.convert("RGB")
    width, height = image.size
    scale = size / min(width, height)
    resized = image.resize((int(width * scale), int(height * scale)), Image.Resampling.LANCZOS)
    left = max(0, (resized.width - size) // 2)
    top = max(0, (resized.height - size) // 2)
    return resized.crop((left, top, left + size, top + size))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare fixed-size Sentinel-2 benchmark patches.")
    parser.add_argument("--source-root", type=Path, default=Path("datasets/sentinel2"))
    parser.add_argument("--output-root", type=Path, default=Path("datasets/sentinel2_100_patch"))
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    dataset = Sentinel2Dataset(args.source_root)
    images_dir = args.output_root / "images"
    metadata_dir = args.output_root / "metadata"
    labels_dir = args.output_root / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx, item in enumerate(dataset):
        if idx >= args.limit:
            break
        image = center_resize(Image.open(item.rgb_path), args.size)
        target = images_dir / f"sentinel2_cems_{idx:04d}.png"
        image.save(target)
        rows.append(
            {
                "source": str(item.rgb_path),
                "target": str(target),
                "size": args.size,
                "metadata": str(item.metadata_path or ""),
                "label_mask": str(item.label_mask_path or ""),
            }
        )
    manifest = {
        "source_root": str(args.source_root),
        "output_root": str(args.output_root),
        "patch_size": args.size,
        "images": len(rows),
        "items": rows,
    }
    (args.output_root / "patch_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"images": len(rows), "output_root": str(args.output_root)}, indent=2))


if __name__ == "__main__":
    main()
