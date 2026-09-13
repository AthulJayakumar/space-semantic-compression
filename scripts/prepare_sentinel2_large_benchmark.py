"""Prepare a larger Sentinel-2 patch benchmark from local imagery.

The public repository does not ship raw Sentinel-2 scenes. When a local dataset
is available, this script tiles each image into fixed-size patches and writes a
manifest. It is intended for scaling from the initial 100-scene evidence toward
500+ benchmark patches for stronger journal-style validation.
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


def patch_positions(width: int, height: int, patch_size: int, stride: int) -> list[tuple[int, int]]:
    """Return deterministic top-left patch positions for one image."""

    if width <= patch_size or height <= patch_size:
        return [(0, 0)]

    xs = list(range(0, max(width - patch_size + 1, 1), stride))
    ys = list(range(0, max(height - patch_size + 1, 1), stride))
    if xs[-1] != width - patch_size:
        xs.append(width - patch_size)
    if ys[-1] != height - patch_size:
        ys.append(height - patch_size)

    positions = [(x, y) for y in ys for x in xs]
    center = ((width - patch_size) // 2, (height - patch_size) // 2)
    if center not in positions:
        positions.insert(0, center)
    return positions


def make_patch(image: Image.Image, left: int, top: int, patch_size: int) -> Image.Image:
    """Crop one patch, resizing very small source images when necessary."""

    image = image.convert("RGB")
    if image.width < patch_size or image.height < patch_size:
        scale = patch_size / min(image.width, image.height)
        image = image.resize((int(image.width * scale), int(image.height * scale)), Image.Resampling.LANCZOS)
        left = max(0, (image.width - patch_size) // 2)
        top = max(0, (image.height - patch_size) // 2)
    return image.crop((left, top, left + patch_size, top + patch_size))


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a 500+ Sentinel-2 fixed-patch benchmark.")
    parser.add_argument("--source-root", type=Path, default=Path("datasets/sentinel2"))
    parser.add_argument("--output-root", type=Path, default=Path("datasets/sentinel2_500_patch"))
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=384)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--max-patches-per-image", type=int, default=8)
    args = parser.parse_args()

    images_dir = args.output_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    dataset = Sentinel2Dataset(args.source_root)
    for source_index, item in enumerate(dataset):
        if len(rows) >= args.limit:
            break
        try:
            image = Image.open(item.rgb_path).convert("RGB")
        except Exception as exc:
            rows.append({"source": str(item.rgb_path), "error": str(exc), "skipped": True})
            continue

        positions = patch_positions(image.width, image.height, args.patch_size, args.stride)
        for patch_index, (left, top) in enumerate(positions[: args.max_patches_per_image]):
            if len(rows) >= args.limit:
                break
            patch = make_patch(image, left, top, args.patch_size)
            target = images_dir / f"sentinel2_patch_{len(rows):05d}.png"
            patch.save(target)
            rows.append(
                {
                    "patch_id": len(rows),
                    "source_index": source_index,
                    "source": str(item.rgb_path),
                    "target": str(target),
                    "left": left,
                    "top": top,
                    "patch_size": args.patch_size,
                    "metadata": str(item.metadata_path or ""),
                    "label_mask": str(item.label_mask_path or ""),
                }
            )

    manifest = {
        "source_root": str(args.source_root),
        "output_root": str(args.output_root),
        "patch_size": args.patch_size,
        "stride": args.stride,
        "limit": args.limit,
        "patches": len([row for row in rows if not row.get("skipped")]),
        "items": rows,
    }
    manifest_path = args.output_root / "patch_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"patches": manifest["patches"], "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
