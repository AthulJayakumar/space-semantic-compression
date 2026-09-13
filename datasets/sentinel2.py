"""datasets.sentinel2

Plain-English purpose: Dataset loader code only; raw imagery is intentionally not committed.

This file is part of the public Space Semantic Compression research demo.
It is documented at module level so researchers, supervisors, and non-specialist
readers can understand where the file fits before reading implementation details.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Sentinel2Item:
    image_path: Path
    rgb_path: Path
    metadata_path: Path | None
    bands: dict[str, Path]
    label_mask_path: Path | None = None


class Sentinel2Dataset:
    """Sentinel-2 Level-2A and Sentinel Hub export loader.

    Expected local structure:
    datasets/sentinel2/
      images/        RGB PNG/JPG/TIFF or Sentinel Hub GeoTIFF exports
      metadata/      optional JSON sidecars with bbox/acquisition metadata
      labels/        optional FIRMS-derived masks
    """

    image_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}
    band_tokens = {
        "B02": ("b02", "blue"),
        "B03": ("b03", "green"),
        "B04": ("b04", "red"),
        "B08": ("b08", "nir"),
        "B11": ("b11", "swir1"),
        "B12": ("b12", "swir2"),
    }

    def __init__(self, root: str | Path = "datasets/sentinel2") -> None:
        self.root = Path(root)
        self.images_dir = self.root / "images"
        self.metadata_dir = self.root / "metadata"
        self.labels_dir = self.root / "labels"

    def __iter__(self) -> Iterator[Sentinel2Item]:
        for image_path in self.discover_rgb_images():
            metadata_path = self.metadata_for(image_path)
            yield Sentinel2Item(
                image_path=image_path,
                rgb_path=self.ensure_rgb_composite(image_path),
                metadata_path=metadata_path,
                bands=self.discover_bands(image_path),
                label_mask_path=self.label_mask_for(image_path),
            )

    def __len__(self) -> int:
        return len(self.discover_rgb_images())

    def discover_rgb_images(self) -> list[Path]:
        if not self.images_dir.exists():
            return []
        candidates = sorted(path for path in self.images_dir.rglob("*") if path.suffix.lower() in self.image_exts)
        rgb_like = [
            path
            for path in candidates
            if not self._is_band_file(path) and "mask" not in path.stem.lower() and "label" not in path.stem.lower()
        ]
        return rgb_like or candidates

    def discover_bands(self, image_path: Path) -> dict[str, Path]:
        parent = image_path.parent
        bands: dict[str, Path] = {}
        for band, tokens in self.band_tokens.items():
            for path in parent.glob("*"):
                stem = path.stem.lower()
                if path.suffix.lower() in self.image_exts and any(token in stem for token in tokens):
                    bands[band] = path
                    break
        return bands

    def ensure_rgb_composite(self, image_path: Path) -> Path:
        if image_path.suffix.lower() not in {".tif", ".tiff"}:
            return image_path
        rgb_dir = self.root / "rgb"
        rgb_dir.mkdir(parents=True, exist_ok=True)
        target = rgb_dir / f"{image_path.stem}.png"
        if target.exists():
            return target
        try:
            import tifffile

            data = tifffile.imread(image_path)
            rgb = self._to_rgb_array(data)
            Image.fromarray(rgb).save(target)
            return target
        except Exception:
            return image_path

    def metadata_for(self, image_path: Path) -> Path | None:
        candidates = [
            self.metadata_dir / f"{image_path.stem}.json",
            image_path.with_suffix(".json"),
        ]
        return next((path for path in candidates if path.exists()), None)

    def label_mask_for(self, image_path: Path) -> Path | None:
        candidates = [
            self.labels_dir / f"{image_path.stem}_firms_mask.png",
            self.labels_dir / f"{image_path.stem}.png",
        ]
        return next((path for path in candidates if path.exists()), None)

    def read_metadata(self, item: Sentinel2Item) -> dict[str, object]:
        if item.metadata_path is None or not item.metadata_path.exists():
            return {}
        return json.loads(item.metadata_path.read_text(encoding="utf-8"))

    def _is_band_file(self, path: Path) -> bool:
        stem = path.stem.lower()
        return any(token in stem for tokens in self.band_tokens.values() for token in tokens)

    def _to_rgb_array(self, data: np.ndarray) -> np.ndarray:
        arr = np.asarray(data)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.ndim == 3 and arr.shape[0] in {3, 4, 10, 12, 13}:
            arr = np.moveaxis(arr, 0, -1)
        if arr.shape[-1] >= 4:
            arr = arr[..., [2, 1, 0]]
        elif arr.shape[-1] >= 3:
            arr = arr[..., :3]
        else:
            arr = np.repeat(arr[..., :1], 3, axis=-1)
        arr = arr.astype("float32")
        high = np.percentile(arr, 98)
        low = np.percentile(arr, 2)
        arr = np.clip((arr - low) / max(high - low, 1e-6), 0.0, 1.0)
        return (arr * 255).astype("uint8")


def discover_sentinel2_dataset(root: str | Path = "datasets/sentinel2") -> Sentinel2Dataset | None:
    dataset = Sentinel2Dataset(root)
    return dataset if len(dataset) > 0 else None
