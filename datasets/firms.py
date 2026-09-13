"""datasets.firms

Plain-English purpose: Dataset loader code only; raw imagery is intentionally not committed.

This file is part of the public Space Semantic Compression research demo.
It is documented at module level so researchers, supervisors, and non-specialist
readers can understand where the file fits before reading implementation details.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from datasets.sentinel2 import Sentinel2Dataset, Sentinel2Item


@dataclass(frozen=True)
class FirmsDetection:
    latitude: float
    longitude: float
    confidence: float
    brightness: float | None = None
    acquisition_date: str | None = None


class FirmsDataset:
    """NASA FIRMS CSV loader and Sentinel-2 mask generator."""

    def __init__(self, csv_path: str | Path) -> None:
        self.csv_path = Path(csv_path)

    def detections(self) -> list[FirmsDetection]:
        if not self.csv_path.exists():
            return []
        detections: list[FirmsDetection] = []
        with self.csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                try:
                    confidence_raw = row.get("confidence", "0")
                    confidence = self._confidence_to_float(confidence_raw)
                    detections.append(
                        FirmsDetection(
                            latitude=float(row.get("latitude") or row.get("lat")),
                            longitude=float(row.get("longitude") or row.get("lon")),
                            confidence=confidence,
                            brightness=self._optional_float(row.get("bright_ti4") or row.get("brightness")),
                            acquisition_date=row.get("acq_date"),
                        )
                    )
                except Exception:
                    continue
        return detections

    def generate_masks(self, sentinel_dataset: Sentinel2Dataset, output_dir: str | Path | None = None, radius_px: int = 6) -> list[Path]:
        output = Path(output_dir) if output_dir else sentinel_dataset.labels_dir
        output.mkdir(parents=True, exist_ok=True)
        detections = self.detections()
        masks: list[Path] = []
        for item in sentinel_dataset:
            metadata = sentinel_dataset.read_metadata(item)
            bbox = self._bbox_from_metadata(metadata)
            if bbox is None:
                continue
            image = Image.open(item.rgb_path)
            mask = np.zeros((image.height, image.width), dtype="float32")
            min_lon, min_lat, max_lon, max_lat = bbox
            for detection in detections:
                if not (min_lon <= detection.longitude <= max_lon and min_lat <= detection.latitude <= max_lat):
                    continue
                x = int((detection.longitude - min_lon) / max(max_lon - min_lon, 1e-9) * (image.width - 1))
                y = int((max_lat - detection.latitude) / max(max_lat - min_lat, 1e-9) * (image.height - 1))
                self._draw_disk(mask, x, y, radius_px, detection.confidence)
            path = output / f"{item.rgb_path.stem}_firms_mask.png"
            Image.fromarray((np.clip(mask, 0, 1) * 255).astype("uint8")).save(path)
            masks.append(path)
        return masks

    def _bbox_from_metadata(self, metadata: dict[str, object]) -> tuple[float, float, float, float] | None:
        bbox = metadata.get("bbox") or metadata.get("bounds")
        if isinstance(bbox, list) and len(bbox) == 4:
            return tuple(float(value) for value in bbox)  # type: ignore[return-value]
        if all(key in metadata for key in ["min_lon", "min_lat", "max_lon", "max_lat"]):
            return (
                float(metadata["min_lon"]),
                float(metadata["min_lat"]),
                float(metadata["max_lon"]),
                float(metadata["max_lat"]),
            )
        return None

    def _draw_disk(self, mask: np.ndarray, x: int, y: int, radius: int, value: float) -> None:
        yy, xx = np.ogrid[: mask.shape[0], : mask.shape[1]]
        disk = (xx - x) ** 2 + (yy - y) ** 2 <= radius**2
        mask[disk] = np.maximum(mask[disk], value)

    def _confidence_to_float(self, value: object) -> float:
        if isinstance(value, str):
            token = value.lower()
            if token == "l":
                return 0.35
            if token == "n":
                return 0.65
            if token == "h":
                return 0.9
        try:
            numeric = float(value)
            return numeric / 100.0 if numeric > 1 else numeric
        except Exception:
            return 0.0

    def _optional_float(self, value: object) -> float | None:
        try:
            return float(value)
        except Exception:
            return None
