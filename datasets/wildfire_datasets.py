"""datasets.wildfire_datasets

Plain-English purpose: Dataset loader code only; raw imagery is intentionally not committed.

This file is part of the public Space Semantic Compression research demo.
It is documented at module level so researchers, supervisors, and non-specialist
readers can understand where the file fits before reading implementation details.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class DatasetItem:
    image_path: Path
    label_path: Path | None = None
    metadata: dict[str, str] | None = None
    split: str = "benchmark"


class WildfireImageDataset:
    image_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}

    def __init__(
        self,
        root: str | Path,
        split: str = "benchmark",
        metadata_csv: str | Path | None = None,
        label_dir: str | Path | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.split_dir = self.root / split if (self.root / split).exists() else self.root
        self.label_dir = Path(label_dir) if label_dir else self.root / "labels"
        self.metadata = self._read_metadata(Path(metadata_csv)) if metadata_csv else {}

    def __iter__(self) -> Iterator[DatasetItem]:
        for image_path in self._image_paths():
            label_path = self._label_for(image_path)
            metadata = self.metadata.get(image_path.name, {})
            yield DatasetItem(image_path=image_path, label_path=label_path, metadata=metadata, split=self.split)

    def __len__(self) -> int:
        return len(list(self._image_paths()))

    def _image_paths(self) -> list[Path]:
        return sorted(path for path in self.split_dir.rglob("*") if path.suffix.lower() in self.image_exts)

    def _label_for(self, image_path: Path) -> Path | None:
        candidates = [
            self.label_dir / f"{image_path.stem}.geojson",
            self.label_dir / f"{image_path.stem}.json",
            self.label_dir / f"{image_path.stem}.csv",
            self.label_dir / f"{image_path.stem}.png",
            self.label_dir / f"{image_path.stem}.txt",
        ]
        return next((path for path in candidates if path.exists()), None)

    def _read_metadata(self, metadata_csv: Path) -> dict[str, dict[str, str]]:
        if not metadata_csv.exists():
            return {}
        with metadata_csv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = {}
            for row in reader:
                key = row.get("filename") or row.get("image") or row.get("image_id")
                if key:
                    rows[Path(key).name] = dict(row)
            return rows


class Sentinel2WildfireDataset(WildfireImageDataset):
    """Local Sentinel-2 wildfire imagery loader with optional labels/metadata alignment."""


class ModisActiveFireDataset(WildfireImageDataset):
    """Local MODIS active-fire product loader with optional fire-label metadata."""


class FirmsWildfireDataset(WildfireImageDataset):
    """Local FIRMS-linked wildfire dataset loader for benchmark/evaluation splits."""


class DFireDataset(WildfireImageDataset):
    """DFire wildfire/smoke dataset adapter with YOLO label alignment."""

    def __init__(self, root: str | Path, split: str = "benchmark") -> None:
        root = Path(root)
        super().__init__(root=root, split=split)
        self.split_dir = self._resolve_split_dir(root, split)
        self.label_dir = self._resolve_label_dir(root, split)

    def _resolve_split_dir(self, root: Path, split: str) -> Path:
        candidates: list[Path] = []
        for name in self._split_aliases(split):
            candidates.extend([root / "images" / name, root / name / "images", root / name])
        candidates.extend([root / "images", root])
        return next((path for path in candidates if path.exists()), root)

    def _resolve_label_dir(self, root: Path, split: str) -> Path:
        candidates: list[Path] = []
        for name in self._split_aliases(split):
            candidates.extend([root / "labels" / name, root / name / "labels", root / name])
        candidates.extend([root / "labels", root])
        return next((path for path in candidates if path.exists()), root / "labels")

    def _split_aliases(self, split: str) -> list[str]:
        if split == "validation":
            return ["validation", "val", "valid"]
        if split == "benchmark":
            return ["benchmark", "test", "val", "valid", "validation"]
        return [split]


class FLAMEDataset(WildfireImageDataset):
    """FLAME wildfire imagery adapter with flexible fire/non-fire folder detection."""

    def __init__(self, root: str | Path, split: str = "benchmark") -> None:
        root = Path(root)
        super().__init__(root=root, split=split)
        self.split_dir = self._resolve_split_dir(root, split)

    def __iter__(self) -> Iterator[DatasetItem]:
        for image_path in self._image_paths():
            parent_tokens = {part.lower() for part in image_path.parts}
            label = "fire" if any(token in parent_tokens for token in {"fire", "wildfire", "flame"}) else "unknown"
            if any(token in parent_tokens for token in {"nofire", "non_fire", "non-fire", "normal"}):
                label = "non_fire"
            metadata = {"class": label, "dataset": "FLAME"}
            yield DatasetItem(image_path=image_path, label_path=self._label_for(image_path), metadata=metadata, split=self.split)

    def _resolve_split_dir(self, root: Path, split: str) -> Path:
        aliases = {
            "validation": ["validation", "val", "valid", "test", "Testing", "Test"],
            "benchmark": ["benchmark", "test", "Testing", "Test", "val", "validation"],
            "train": ["train", "Training", "Train"],
        }.get(split, [split])
        return next((root / alias for alias in aliases if (root / alias).exists()), root)


def discover_wildfire_datasets(base_dir: str | Path = "datasets", split: str = "benchmark") -> dict[str, WildfireImageDataset]:
    base = Path(base_dir)
    candidates: dict[str, type[WildfireImageDataset]] = {
        "dfire": DFireDataset,
        "flame": FLAMEDataset,
    }
    discovered: dict[str, WildfireImageDataset] = {}
    for name, dataset_cls in candidates.items():
        for path in [base / name, base / name.upper(), base / name.capitalize()]:
            if path.exists():
                dataset = dataset_cls(path, split=split)
                if len(dataset) > 0:
                    discovered[name] = dataset
                    break
    return discovered
