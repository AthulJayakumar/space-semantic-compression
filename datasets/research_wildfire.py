"""Utilities for large wildfire Earth Observation research datasets.

The public repository does not store raw satellite imagery. This module keeps a
small, reproducible catalogue of recommended external datasets and provides
helpers for validating local downloads and building image/mask manifests.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
MASK_KEYWORDS = ("mask", "masks", "label", "labels", "raster", "rasters", "burn", "burned")


@dataclass(frozen=True)
class ResearchDatasetSpec:
    """Download metadata for one external wildfire EO dataset."""

    name: str
    repo_id: str
    local_dir: str
    priority: str
    task: str
    license: str
    expected_scale: str
    notes: str
    small_allow_patterns: tuple[str, ...] = ()
    standard_allow_patterns: tuple[str, ...] = ()
    full_allow_patterns: tuple[str, ...] = ()


@dataclass(frozen=True)
class ManifestRow:
    """One paired training or benchmark item."""

    dataset: str
    image_path: str
    mask_path: str
    split: str
    width: int
    height: int
    source_id: str


RECOMMENDED_DATASETS: dict[str, ResearchDatasetSpec] = {
    "cems_hls": ResearchDatasetSpec(
        name="cems_hls",
        repo_id="morenoj11/CEMS-HLS",
        local_dir="cems_hls",
        priority="start_here",
        task="burn-scar segmentation",
        license="apache-2.0",
        expected_scale="<1K paired scenes",
        notes="Good first mask-supervised satellite dataset because it is compact and wildfire-specific.",
        small_allow_patterns=("**/*.png", "**/*.jpg", "**/*.jpeg", "**/*.tif", "**/*.tiff", "**/*.json", "**/*.csv"),
        standard_allow_patterns=("**/*",),
        full_allow_patterns=("**/*",),
    ),
    "hls_burn_scars": ResearchDatasetSpec(
        name="hls_burn_scars",
        repo_id="ibm-nasa-geospatial/hls_burn_scars",
        local_dir="hls_burn_scars",
        priority="start_here",
        task="burn-scar segmentation",
        license="cc-by-4.0",
        expected_scale="804 scenes with masks",
        notes="Harmonized Landsat/Sentinel-2 burn-scar data; useful for real mask supervision.",
        small_allow_patterns=("*.py", "*.md", "*.json", "**/*.png", "**/*.jpg", "**/*.tif", "**/*.tiff"),
        standard_allow_patterns=("**/*",),
        full_allow_patterns=("**/*",),
    ),
    "firescope_small": ResearchDatasetSpec(
        name="firescope_small",
        repo_id="INSAIT-Institute/FireScope-Bench",
        local_dir="firescope_bench",
        priority="scale_after_first_pass",
        task="Sentinel-2 wildfire event/risk masks",
        license="cc-by-4.0",
        expected_scale="10K-100K rows; full dataset is very large",
        notes="Use the small-sample or event-mask subsets first; full acquisition can be >100GB.",
        small_allow_patterns=(
            "usa/wildfire_rasters/small_sample/**",
            "README.md",
            "climate_data.json",
        ),
        standard_allow_patterns=(
            "usa/wildfire_rasters/small_sample/**",
            "usa/wildfire_events/images/**",
            "usa/wildfire_events/masks/**",
            "europe/wildfire_events/images/**",
            "europe/wildfire_events/masks/**",
            "README.md",
            "climate_data.json",
        ),
        full_allow_patterns=("**/*",),
    ),
    "eo4wildfires": ResearchDatasetSpec(
        name="eo4wildfires",
        repo_id="AUA-Informatics-Lab/eo4wildfires",
        local_dir="eo4wildfires",
        priority="later_large_scale",
        task="multi-sensor wildfire impact prediction",
        license="cc-by-sa-4.0",
        expected_scale="31K+ wildfire events",
        notes="Excellent for later scale/generalization, but it requires a heavier geospatial stack.",
        small_allow_patterns=("README.md", "files_test.csv.gz", "files_val.csv.gz", "how-to-use-eo4wildfires.ipynb"),
        standard_allow_patterns=("README.md", "files_*.csv.gz", "how-to-use-eo4wildfires.ipynb"),
        full_allow_patterns=("**/*",),
    ),
}


def dataset_plan(names: Iterable[str] | None = None) -> list[dict[str, object]]:
    """Return a plain serializable description of selected recommended datasets."""

    selected = selected_specs(names)
    return [asdict(spec) for spec in selected]


def selected_specs(names: Iterable[str] | None = None) -> list[ResearchDatasetSpec]:
    if names is None:
        return list(RECOMMENDED_DATASETS.values())
    specs: list[ResearchDatasetSpec] = []
    for name in names:
        key = name.strip().lower()
        if key not in RECOMMENDED_DATASETS:
            raise ValueError(f"Unknown dataset '{name}'. Choices: {', '.join(RECOMMENDED_DATASETS)}")
        specs.append(RECOMMENDED_DATASETS[key])
    return specs


def allow_patterns_for(spec: ResearchDatasetSpec, profile: str) -> tuple[str, ...]:
    """Return Hugging Face snapshot allow-patterns for the requested acquisition size."""

    if profile == "small":
        return spec.small_allow_patterns
    if profile == "standard":
        return spec.standard_allow_patterns or spec.small_allow_patterns
    if profile == "full":
        return spec.full_allow_patterns or spec.standard_allow_patterns or spec.small_allow_patterns
    raise ValueError("profile must be one of: small, standard, full")


def build_manifest(dataset_root: Path, dataset_name: str, split: str = "train") -> list[ManifestRow]:
    """Scan a prepared dataset folder and pair likely images with likely masks."""

    images = [path for path in discover_images(dataset_root) if not looks_like_mask(path)]
    masks = [path for path in discover_images(dataset_root) if looks_like_mask(path)]
    mask_index = index_masks(masks)
    rows: list[ManifestRow] = []
    for image_path in images:
        mask_path = match_mask(image_path, mask_index)
        if mask_path is None:
            continue
        width, height = image_size(image_path)
        rows.append(
            ManifestRow(
                dataset=dataset_name,
                image_path=str(image_path),
                mask_path=str(mask_path),
                split=infer_split(image_path, split),
                width=width,
                height=height,
                source_id=image_path.stem,
            )
        )
    return rows


def validate_manifest(rows: list[ManifestRow], sample_limit: int = 100) -> dict[str, object]:
    """Check whether manifest image/mask paths exist and are readable."""

    checked = rows[:sample_limit]
    readable_images = 0
    readable_masks = 0
    for row in checked:
        if can_open_image(Path(row.image_path)):
            readable_images += 1
        if can_open_image(Path(row.mask_path)):
            readable_masks += 1
    return {
        "items": len(rows),
        "checked": len(checked),
        "readable_images": readable_images,
        "readable_masks": readable_masks,
        "valid": len(rows) > 0 and readable_images == len(checked) and readable_masks == len(checked),
    }


def write_manifest(rows: list[ManifestRow], csv_path: Path, json_path: Path | None = None) -> None:
    """Write paired image/mask rows in CSV and optional JSON form."""

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(ManifestRow.__dataclass_fields__.keys())
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps([asdict(row) for row in rows], indent=2), encoding="utf-8")


def read_manifest(csv_path: Path) -> list[ManifestRow]:
    """Read a paired image/mask manifest generated by this module."""

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        return [
            ManifestRow(
                dataset=row["dataset"],
                image_path=row["image_path"],
                mask_path=row["mask_path"],
                split=row.get("split", "train"),
                width=int(float(row.get("width", 0) or 0)),
                height=int(float(row.get("height", 0) or 0)),
                source_id=row.get("source_id", Path(row["image_path"]).stem),
            )
            for row in csv.DictReader(handle)
        ]


def discover_images(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTS)


def looks_like_mask(path: Path) -> bool:
    parts = [part.lower() for part in path.parts]
    stem = path.stem.lower()
    return any(keyword in stem for keyword in MASK_KEYWORDS) or any(part in MASK_KEYWORDS for part in parts)


def index_masks(masks: Iterable[Path]) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for mask in masks:
        for key in mask_keys(mask):
            index.setdefault(key, mask)
    return index


def match_mask(image_path: Path, mask_index: dict[str, Path]) -> Path | None:
    for key in mask_keys(image_path):
        if key in mask_index:
            return mask_index[key]
    return None


def mask_keys(path: Path) -> set[str]:
    stem = path.stem.lower()
    cleaned = stem
    for token in (
        "_mask",
        "-mask",
        "_label",
        "-label",
        "_raster",
        "-raster",
        "_burned",
        "-burned",
        "_satellite",
        "-satellite",
    ):
        cleaned = cleaned.replace(token, "")
    return {
        stem,
        cleaned,
        cleaned.replace("_image", ""),
        cleaned.replace("-image", ""),
        cleaned.replace("_rgb", ""),
        cleaned.replace("-rgb", ""),
    }


def infer_split(path: Path, fallback: str) -> str:
    tokens = {part.lower() for part in path.parts}
    if tokens & {"validation", "valid", "val"}:
        return "validation"
    if "test" in tokens:
        return "test"
    if "train" in tokens:
        return "train"
    return fallback


def image_size(path: Path) -> tuple[int, int]:
    if path.suffix.lower() in {".tif", ".tiff"}:
        try:
            import tifffile

            data = tifffile.imread(path)
            height, width = spatial_shape(data)
            return (width, height)
        except Exception:
            return (0, 0)
    try:
        with Image.open(path) as image:
            return image.size
    except Exception:
        try:
            import tifffile

            data = tifffile.imread(path)
            height, width = spatial_shape(data)
            return (width, height)
        except Exception:
            return (0, 0)


def can_open_image(path: Path) -> bool:
    if path.suffix.lower() in {".tif", ".tiff"}:
        try:
            import tifffile

            data = tifffile.imread(path)
            return data.size > 0
        except Exception:
            return False
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        try:
            import tifffile

            data = tifffile.imread(path)
            return data.size > 0
        except Exception:
            return False


def load_rgb_image(path: Path) -> Image.Image:
    """Load RGB imagery, including multispectral GeoTIFFs."""

    if path.suffix.lower() in {".tif", ".tiff"}:
        import tifffile

        data = tifffile.imread(path)
        return Image.fromarray(to_rgb_array(data))
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        import tifffile

        data = tifffile.imread(path)
        return Image.fromarray(to_rgb_array(data))


def load_grayscale_image(path: Path) -> Image.Image:
    """Load a label mask as a grayscale image, including single-band TIFFs."""

    if path.suffix.lower() in {".tif", ".tiff"}:
        import tifffile

        data = np.asarray(tifffile.imread(path))
        if data.ndim == 3:
            data = data[..., 0] if data.shape[-1] < data.shape[0] else data[0]
        data = normalize_to_uint8(data)
        return Image.fromarray(data).convert("L")
    try:
        return Image.open(path).convert("L")
    except Exception:
        import tifffile

        data = np.asarray(tifffile.imread(path))
        if data.ndim == 3:
            data = data[..., 0] if data.shape[-1] < data.shape[0] else data[0]
        data = normalize_to_uint8(data)
        return Image.fromarray(data).convert("L")


def spatial_shape(data: object) -> tuple[int, int]:
    arr = np.asarray(data)
    if arr.ndim == 2:
        return int(arr.shape[0]), int(arr.shape[1])
    if arr.ndim == 3 and arr.shape[0] <= 16 and arr.shape[1] > 16:
        return int(arr.shape[1]), int(arr.shape[2])
    return int(arr.shape[0]), int(arr.shape[1])


def to_rgb_array(data: object) -> np.ndarray:
    arr = np.asarray(data)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim == 3 and arr.shape[0] <= 16 and arr.shape[1] > 16:
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim != 3:
        raise ValueError("Expected a 2D or 3D image array.")
    if arr.shape[-1] >= 4:
        arr = arr[..., [2, 1, 0]]
    elif arr.shape[-1] >= 3:
        arr = arr[..., :3]
    else:
        arr = np.repeat(arr[..., :1], 3, axis=-1)
    return normalize_to_uint8(arr)


def normalize_to_uint8(values: object) -> np.ndarray:
    arr = np.asarray(values).astype("float32")
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros(arr.shape, dtype="uint8")
    lo = float(np.percentile(arr[finite], 2))
    hi = float(np.percentile(arr[finite], 98))
    if hi - lo < 1e-8:
        return np.zeros(arr.shape, dtype="uint8")
    return (np.clip((arr - lo) / (hi - lo), 0.0, 1.0) * 255).astype("uint8")
