"""Prepare a frozen, event-balanced ImpactMesh-Fire external lockbox."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("datasets/impactmesh_fire_v1"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/external_lockbox_impactmesh_100"),
    )
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument(
        "--dataset-revision",
        default="698aed80c24ac7e9b8ca2e70e5bd6960c29a3d55",
    )
    args = parser.parse_args()

    holdout_path = args.source_root / "split" / "impactmesh_fire_test_holdout.txt"
    image_root = args.source_root / "extracted" / "S2RGB"
    mask_root = args.source_root / "extracted" / "MASK"
    sample_ids = [line.strip() for line in holdout_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = event_balanced_selection(sample_ids, args.samples, args.seed)

    image_output = args.output_dir / "images"
    mask_output = args.output_dir / "masks"
    image_output.mkdir(parents=True, exist_ok=True)
    mask_output.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for index, sample_id in enumerate(selected, start=1):
        print(f"preparing ImpactMesh lockbox {index}/{len(selected)}: {sample_id}", flush=True)
        source_image = image_root / f"{sample_id}_S2RGB.tif"
        source_mask = mask_root / f"{sample_id}_annotation_wildfire.tif"
        if not source_image.exists() or not source_mask.exists():
            raise FileNotFoundError(f"Missing image or mask for {sample_id}")

        image_array = normalize_rgb(tifffile.imread(source_image))
        mask_array = np.asarray(tifffile.imread(source_mask)) > 0
        if image_array.shape[:2] != mask_array.shape:
            raise ValueError(f"Image/mask shape mismatch for {sample_id}")

        image_path = image_output / f"{sample_id}.png"
        mask_path = mask_output / f"{sample_id}.png"
        Image.fromarray(image_array).save(image_path, optimize=True)
        Image.fromarray(np.where(mask_array, 255, 0).astype("uint8")).save(mask_path, optimize=True)

        positive_fraction = float(mask_array.mean())
        rows.append(
            {
                "sample_id": sample_id,
                "dataset": "impactmesh_fire_v1_test_holdout",
                "image_path": str(image_path.resolve()),
                "mask_path": str(mask_path.resolve()),
                "split": "external_test",
                "source_split": "test_holdout",
                "geographic_group": event_id(sample_id),
                "event_group": event_id(sample_id),
                "label_source": "Copernicus Emergency Management Service",
                "image_unit": "independent_256x256_sentinel2_rgb_tile",
                "fire_status": "positive" if positive_fraction > 0 else "negative",
                "positive_pixel_fraction": positive_fraction,
                "width": image_array.shape[1],
                "height": image_array.shape[0],
                "source_image_sha256": sha256_file(source_image),
                "source_mask_sha256": sha256_file(source_mask),
                "image_sha256": sha256_file(image_path),
                "mask_sha256": sha256_file(mask_path),
            }
        )

    manifest_path = args.output_dir / "impactmesh_100_lockbox_manifest.csv"
    write_csv(manifest_path, rows)
    freeze = {
        "status": "FROZEN_BEFORE_EVALUATION",
        "dataset": "ibm-esa-geospatial/ImpactMesh-Fire",
        "dataset_revision": args.dataset_revision,
        "license": "CC-BY-4.0",
        "source_partition": "test_holdout",
        "selection": "deterministic event-balanced round-robin; no model outputs used",
        "seed": args.seed,
        "requested_samples": args.samples,
        "selected_samples": len(rows),
        "unique_events": len({str(row["event_group"]) for row in rows}),
        "positive_scenes": sum(row["fire_status"] == "positive" for row in rows),
        "negative_scenes": sum(row["fire_status"] == "negative" for row in rows),
        "holdout_list_sha256": sha256_file(holdout_path),
        "manifest_sha256": sha256_file(manifest_path),
        "model_selection_permitted": False,
        "evaluation_policy": "one-shot comparison of the already frozen candidate against the pre-existing baseline",
    }
    (args.output_dir / "LOCKBOX_FROZEN.json").write_text(json.dumps(freeze, indent=2), encoding="utf-8")
    print(json.dumps(freeze, indent=2))


def event_balanced_selection(sample_ids: list[str], count: int, seed: int) -> list[str]:
    if count > len(sample_ids):
        raise ValueError(f"Requested {count} samples from a holdout containing {len(sample_ids)}")
    grouped: dict[str, list[str]] = defaultdict(list)
    for sample_id in sample_ids:
        grouped[event_id(sample_id)].append(sample_id)
    for event, values in grouped.items():
        values.sort(key=lambda value: stable_key(seed, event, value))
    event_order = sorted(grouped, key=lambda event: stable_key(seed, event))

    selected: list[str] = []
    depth = 0
    while len(selected) < count:
        added = False
        for event in event_order:
            values = grouped[event]
            if depth < len(values):
                selected.append(values[depth])
                added = True
                if len(selected) == count:
                    return selected
        if not added:
            break
        depth += 1
    raise RuntimeError("Unable to select the requested number of samples")


def event_id(sample_id: str) -> str:
    parts = sample_id.split("_")
    if len(parts) < 2:
        raise ValueError(f"Unexpected ImpactMesh sample ID: {sample_id}")
    return "_".join(parts[:2])


def stable_key(seed: int, *values: str) -> str:
    return hashlib.sha256(":".join([str(seed), *values]).encode("utf-8")).hexdigest()


def normalize_rgb(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim == 3 and array.shape[0] == 3 and array.shape[-1] != 3:
        array = np.moveaxis(array, 0, -1)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"Expected H x W x 3 RGB data, received {array.shape}")
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype("uint8")
    return array


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty lockbox manifest")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
