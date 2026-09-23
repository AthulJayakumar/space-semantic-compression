"""Validate the byte-valued EcoFireBias dNBR proxy on train-split labels only."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
import numpy as np
import tifffile


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "datasets/wildfire_global_v1"
OUT = ROOT / "results/future_satellite_cohort_audit/dnbr_encoding_train_only.json"
REVISION = "39f331e50458fba3669837d69fc2fdddbb1d1d69"
TARGETS = (("010", 0.10), ("027", 0.27), ("044", 0.44))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_train_pairs(rows: list[dict[str, str]], pairs_per_continent: int = 3) -> list[dict[str, str]]:
    events: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["split"] == "train":
            events[row["event_id"]].append(row)
    continents: dict[str, list[str]] = defaultdict(list)
    for event_id, pair in events.items():
        if len(pair) == 2 and {row["kind"] for row in pair} == {"burn", "neg"}:
            continents[pair[0]["continent"]].append(event_id)
    chosen = {event_id for continent in sorted(continents) for event_id in sorted(continents[continent])[:pairs_per_continent]}
    return [row for row in rows if row["event_id"] in chosen and row["split"] == "train"]


def best_byte_threshold(arrays: list[np.ndarray], expected: np.ndarray) -> dict[str, object]:
    errors = np.zeros(256, dtype=float)
    for threshold in range(256):
        predicted = np.array([(array > threshold).mean() for array in arrays])
        errors[threshold] = np.mean(np.abs(predicted - expected))
    threshold = int(np.argmin(errors))
    predicted = np.array([(array > threshold).mean() for array in arrays])
    return {
        "pixel_rule": f"byte > {threshold}",
        "mean_absolute_fraction_error": float(errors[threshold]),
        "maximum_absolute_fraction_error": float(np.max(np.abs(predicted - expected))),
        "per_image_absolute_errors": np.abs(predicted - expected).round(5).tolist(),
    }


def verify() -> dict[str, object]:
    metadata_path = SOURCE / "metadata.csv"
    with metadata_path.open(newline="", encoding="utf-8") as handle:
        selected = select_train_pairs(list(csv.DictReader(handle)))
    remote = HfApi().list_repo_files("moritzrengert1/wildfire_global", repo_type="dataset", revision=REVISION)
    paths = {Path(path).stem: path for path in remote if path.startswith("geotiff/dnbr/")}
    arrays = []
    evidence = []
    for index, row in enumerate(selected, start=1):
        sample_id = row["example_id"]
        if sample_id not in paths:
            raise FileNotFoundError(f"No pinned dNBR file for training sample {sample_id}")
        local = Path(hf_hub_download(
            repo_id="moritzrengert1/wildfire_global", filename=paths[sample_id],
            repo_type="dataset", revision=REVISION, local_dir=SOURCE,
        ))
        array = tifffile.imread(local)
        if array.shape != (224, 224) or array.dtype != np.uint8:
            raise ValueError(f"Unexpected training dNBR file: {local} {array.shape} {array.dtype}")
        arrays.append(array)
        evidence.append({
            "example_id": sample_id,
            "continent": row["continent"],
            "kind": row["kind"],
            "path": str(local.relative_to(SOURCE)),
            "sha256": sha256(local),
            "byte_mean": float(array.mean()),
            "metadata_dnbr_mean": float(row["dnbr_mean"]),
            "expected_fractions": {key: float(row[f"burn_pixel_fraction_{key}"]) for key, _ in TARGETS},
        })
        print(f"validated training label {index}/{len(selected)}", flush=True)
    thresholds = {}
    for key, value in TARGETS:
        expected = np.array([row["expected_fractions"][key] for row in evidence])
        thresholds[key] = {"physical_dnbr_threshold": value, **best_byte_threshold(arrays, expected)}
    means = np.array([row["byte_mean"] for row in evidence])
    original_means = np.array([row["metadata_dnbr_mean"] for row in evidence])
    fit = np.polyfit(means, original_means, 1)
    result = {
        "status": "train_label_encoding_validation_only",
        "dataset_revision": REVISION,
        "metadata_sha256": sha256(metadata_path),
        "model_outputs_or_test_labels_accessed": False,
        "training_examples": len(evidence),
        "training_events": len({row["event_id"] for row in selected}),
        "mean_affine_fit_physical_per_byte": fit.tolist(),
        "mean_affine_fit_max_residual": float(np.max(np.abs(np.polyval(fit, means) - original_means))),
        "thresholds": thresholds,
        "samples": evidence,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    result = verify()
    print(json.dumps({key: value for key, value in result.items() if key != "samples"}, indent=2))


if __name__ == "__main__":
    main()
