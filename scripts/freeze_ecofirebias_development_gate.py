"""Freeze the training-split development screen before model evaluation."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.verify_ecofirebias_dnbr_encoding import select_train_pairs


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/future_satellite_cohort_audit/DEVELOPMENT_GATE_FROZEN.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze() -> dict[str, object]:
    metadata_path = ROOT / "datasets/wildfire_global_v1/metadata.csv"
    encoding_path = ROOT / "results/future_satellite_cohort_audit/dnbr_encoding_train_only.json"
    with metadata_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    calibration_ids = {row["example_id"] for row in select_train_pairs(rows, 3)}
    development = [row for row in select_train_pairs(rows, 6) if row["example_id"] not in calibration_ids]
    if len(development) != 36 or len({row["event_id"] for row in development}) != 18:
        raise ValueError("Expected 18 complete development event pairs")
    frozen = {
        "status": "FROZEN_BEFORE_DEVELOPMENT_MODEL_EVALUATION",
        "source_split": "train",
        "dataset_revision": "39f331e50458fba3669837d69fc2fdddbb1d1d69",
        "metadata_sha256": sha256(metadata_path),
        "encoding_training_evidence_sha256": sha256(encoding_path),
        "selection": "event IDs sorted within each continent; take events 4-6, excluding events 1-3 used for label encoding verification",
        "development_ids": [row["example_id"] for row in development],
        "development_event_ids": sorted({row["event_id"] for row in development}),
        "image_size": [224, 224],
        "maximum_wire_bytes_per_image": 1200,
        "wire_counting": "include all transmitted token IDs, coordinates, headers and other codec payload; no hidden side information",
        "reference_codecs": ["JPEG", "JPEG2000 RDO"],
        "primary_reference": "JPEG2000 RDO",
        "mask_proxy": "dNBR GeoTIFF byte > 85, fixed using separate train-split events; not a human burn-scar mask",
        "gate": {
            "burn_positive_sus_mean_difference_at_least": 3.0,
            "burn_positive_proxy_dice_mean_difference_at_least": 0.02,
            "paired_event_bootstrap_95_percent_lower_bound_for_both_differences_above": 0.0,
            "negative_pair_false_positive_area_mean_increase_at_most": 0.02,
            "bootstrap_seed": 20260922,
            "bootstrap_resamples": 10000,
        },
        "advance_if_all_gates_pass": True,
        "sealed_cohort": "37 eligible EcoFireBias test-split event pairs; no model scoring before development gate passes",
    }
    if OUT.exists():
        previous = json.loads(OUT.read_text(encoding="utf-8"))
        if previous != frozen:
            raise RuntimeError("Existing development gate differs; refusal to silently change it")
    else:
        OUT.write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")
    return {"output": str(OUT), "images": len(development), "events": 18, "byte_ceiling": 1200}


def main() -> None:
    print(json.dumps(freeze(), indent=2))


if __name__ == "__main__":
    main()
