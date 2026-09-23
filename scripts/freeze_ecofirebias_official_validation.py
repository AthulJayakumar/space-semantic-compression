"""Freeze one event-disjoint EcoFireBias validation-split generalization check."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "results/future_satellite_cohort_audit"
OUT = AUDIT / "ecofirebias_official_validation"
SEED = 20261004


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_event_pairs(rows: list[dict[str, str]], seed: int, events_per_continent: int) -> list[dict[str, str]]:
    by_event: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["split"] == "val":
            by_event[row["event_id"]].append(row)
    groups: dict[str, list[str]] = defaultdict(list)
    for event_id, pair in by_event.items():
        if len(pair) != 2 or {row["kind"] for row in pair} != {"burn", "neg"}:
            raise ValueError(f"Incomplete validation event pair: {event_id}")
        groups[pair[0]["continent"]].append(event_id)
    if len(groups) != 6 or any(len(group) < events_per_continent for group in groups.values()):
        raise ValueError("Insufficient balanced validation events")
    rng = np.random.default_rng(seed)
    chosen = {event_id for continent in sorted(groups) for event_id in rng.permutation(sorted(groups[continent]))[:events_per_continent]}
    return [row for row in rows if row["split"] == "val" and row["event_id"] in chosen]


def freeze() -> dict[str, object]:
    metadata_path = ROOT / "datasets/wildfire_global_v1/metadata.csv"
    with metadata_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = select_event_pairs(rows, SEED, 10)
    train_plan_path = AUDIT / "ecofirebias_training_expansion/TRAINING_PLAN_FROZEN.json"
    train_plan = json.loads(train_plan_path.read_text(encoding="utf-8"))
    development_gate = json.loads((AUDIT / "DEVELOPMENT_GATE_FROZEN.json").read_text(encoding="utf-8"))
    calibration = json.loads((AUDIT / "dnbr_encoding_train_only.json").read_text(encoding="utf-8"))
    by_sample = {row["example_id"]: row for row in rows}
    excluded = set(train_plan["event_ids"])
    excluded.update(development_gate["development_event_ids"])
    excluded.update(by_sample[row["example_id"]]["event_id"] for row in calibration["samples"])
    excluded.update(row["event_id"] for row in rows if row["split"] == "test")
    selected_events = {row["event_id"] for row in selected}
    if selected_events & excluded or len(selected) != 120 or len(selected_events) != 60:
        raise ValueError("Official-validation selection is not event-disjoint")
    train_countries = {by_sample[sample_id]["country"] for sample_id in train_plan["sample_ids"]}
    frozen = {
        "status": "FROZEN_BEFORE_OFFICIAL_VALIDATION_MODEL_EVALUATION",
        "dataset_revision": "39f331e50458fba3669837d69fc2fdddbb1d1d69",
        "metadata_sha256": sha256(metadata_path),
        "training_plan_sha256": sha256(train_plan_path),
        "development_gate_sha256": sha256(AUDIT / "DEVELOPMENT_GATE_FROZEN.json"),
        "seed": SEED,
        "selection": "10 paired validation events per continent sampled from sorted event IDs with a fixed seed",
        "sample_ids": [row["example_id"] for row in selected],
        "event_ids": sorted(selected_events),
        "images": len(selected),
        "events": len(selected_events),
        "continent_counts": dict(Counter(row["continent"] for row in selected)),
        "events_in_countries_seen_in_selected_training": len({row["event_id"] for row in selected if row["country"] in train_countries}),
        "native_image_size": [224, 224],
        "maximum_wire_bytes_per_image": 1200,
        "proxy_mask_rule": "quantized dNBR byte > 85; exclude nonoverlapping reprojected pixels",
        "comparison_methods": ["JPEG", "JPEG2000 RDO", "fixed-utility VQ-VAE base", "fixed-utility VQ-VAE adapted"],
        "candidate_checkpoint_sha256": {
            "base": sha256(ROOT / "models/checkpoints/vqvae_mixed_wire_finetuned.pt"),
            "adapted": sha256(ROOT / "models/checkpoints/vqvae_ecofirebias_train_adapted.pt"),
        },
        "analysis": "event-paired burn SUS and quantized-proxy Dice, negative false-positive area, actual serialized bytes; 10000 paired-event bootstrap draws with seed 20260922",
        "limitations": "Validation events are separate, but most countries overlap selected training; this is not a country-disjoint or sealed-test result",
        "sealed_test_scoring_authorized": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT / "VALIDATION_PLAN_FROZEN.json"
    if destination.exists():
        if json.loads(destination.read_text(encoding="utf-8")) != frozen:
            raise RuntimeError("Refusing to change an existing frozen validation plan")
    else:
        destination.write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")
    return {"plan": str(destination), "images": len(selected), "events": len(selected_events), "shared_country_events": frozen["events_in_countries_seen_in_selected_training"]}


def main() -> None:
    print(json.dumps(freeze(), indent=2))


if __name__ == "__main__":
    main()
