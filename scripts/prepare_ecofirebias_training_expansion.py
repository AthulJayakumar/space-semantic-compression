"""Freeze and fetch event-disjoint EcoFireBias TRAIN imagery for VQ fine-tuning."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "datasets/wildfire_global_v1"
OUT = ROOT / "results/future_satellite_cohort_audit/ecofirebias_training_expansion"
REVISION = "39f331e50458fba3669837d69fc2fdddbb1d1d69"
SEED = 20261003


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metadata_rows() -> list[dict[str, str]]:
    with (SOURCE / "metadata.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def select_training_rows(
    rows: list[dict[str, str]], excluded_events: set[str], seed: int, events_per_continent: int = 100
) -> list[dict[str, str]]:
    by_event: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_event[row["event_id"]].append(row)
    available: dict[str, list[str]] = defaultdict(list)
    for event_id, pair in by_event.items():
        if event_id in excluded_events or len(pair) != 2 or {row["kind"] for row in pair} != {"burn", "neg"}:
            continue
        if {row["split"] for row in pair} != {"train"}:
            continue
        available[pair[0]["continent"]].append(event_id)
    if len(available) != 6 or any(len(events) < events_per_continent for events in available.values()):
        raise ValueError("Insufficient clean training events for continent-balanced selection")
    rng = np.random.default_rng(seed)
    selected = []
    for continent in sorted(available):
        event_ids = rng.permutation(sorted(available[continent]))[:events_per_continent]
        for index, event_id in enumerate(event_ids):
            kind = "burn" if index < events_per_continent // 2 else "neg"
            selected.append(next(row for row in by_event[event_id] if row["kind"] == kind))
    if len({row["event_id"] for row in selected}) != len(selected):
        raise ValueError("Training selection must have one image per event")
    return selected


def freeze() -> dict[str, object]:
    rows = metadata_rows()
    by_sample = {row["example_id"]: row for row in rows}
    gate = json.loads((ROOT / "results/future_satellite_cohort_audit/DEVELOPMENT_GATE_FROZEN.json").read_text(encoding="utf-8"))
    calibration = json.loads((ROOT / "results/future_satellite_cohort_audit/dnbr_encoding_train_only.json").read_text(encoding="utf-8"))
    calibration_events = {by_sample[row["example_id"]]["event_id"] for row in calibration["samples"]}
    excluded_events = set(gate["development_event_ids"]) | calibration_events
    all_test_events = {row["event_id"] for row in rows if row["split"] == "test"}
    excluded_events.update(all_test_events)
    selected = select_training_rows(rows, excluded_events, SEED)
    if len(selected) != 600 or len({row["event_id"] for row in selected}) != 600:
        raise ValueError("Training selection must have one image per event")
    base = ROOT / "models/checkpoints/vqvae_mixed_wire_finetuned.pt"
    frozen = {
        "status": "FROZEN_BEFORE_TRAINING",
        "dataset_revision": REVISION,
        "metadata_sha256": sha256(SOURCE / "metadata.csv"),
        "base_checkpoint": str(base),
        "base_checkpoint_sha256": sha256(base),
        "source_split": "train",
        "excluded_gate_events": len(gate["development_event_ids"]),
        "excluded_label_calibration_events": len(calibration_events),
        "excluded_all_test_events": len(all_test_events),
        "selection": "seeded 100 events per continent; first 50 burn chips and next 50 negative chips; one image per event",
        "seed": SEED,
        "sample_ids": [row["example_id"] for row in selected],
        "event_ids": [row["event_id"] for row in selected],
        "trainer": "scripts/finetune_vqvae_satellite.py",
        "trainer_arguments": {
            "epochs": 2,
            "batch": 4,
            "image_size": 224,
            "learning_rate": 1e-5,
            "validation_fraction": 0.2,
            "seed": 1234,
            "teacher_consistency_weight": 0.1,
            "semantic_consistency_weight": 0.1,
            "pruned_training_weight": 0.5,
            "pruned_retention": 0.7,
            "wire_fallback": True,
            "semantic_detector_checkpoint": "models/checkpoints/wildfire_utility_segmentation_retrained.pt",
            "freeze_codebook": True,
        },
        "checkpoint_output": "models/checkpoints/vqvae_ecofirebias_train_adapted.pt",
        "interpretation": "Image-only train split adaptation, no new architecture; internal validation and any reused development gate are exploratory",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT / "TRAINING_PLAN_FROZEN.json"
    if destination.exists():
        if json.loads(destination.read_text(encoding="utf-8")) != frozen:
            raise RuntimeError("Refusing to change the frozen training plan")
    else:
        destination.write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")
    return frozen


def download(frozen: dict[str, object]) -> dict[str, object]:
    rows = {row["example_id"]: row for row in metadata_rows()}
    remote = HfApi().list_repo_files("moritzrengert1/wildfire_global", repo_type="dataset", revision=REVISION)
    paths = {Path(path).stem: path for path in remote if path.startswith("images/post/")}
    missing = set(frozen["sample_ids"]) - set(paths)
    if missing:
        raise FileNotFoundError(f"Missing selected images in pinned revision: {len(missing)}")
    manifest = []
    asset_hashes = {}
    for index, sample_id in enumerate(frozen["sample_ids"], start=1):
        row = rows[sample_id]
        local = Path(hf_hub_download(
            repo_id="moritzrengert1/wildfire_global", filename=paths[sample_id],
            repo_type="dataset", revision=REVISION, local_dir=SOURCE,
        ))
        with Image.open(local) as image:
            if image.size != (224, 224) or image.mode != "RGB":
                raise ValueError(f"Unexpected training RGB layout: {local}")
        asset_hashes[sample_id] = sha256(local)
        manifest.append({
            "dataset": "ecofirebias_train_adaptation",
            "image_path": str(local),
            "mask_path": "",
            "split": "train",
            "width": 224,
            "height": 224,
            "source_id": sample_id,
            "event_group": row["event_id"],
            "continent": row["continent"],
            "kind": row["kind"],
        })
        if index % 50 == 0 or index == len(frozen["sample_ids"]):
            print(f"validated training RGB {index}/{len(frozen['sample_ids'])}", flush=True)
    destination = OUT / "training_manifest.csv"
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    indices = np.arange(len(manifest))
    np.random.default_rng(frozen["trainer_arguments"]["seed"]).shuffle(indices)
    val_count = round(len(manifest) * frozen["trainer_arguments"]["validation_fraction"])
    val_events = {manifest[int(index)]["event_group"] for index in indices[:val_count]}
    train_events = {manifest[int(index)]["event_group"] for index in indices[val_count:]}
    if train_events & val_events or len(train_events) != 480 or len(val_events) != 120:
        raise ValueError("Replayed trainer split is not event-disjoint")
    result = {
        "training_plan_sha256": sha256(OUT / "TRAINING_PLAN_FROZEN.json"),
        "training_manifest_sha256": sha256(destination),
        "samples": len(manifest),
        "train_events": len(train_events),
        "internal_validation_events": len(val_events),
        "train_validation_event_overlap": 0,
        "kind_counts": dict(Counter(row["kind"] for row in manifest)),
        "continent_counts": dict(Counter(row["continent"] for row in manifest)),
        "asset_sha256": asset_hashes,
    }
    (OUT / "TRAINING_DATA_AUDIT.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return {key: value for key, value in result.items() if key != "asset_sha256"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    frozen = freeze()
    if args.download:
        print(json.dumps(download(frozen), indent=2))
    else:
        print(json.dumps({"plan": str(OUT / "TRAINING_PLAN_FROZEN.json"), "samples": len(frozen["sample_ids"])}, indent=2))


if __name__ == "__main__":
    main()
