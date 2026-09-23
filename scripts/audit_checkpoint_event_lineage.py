"""Reconstruct recorded CEMS exposure in the current VQ-VAE checkpoint chain.

This audits input lineage only. It never loads image pixels or evaluates a model.
Replayed splits are conditional on the current manifests matching training time:
the checkpoints record paths and seeds, but not source-manifest hashes.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/future_satellite_cohort_audit"
CHECKPOINTS = (
    "vqvae_s16k8_mixed_regularized_finetuned.pt",
    "vqvae_s16k8_semantic_w025_finetuned.pt",
    "vqvae_s16k8_pruned_semantic_finetuned.pt",
    "vqvae_mixed_wire_finetuned.pt",
    "vqvae_mixed_wire_mask_aware.pt",
)
EVENT_PATTERN = re.compile(r"EMSR\d+", re.IGNORECASE)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def training_indices(count: int, fraction: float, seed: int) -> set[int]:
    if count == 0:
        return set()
    indices = np.arange(count)
    np.random.default_rng(seed).shuffle(indices)
    validation_count = max(1, round(count * float(np.clip(fraction, 0.05, 0.8))))
    return set(indices[validation_count:].tolist())


def path_in_root(root: Path, raw: str) -> Path:
    return root / Path(raw.replace("\\", "/"))


def patch_events(root: Path, image_dir: str, limit: int | None, fraction: float, seed: int) -> tuple[set[str], int, str]:
    directory = path_in_root(root, image_dir)
    manifest = directory.parent / "patch_manifest.json" if directory.name == "images" else directory / "patch_manifest.json"
    items = json.loads(manifest.read_text(encoding="utf-8"))["items"]
    sources = {Path(item["target"].replace("\\", "/")).name: item["source"] for item in items}
    paths = sorted(path for path in directory.rglob("*") if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"})
    paths = paths[:limit] if limit else paths
    selected = training_indices(len(paths), fraction, seed)
    events = set()
    for index in selected:
        source = sources.get(paths[index].name)
        if source is None:
            raise ValueError(f"Missing patch provenance for {paths[index]}")
        match = EVENT_PATTERN.search(source)
        if match:
            events.add(match.group(0).upper())
    return events, len(selected), sha256(manifest)


def audit(root: Path) -> dict[str, object]:
    georef = read_csv(root / "results/future_satellite_cohort_audit/cems_georeference.csv")
    by_id = {row["sample_id"]: row["event_group"] for row in georef}
    impactmesh = read_csv(root / "results/external_lockbox_impactmesh_100/impactmesh_100_lockbox_manifest.csv")
    impact_codes = [row["event_group"].split("_", 1)[0] for row in impactmesh]
    stage_results: list[dict[str, object]] = []
    inherited_training: set[str] = set()
    opaque_ancestor = False

    for name in CHECKPOINTS:
        checkpoint = root / "models/checkpoints" / name
        info = torch.load(checkpoint, map_location="cpu", weights_only=True).get("fine_tuning", {})
        manifest_raw = info.get("manifest")
        direct_events: set[str] = set()
        extra_events: set[str] = set()
        direct_count = 0
        extra_count = 0
        input_hashes: dict[str, str] = {}
        limitations: list[str] = []
        if manifest_raw:
            manifest = path_in_root(root, manifest_raw)
            rows = read_csv(manifest)
            expected = int(info["train_samples"]) + int(info["validation_samples"])
            if len(rows) != expected:
                opaque_ancestor = True
                limitations.append(f"Current manifest has {len(rows)} rows, but checkpoint records {expected}; split cannot be replayed")
            else:
                input_hashes["manifest_current_sha256"] = sha256(manifest)
                indices = training_indices(len(rows), info["validation_fraction"], info["seed"])
                direct_count = len(indices)
                if direct_count != info["train_samples"]:
                    raise ValueError(f"Replayed direct split count differs for {name}")
                for index in indices:
                    row = rows[index]
                    sample_id = row.get("source_id") or row.get("sample_id", "")
                    if sample_id in by_id:
                        direct_events.add(by_id[sample_id])
                    else:
                        match = EVENT_PATTERN.search(row.get("image_path", ""))
                        if match:
                            direct_events.add(match.group(0).upper())
        else:
            limitations.append("This stage does not record a row-level manifest")

        for index, image_dir in enumerate(info.get("extra_image_dirs", [])):
            events, count, digest = patch_events(
                root, image_dir, info.get("extra_limit"), info["validation_fraction"], info["seed"] + index + 1
            )
            extra_events.update(events)
            extra_count += count
            input_hashes[f"patch_manifest_{index}_current_sha256"] = digest
        if extra_count != info.get("extra_train_samples", 0):
            raise ValueError(f"Replayed patch split count differs for {name}")

        inherited_training.update(direct_events | extra_events)
        shared = inherited_training & set(impact_codes)
        stage_results.append({
            "checkpoint": name,
            "recorded_base": info.get("base_checkpoint"),
            "replayed_direct_train_images": direct_count,
            "replayed_patch_train_images": extra_count,
            "direct_cems_event_count": len(direct_events),
            "patch_cems_event_count": len(extra_events),
            "known_inherited_cems_event_count": len(inherited_training),
            "known_inherited_impactmesh_overlap_events": sorted(shared),
            "known_inherited_impactmesh_overlap_tiles": sum(code in shared for code in impact_codes),
            "input_hashes": input_hashes,
            "limitations": limitations,
        })

    return {
        "status": "conditional_replay_not_proof_of_complete_lineage",
        "image_pixels_or_model_outputs_accessed": False,
        "source_manifest_hashes_recorded_at_training": False,
        "incomplete_ancestor_lineage": opaque_ancestor,
        "impactmesh_evaluated_tiles": len(impactmesh),
        "stages": stage_results,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    result = audit(ROOT)
    destination = OUT / "checkpoint_lineage.json"
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(destination), "stages": result["stages"][-1]}, indent=2))


if __name__ == "__main__":
    main()
