"""Provenance, leakage, and label audits for satellite benchmark datasets."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import tarfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from datasets.research_wildfire import load_grayscale_image, load_rgb_image


HLS_TILE_PATTERN = re.compile(r"HLS\.[LS]30\.T(?P<tile>[0-9A-Z]{5})\.(?P<date>\d{7})", re.IGNORECASE)


@dataclass(frozen=True)
class ValidatedScene:
    sample_id: str
    dataset: str
    image_path: str
    mask_path: str
    split: str
    source_split: str
    geographic_group: str
    event_group: str
    label_source: str
    image_unit: str
    fire_status: str
    positive_pixel_fraction: float
    cloud_fraction: float
    cloud_case: bool
    smoke_label_status: str
    width: int
    height: int
    image_sha256: str
    mask_sha256: str


def extract_hls_pairs(archive_path: Path, output_root: Path, limit: int = 500) -> list[tuple[Path, Path, str]]:
    """Extract a deterministic scene-level subset from the official HLS archive."""

    output_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, mode="r:gz") as archive:
        members = [member for member in archive.getmembers() if member.isfile() and member.name.lower().endswith((".tif", ".tiff"))]
        images: dict[str, tarfile.TarInfo] = {}
        masks: dict[str, tarfile.TarInfo] = {}
        source_splits: dict[str, str] = {}
        for member in members:
            path = Path(member.name)
            stem = path.stem.lower()
            split = _source_split(path)
            key = _hls_pair_key(path)
            if "mask" in stem:
                masks.setdefault(key, member)
            else:
                images.setdefault(key, member)
                source_splits.setdefault(key, split)

        pairs: dict[str, tuple[tarfile.TarInfo, tarfile.TarInfo, str]] = {}
        for key, image_member in images.items():
            mask_member = masks.get(key)
            if mask_member is not None:
                pairs.setdefault(key, (image_member, mask_member, source_splits.get(key, "unknown")))
        selected_keys = sorted(pairs, key=lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest())[:limit]
        selected = [pairs[key] for key in selected_keys]
        selected_members = sorted(
            {member.name: member for image_member, mask_member, _ in selected for member in (image_member, mask_member)}.values(),
            key=lambda member: member.offset,
        )
        extracted_paths: dict[str, Path] = {}
        for member in selected_members:
            extracted_paths[member.name] = _safe_extract_member(archive, member, output_root)
        extracted = [
            (extracted_paths[image_member.name], extracted_paths[mask_member.name], source_split)
            for image_member, mask_member, source_split in selected
        ]
    return extracted


def discover_extracted_hls_pairs(root: Path) -> list[tuple[Path, Path, str]]:
    """Discover already extracted image/mask pairs without reopening the archive."""

    images: dict[str, Path] = {}
    masks: dict[str, Path] = {}
    for path in sorted(root.rglob("*.tif")):
        key = _hls_pair_key(path)
        if "mask" in path.stem.lower():
            masks[key] = path
        else:
            images[key] = path
    return [(image, masks[key], _source_split(image)) for key, image in images.items() if key in masks]


def validate_hls_pairs(
    pairs: list[tuple[Path, Path, str]],
    dataset: str = "ibm_nasa_hls_burn_scars",
    label_source: str = "IBM/NASA HLS Burn Scars mask",
) -> tuple[list[ValidatedScene], dict[str, object]]:
    preliminary: list[dict[str, object]] = []
    for image_path, mask_path, source_split in pairs:
        match = HLS_TILE_PATTERN.search(image_path.name)
        geographic_group = match.group("tile").upper() if match else "unknown"
        event_group = "unavailable"
        image = load_rgb_image(image_path)
        mask = np.asarray(load_grayscale_image(mask_path), dtype="float32")
        positive_fraction = float((mask > 0).mean())
        cloud_fraction = estimate_cloud_fraction(image)
        preliminary.append(
            {
                "sample_id": image_path.stem,
                "dataset": dataset,
                "image_path": str(image_path.resolve()),
                "mask_path": str(mask_path.resolve()),
                "source_split": source_split,
                "geographic_group": geographic_group,
                "event_group": event_group,
                "label_source": label_source,
                "image_unit": "independent_georeferenced_512x512_tile",
                "fire_status": "positive" if positive_fraction > 0.0 else "negative",
                "positive_pixel_fraction": positive_fraction,
                "cloud_fraction": cloud_fraction,
                "cloud_case": cloud_fraction >= 0.10,
                "smoke_label_status": "unavailable",
                "width": image.width,
                "height": image.height,
                "image_sha256": sha256_file(image_path),
                "mask_sha256": sha256_file(mask_path),
            }
        )

    geography_groups = sorted({str(row["geographic_group"]) for row in preliminary if row["geographic_group"] != "unknown"})
    split_by_group = {group: deterministic_group_split(group) for group in geography_groups}
    rows = [
        ValidatedScene(
            **row,
            split=split_by_group.get(str(row["geographic_group"]), "excluded_unknown_geography"),
        )
        for row in preliminary
    ]
    audit = audit_scene_manifest(rows)
    return rows, audit


def deterministic_group_split(group: str) -> str:
    """Assign an entire geographic group to one reproducible partition."""

    value = int(hashlib.sha256(f"geo-split-v1:{group}".encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    if value < 0.70:
        return "train"
    if value < 0.85:
        return "validation"
    return "test"


def audit_scene_manifest(rows: list[ValidatedScene]) -> dict[str, object]:
    split_names = ("train", "validation", "test")
    source_sets = {split: {row.sample_id for row in rows if row.split == split} for split in split_names}
    hash_sets = {split: {row.image_sha256 for row in rows if row.split == split} for split in split_names}
    geography_sets = {split: {row.geographic_group for row in rows if row.split == split} for split in split_names}
    event_sets = {
        split: {row.event_group for row in rows if row.split == split and row.event_group != "unavailable"}
        for split in split_names
    }
    split_pairs = (("train", "validation"), ("train", "test"), ("validation", "test"))
    source_overlap = {f"{a}_vs_{b}": sorted(source_sets[a] & source_sets[b]) for a, b in split_pairs}
    hash_overlap = {f"{a}_vs_{b}": sorted(hash_sets[a] & hash_sets[b]) for a, b in split_pairs}
    geography_overlap = {f"{a}_vs_{b}": sorted(geography_sets[a] & geography_sets[b]) for a, b in split_pairs}
    event_overlap = {f"{a}_vs_{b}": sorted(event_sets[a] & event_sets[b]) for a, b in split_pairs}
    duplicate_hashes = len(rows) - len({row.image_sha256 for row in rows})
    return {
        "total_independent_tiles": len(rows),
        "full_sentinel_granules": 0,
        "image_unit": "independent georeferenced 512x512 HLS tiles; not full Sentinel granules and not extracted overlapping patches",
        "split_counts": {split: sum(row.split == split for row in rows) for split in (*split_names, "excluded_unknown_geography")},
        "geographic_groups": len({row.geographic_group for row in rows if row.geographic_group != "unknown"}),
        "event_groups": len({row.event_group for row in rows if row.event_group != "unavailable"}),
        "fire_positive": sum(row.fire_status == "positive" for row in rows),
        "fire_negative": sum(row.fire_status == "negative" for row in rows),
        "cloud_cases_proxy": sum(row.cloud_case for row in rows),
        "smoke_labelled_cases": 0,
        "smoke_label_note": "The source dataset has burn-scar masks but no independent smoke annotations; smoke-stratified claims are not made.",
        "independent_label_source": sorted({row.label_source for row in rows}),
        "duplicate_image_hashes": duplicate_hashes,
        "source_overlap": source_overlap,
        "hash_overlap": hash_overlap,
        "geography_overlap": geography_overlap,
        "event_overlap": event_overlap,
        "passes_no_source_leakage": not any(source_overlap.values()),
        "passes_no_exact_duplicate_leakage": not any(hash_overlap.values()) and duplicate_hashes == 0,
        "passes_geographic_separation": not any(geography_overlap.values()),
        "passes_event_separation": False,
        "event_separation_status": "not verifiable: the source metadata does not provide wildfire-event identifiers",
        "passes_minimum_500_tiles": len(rows) >= 500,
    }


def write_validity_outputs(rows: list[ValidatedScene], audit: dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "validated_scene_manifest.csv"
    fieldnames = list(ValidatedScene.__dataclass_fields__)
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    (output_dir / "dataset_integrity.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    split_rows = []
    for split in ("train", "validation", "test", "excluded_unknown_geography"):
        subset = [row for row in rows if row.split == split]
        split_rows.append(
            {
                "split": split,
                "n": len(subset),
                "geographic_groups": len({row.geographic_group for row in subset}),
                "event_groups": len({row.event_group for row in subset if row.event_group != "unavailable"}),
                "fire_positive": sum(row.fire_status == "positive" for row in subset),
                "fire_negative": sum(row.fire_status == "negative" for row in subset),
                "cloud_cases_proxy": sum(row.cloud_case for row in subset),
                "smoke_labelled_cases": 0,
            }
        )
    with (output_dir / "split_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(split_rows[0]))
        writer.writeheader()
        writer.writerows(split_rows)
    checks = [
        ("Minimum 500 independent tiles", audit["passes_minimum_500_tiles"]),
        ("No source identity leakage", audit["passes_no_source_leakage"]),
        ("No exact duplicate leakage", audit["passes_no_exact_duplicate_leakage"]),
        ("Geographic groups separated", audit["passes_geographic_separation"]),
        ("Wildfire-event groups separated", audit["passes_event_separation"]),
    ]
    lines = [
        "# Satellite Dataset Validity Report",
        "",
        "## Scope",
        "",
        f"This benchmark contains {audit['total_independent_tiles']} independent georeferenced 512x512 HLS satellite tiles. These are analysis-ready tiles, not full Sentinel-2 granules and not multiple patches extracted from the same source tile.",
        "",
        "## Integrity checks",
        "",
        "| Check | Result |",
        "| --- | --- |",
        *[f"| {name} | {'PASS' if passed else 'FAIL'} |" for name, passed in checks],
        "",
        "## Coverage",
        "",
        f"- Geographic groups: {audit['geographic_groups']}",
        f"- Fire-positive tiles: {audit['fire_positive']}",
        f"- Fire-negative tiles: {audit['fire_negative']}",
        f"- Cloud cases identified by a documented RGB proxy: {audit['cloud_cases_proxy']}",
        f"- Independent smoke annotations: {audit['smoke_labelled_cases']}",
        f"- Label source: {', '.join(audit['independent_label_source'])}",
        "",
        "## Important limitation",
        "",
        str(audit["smoke_label_note"]),
        str(audit["event_separation_status"]),
        "Cloud coverage is a reproducible RGB brightness/saturation proxy, not an official cloud mask. Results may be stratified by this proxy, but it must not be described as ground-truth cloud labelling.",
    ]
    (output_dir / "dataset_validity_report.md").write_text("\n".join(lines), encoding="utf-8")


def estimate_cloud_fraction(image) -> float:
    rgb = np.asarray(image.convert("RGB"), dtype="float32") / 255.0
    brightness = rgb.mean(axis=2)
    saturation = rgb.max(axis=2) - rgb.min(axis=2)
    return float(((brightness >= 0.72) & (saturation <= 0.18)).mean())


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_extract_member(archive: tarfile.TarFile, member: tarfile.TarInfo, output_root: Path) -> Path:
    target = (output_root / member.name).resolve()
    root = output_root.resolve()
    if root != target and root not in target.parents:
        raise ValueError(f"Unsafe archive member: {member.name}")
    target.parent.mkdir(parents=True, exist_ok=True)
    source = archive.extractfile(member)
    if source is None:
        raise ValueError(f"Could not read archive member: {member.name}")
    with source, target.open("wb") as handle:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    return target


def _source_split(path: Path) -> str:
    parts = {part.lower() for part in path.parts}
    if parts & {"validation", "valid", "val"}:
        return "validation"
    if "test" in parts:
        return "test"
    if parts & {"training", "train"}:
        return "train"
    return "unknown"


def _hls_pair_key(path: Path) -> str:
    stem = path.stem.lower()
    stem = re.sub(r"(?:[._-](?:merged|mask))+$", "", stem)
    return stem
