"""Build and audit a frozen FireScope external evaluation lockbox.

The selection code is deliberately separate from model training.  It pairs the
public FireScope event imagery with MTBS/EFFIS masks, draws a deterministic
continent- and class-balanced sample, and records enough provenance to repeat
the acquisition without using the lockbox for model selection.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Iterable

import numpy as np
from PIL import Image

from evaluation.dataset_validity import estimate_cloud_fraction, sha256_file


FIRESCOPE_REPO_ID = "INSAIT-Institute/FireScope-Bench"
LOCKBOX_SEED = "compressai-external-lockbox-v1"
DEFAULT_QUOTAS = {
    "europe_positive": 40,
    "usa_positive": 40,
    "europe_negative": 10,
    "usa_negative": 10,
}


@dataclass(frozen=True)
class FireScopeCandidate:
    """One source image/mask pair before download."""

    sample_id: str
    region: str
    fire_status: str
    geographic_group: str
    event_group: str
    source_image_path: str
    source_mask_path: str


def pair_firescope_files(paths: Iterable[str]) -> list[FireScopeCandidate]:
    """Pair event and control files using their source filename identity."""

    paths = sorted(set(paths))
    masks_by_scope: dict[tuple[str, str, str], str] = {}
    for path in paths:
        parsed = PurePosixPath(path)
        parts = parsed.parts
        if len(parts) < 4 or parts[0] not in {"europe", "usa"} or parts[1] != "wildfire_events":
            continue
        if parts[2] not in {"masks", "controls"} or parsed.suffix.lower() != ".npy":
            continue
        kind = "positive" if parts[2] == "masks" else "negative"
        masks_by_scope[(parts[0], kind, parsed.stem)] = path

    candidates: list[FireScopeCandidate] = []
    for path in paths:
        parsed = PurePosixPath(path)
        parts = parsed.parts
        if len(parts) < 4 or parts[0] not in {"europe", "usa"} or parts[1] != "wildfire_events":
            continue
        if parts[2] not in {"images", "controls_images"} or parsed.suffix.lower() != ".png":
            continue
        kind = "positive" if parts[2] == "images" else "negative"
        mask_path = masks_by_scope.get((parts[0], kind, parsed.stem))
        if mask_path is None:
            continue
        candidates.append(
            FireScopeCandidate(
                sample_id=f"firescope_{parts[0]}_{parsed.stem}",
                region=parts[0],
                fire_status=kind,
                geographic_group=_geographic_group(parts[0], parsed.stem),
                event_group=parsed.stem,
                source_image_path=path,
                source_mask_path=mask_path,
            )
        )
    return candidates


def select_lockbox_candidates(
    candidates: Iterable[FireScopeCandidate],
    quotas: dict[str, int] | None = None,
    seed: str = LOCKBOX_SEED,
) -> list[FireScopeCandidate]:
    """Select a deterministic, geographically dispersed lockbox."""

    quotas = quotas or DEFAULT_QUOTAS
    candidates = list(candidates)
    selected: list[FireScopeCandidate] = []
    for key, count in quotas.items():
        region, fire_status = key.rsplit("_", 1)
        eligible = [
            candidate
            for candidate in candidates
            if candidate.region == region and candidate.fire_status == fire_status
        ]
        chosen = _round_robin_groups(eligible, count, f"{seed}:{key}")
        if len(chosen) != count:
            raise ValueError(f"Requested {count} {key} scenes, but only {len(chosen)} paired scenes are available")
        selected.extend(chosen)
    return sorted(selected, key=lambda item: _stable_digest(seed, item.sample_id))


def download_and_prepare_lockbox(
    selected: Iterable[FireScopeCandidate],
    data_root: Path,
    revision: str | None = None,
) -> list[dict[str, object]]:
    """Download selected pairs and convert source NumPy masks to lossless PNG."""

    rows: list[dict[str, object]] = []
    converted_mask_root = data_root / "converted_masks"
    converted_mask_root.mkdir(parents=True, exist_ok=True)
    for index, candidate in enumerate(selected, start=1):
        print(f"lockbox download {index}: {candidate.sample_id}", flush=True)
        image_path = data_root / PurePosixPath(candidate.source_image_path)
        source_mask_path = data_root / PurePosixPath(candidate.source_mask_path)
        if not image_path.exists() or not source_mask_path.exists():
            raise FileNotFoundError(
                f"Bulk acquisition did not provide both files for {candidate.sample_id}: "
                f"image={image_path.exists()}, mask={source_mask_path.exists()}"
            )
        mask_array = _load_mask(source_mask_path)
        is_positive = bool((mask_array > 0).any())
        expected_positive = candidate.fire_status == "positive"
        if is_positive != expected_positive:
            raise ValueError(
                f"Label integrity failure for {candidate.sample_id}: "
                f"expected {candidate.fire_status}, observed {'positive' if is_positive else 'negative'} mask"
            )
        mask_path = converted_mask_root / f"{candidate.sample_id}.png"
        Image.fromarray(np.where(mask_array > 0, 255, 0).astype("uint8"), mode="L").save(mask_path)
        with Image.open(image_path) as source_image:
            image = source_image.convert("RGB")
            width, height = image.size
            cloud_fraction = estimate_cloud_fraction(image)
        source_year = _source_year(candidate.source_image_path)
        label_source = "EFFIS Burnt Areas" if candidate.region == "europe" else "MTBS burn severity"
        if candidate.fire_status == "negative":
            label_source = f"FireScope zero-mask control ({label_source} domain)"
        rows.append(
            {
                "sample_id": candidate.sample_id,
                "dataset": "firescope_bench",
                "image_path": str(image_path.resolve()),
                "mask_path": str(mask_path.resolve()),
                "split": "lockbox",
                "source_split": "external_lockbox",
                "geographic_group": candidate.geographic_group,
                "event_group": candidate.event_group,
                "label_source": label_source,
                "image_unit": "independent_firescope_sentinel2_event_tile",
                "fire_status": candidate.fire_status,
                "positive_pixel_fraction": float((mask_array > 0).mean()),
                "cloud_fraction": cloud_fraction,
                "cloud_case": cloud_fraction >= 0.10,
                "smoke_label_status": "unavailable",
                "width": width,
                "height": height,
                "source_year": source_year,
                "source_image_path": candidate.source_image_path,
                "source_mask_path": candidate.source_mask_path,
                "image_sha256": sha256_file(image_path),
                "source_mask_sha256": sha256_file(source_mask_path),
                "mask_sha256": sha256_file(mask_path),
            }
        )
    return rows


def acquire_firescope_selection(
    selected: Iterable[FireScopeCandidate],
    data_root: Path,
    revision: str,
    max_workers: int = 2,
) -> Path:
    """Acquire all selected files in one resumable, revision-pinned snapshot."""

    # Anonymous Xet token refreshes are rate-limited quickly when many small
    # files are requested.  Standard HTTP is slower but much more reliable for
    # this intentionally small 200-file subset.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("Install huggingface_hub to acquire the FireScope lockbox") from exc

    selected = list(selected)
    allow_patterns = sorted(
        {
            path
            for candidate in selected
            for path in (candidate.source_image_path, candidate.source_mask_path)
        }
    )
    data_root.mkdir(parents=True, exist_ok=True)
    resolved = snapshot_download(
        repo_id=FIRESCOPE_REPO_ID,
        repo_type="dataset",
        revision=revision,
        local_dir=data_root,
        allow_patterns=allow_patterns,
        max_workers=max_workers,
    )
    return Path(resolved)


def audit_and_freeze_lockbox(
    rows: list[dict[str, object]],
    manifest_path: Path,
    freeze_path: Path,
    source_revision: str,
    repository_root: Path,
    checkpoint_paths: Iterable[Path],
    expected_count: int = 100,
) -> dict[str, object]:
    """Write the immutable evaluation manifest and its provenance record."""

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    prior_hashes = _existing_manifest_hashes(repository_root, exclude=manifest_path)
    image_hashes = [str(row["image_sha256"]) for row in rows]
    mask_hashes = [str(row["mask_sha256"]) for row in rows]
    positive_mask_hashes = [
        str(row["mask_sha256"]) for row in rows if row["fire_status"] == "positive"
    ]
    sample_ids = [str(row["sample_id"]) for row in rows]
    source_paths = [str(row["source_image_path"]) for row in rows]
    overlap = sorted(set(image_hashes) & prior_hashes)
    checks = {
        "expected_scene_count": len(rows) == expected_count,
        "unique_sample_ids": len(sample_ids) == len(set(sample_ids)),
        "unique_source_paths": len(source_paths) == len(set(source_paths)),
        "unique_image_hashes": len(image_hashes) == len(set(image_hashes)),
        "unique_positive_mask_hashes": len(positive_mask_hashes) == len(set(positive_mask_hashes)),
        "negative_masks_are_empty": all(
            float(row["positive_pixel_fraction"]) == 0.0
            for row in rows
            if row["fire_status"] == "negative"
        ),
        "no_hash_overlap_with_existing_manifests": not overlap,
        "all_files_exist": all(Path(str(row["image_path"])).exists() and Path(str(row["mask_path"])).exists() for row in rows),
        "both_continents_present": {str(row["geographic_group"]).split(":", 1)[0] for row in rows} == {"europe", "usa"},
        "positive_and_negative_present": {str(row["fire_status"]) for row in rows} == {"positive", "negative"},
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"Lockbox audit failed: {', '.join(failed)}")

    _write_csv(manifest_path, rows)
    checkpoints = {
        str(path.as_posix()): sha256_file(path)
        for path in checkpoint_paths
        if path.exists()
    }
    freeze = {
        "status": "FROZEN_BEFORE_EVALUATION",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_repository": FIRESCOPE_REPO_ID,
        "source_revision": source_revision,
        "selection_seed": LOCKBOX_SEED,
        "selection_quotas": DEFAULT_QUOTAS,
        "scene_count": len(rows),
        "continent_counts": _counts(rows, lambda row: str(row["geographic_group"]).split(":", 1)[0]),
        "class_counts": _counts(rows, lambda row: str(row["fire_status"])),
        "geographic_group_count": len({str(row["geographic_group"]) for row in rows}),
        "event_group_count": len({str(row["event_group"]) for row in rows}),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "checkpoint_sha256": checkpoints,
        "audit_checks": checks,
        "prior_manifest_image_hash_overlap": overlap,
        "distinct_mask_hashes": len(set(mask_hashes)),
        "protocol": [
            "No model fitting, threshold calibration, selector tuning, or architecture selection may use this lockbox.",
            "Run the preregistered detector and matched-rate settings once, then report all exclusions.",
            "Any later model change requires a different external lockbox or must label this set as previously observed.",
        ],
        "limitations": [
            "FireScope event imagery is pre-event imagery associated with next-year fire masks; it tests cross-dataset mission relevance, not active-flame detection.",
            "Independent smoke labels are unavailable.",
        ],
    }
    freeze_path.parent.mkdir(parents=True, exist_ok=True)
    freeze_path.write_text(json.dumps(freeze, indent=2), encoding="utf-8")
    return freeze


def _round_robin_groups(
    candidates: list[FireScopeCandidate], count: int, seed: str
) -> list[FireScopeCandidate]:
    groups: dict[str, list[FireScopeCandidate]] = {}
    for candidate in candidates:
        groups.setdefault(candidate.geographic_group, []).append(candidate)
    for group, members in groups.items():
        groups[group] = sorted(members, key=lambda item: _stable_digest(seed, item.sample_id))
    group_order = sorted(groups, key=lambda group: _stable_digest(seed, group))
    selected: list[FireScopeCandidate] = []
    offset = 0
    while len(selected) < count:
        added = False
        for group in group_order:
            members = groups[group]
            if offset < len(members):
                selected.append(members[offset])
                added = True
                if len(selected) == count:
                    break
        if not added:
            break
        offset += 1
    return selected


def _geographic_group(region: str, stem: str) -> str:
    parts = stem.split("_")
    if len(parts) >= 2 and parts[1] not in {"NN", "negative"}:
        return f"{region}:{parts[1]}"
    lon_match = re.search(r"lon(-?\d+(?:\.\d+)?)", stem)
    lat_match = re.search(r"lat(-?\d+(?:\.\d+)?)", stem)
    if lon_match and lat_match:
        lon_bin = int(float(lon_match.group(1)) // 5 * 5)
        lat_bin = int(float(lat_match.group(1)) // 5 * 5)
        return f"{region}:grid_{lon_bin}_{lat_bin}"
    return f"{region}:unknown"


def _source_year(path: str) -> str:
    parts = PurePosixPath(path).parts
    return parts[3] if len(parts) > 4 and parts[3].isdigit() else "unavailable"


def _load_mask(path: Path) -> np.ndarray:
    mask = np.asarray(np.load(path, allow_pickle=False))
    mask = np.squeeze(mask)
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D FireScope mask, got shape {mask.shape} from {path}")
    if not np.isfinite(mask).all():
        raise ValueError(f"Mask contains non-finite values: {path}")
    return mask


def _stable_digest(seed: str, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def _existing_manifest_hashes(root: Path, exclude: Path) -> set[str]:
    hashes: set[str] = set()
    excluded = exclude.resolve()
    for path in root.rglob("*.csv"):
        if path.resolve() == excluded:
            continue
        try:
            with path.open("r", newline="", encoding="utf-8-sig") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames or "image_sha256" not in reader.fieldnames:
                    continue
                hashes.update(row["image_sha256"].strip() for row in reader if row.get("image_sha256", "").strip())
        except (OSError, UnicodeError, csv.Error):
            continue
    return hashes


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty lockbox manifest")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _counts(rows: list[dict[str, object]], key) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = key(row)
        counts[value] = counts.get(value, 0) + 1
    return counts
