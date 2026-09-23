"""Prepare a continent-balanced EcoFireBias external lockbox."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("datasets/wildfire_global_v1"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/external_lockbox_ecofirebias_120"),
    )
    parser.add_argument("--events-per-continent", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--exclude-manifest", type=Path, default=None)
    parser.add_argument("--freeze-selection-only", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--dataset-revision", default="39f331e50458fba3669837d69fc2fdddbb1d1d69")
    args = parser.parse_args()

    metadata_path = args.source_root / "metadata.csv"
    with metadata_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    excluded_events: set[str] = set()
    if args.exclude_manifest is not None:
        with args.exclude_manifest.open("r", newline="", encoding="utf-8") as handle:
            excluded_events = {row["event_group"] for row in csv.DictReader(handle)}
    selected = balanced_event_pairs(rows, args.events_per_continent, args.seed, excluded_events)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selection = {
        "status": "FROZEN_BEFORE_SELECTOR_DEVELOPMENT",
        "dataset_revision": args.dataset_revision,
        "metadata_sha256": sha256_file(metadata_path),
        "excluded_manifest_sha256": sha256_file(args.exclude_manifest) if args.exclude_manifest else None,
        "seed": args.seed,
        "events_per_continent": args.events_per_continent,
        "selected_ids": [row["example_id"] for row in selected],
        "event_groups": sorted({row["event_id"] for row in selected}),
    }
    selection_path = args.output_dir / "SELECTION_FROZEN.json"
    if selection_path.exists():
        previous = json.loads(selection_path.read_text(encoding="utf-8"))
        if previous != selection:
            raise RuntimeError("Frozen lockbox selection differs from the current source or arguments")
    else:
        selection_path.write_text(json.dumps(selection, indent=2), encoding="utf-8")
    if args.freeze_selection_only:
        print(json.dumps({"selection": str(selection_path), "events": len(selection["event_groups"]), "samples": len(selected)}, indent=2))
        return
    selected_ids = {row["example_id"] for row in selected}
    repository_files = HfApi().list_repo_files(
        "moritzrengert1/wildfire_global",
        repo_type="dataset",
        revision=args.dataset_revision,
    )
    resolved_paths = {
        Path(path).stem: path
        for path in repository_files
        if path.startswith("images/post/") and Path(path).stem in selected_ids
    }
    missing_ids = selected_ids - set(resolved_paths)
    if missing_ids:
        raise FileNotFoundError(f"Selected example IDs are absent from the repository: {sorted(missing_ids)}")
    remote_paths = [resolved_paths[row["example_id"]] for row in selected]
    if args.download:
        for index, remote_path in enumerate(remote_paths, start=1):
            print(f"downloading EcoFireBias image {index}/{len(remote_paths)}: {remote_path}", flush=True)
            hf_hub_download(
                repo_id="moritzrengert1/wildfire_global",
                filename=remote_path,
                repo_type="dataset",
                revision=args.dataset_revision,
                local_dir=args.source_root,
            )

    image_root = args.output_dir / "images"
    image_root.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, object]] = []
    for index, row in enumerate(selected, start=1):
        resolved_source_path = resolved_paths[row["example_id"]]
        source_image = args.source_root / resolved_source_path
        if not source_image.exists():
            raise FileNotFoundError(f"Missing selected image: {source_image}. Run with --download.")
        image_path = image_root / f"{row['example_id']}.png"
        shutil.copy2(source_image, image_path)
        print(f"preparing EcoFireBias lockbox {index}/{len(selected)}: {row['example_id']}", flush=True)
        manifest_rows.append(
            {
                "sample_id": row["example_id"],
                "dataset": "ecofirebias_test",
                "image_path": str(image_path.resolve()),
                "mask_path": "",
                "split": "external_test",
                "source_split": row["split"],
                "geographic_group": f"{row['continent']}:{row['country']}",
                "event_group": row["event_id"],
                "label_source": "MODIS MCD64A1 and Sentinel-2 dNBR",
                "image_unit": "independent_224x224_sentinel2_post_event_rgb_chip",
                "fire_status": "positive" if row["kind"] == "burn" else "negative",
                "positive_pixel_fraction": float(row["burn_pixel_fraction_010"]),
                "continent": row["continent"],
                "country": row["country"],
                "ecosystem_group": row["ecosystem_group"],
                "image_sha256": sha256_file(image_path),
                "source_path": resolved_source_path,
            }
        )

    manifest_path = args.output_dir / "ecofirebias_120_lockbox_manifest.csv"
    write_csv(manifest_path, manifest_rows)
    freeze = {
        "status": "FROZEN_BEFORE_EVALUATION",
        "dataset": "moritzrengert1/wildfire_global (EcoFireBias)",
        "dataset_revision": args.dataset_revision,
        "license": "CC-BY-4.0",
        "source_partition": "event-level test split",
        "selection": "deterministic continent-balanced event pairs; no model outputs used",
        "seed": args.seed,
        "events_per_continent": args.events_per_continent,
        "selected_samples": len(manifest_rows),
        "unique_events": len({str(row["event_group"]) for row in manifest_rows}),
        "continents": sorted({str(row["continent"]) for row in manifest_rows}),
        "positive_scenes": sum(row["fire_status"] == "positive" for row in manifest_rows),
        "negative_scenes": sum(row["fire_status"] == "negative" for row in manifest_rows),
        "metadata_sha256": sha256_file(metadata_path),
        "manifest_sha256": sha256_file(manifest_path),
        "selection_sha256": sha256_file(selection_path),
        "excluded_event_groups": len(excluded_events),
        "model_selection_permitted": False,
        "evaluation_policy": "one-shot comparison of the frozen pruned-token candidate against the pre-existing baseline",
    }
    (args.output_dir / "LOCKBOX_FROZEN.json").write_text(json.dumps(freeze, indent=2), encoding="utf-8")
    print(json.dumps(freeze, indent=2))


def balanced_event_pairs(
    rows: list[dict[str, str]],
    events_per_continent: int,
    seed: int,
    excluded_events: set[str] | None = None,
) -> list[dict[str, str]]:
    grouped: dict[str, dict[str, dict[str, list[dict[str, str]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for row in rows:
        if row.get("split") == "test" and row.get("kind") in {"burn", "neg"} and row.get("event_id") not in (excluded_events or set()):
            grouped[row["continent"]][row["event_id"]][row["kind"]].append(row)

    selected: list[dict[str, str]] = []
    for continent in sorted(grouped):
        eligible = [
            event for event, kinds in grouped[continent].items() if kinds.get("burn") and kinds.get("neg")
        ]
        eligible.sort(key=lambda event: stable_key(seed, continent, event))
        if len(eligible) < events_per_continent:
            raise ValueError(f"{continent} has only {len(eligible)} eligible paired test events")
        for event in eligible[:events_per_continent]:
            for kind in ("burn", "neg"):
                candidates = sorted(
                    grouped[continent][event][kind],
                    key=lambda row: stable_key(seed, row["example_id"]),
                )
                selected.append(candidates[0])
    return selected


def stable_key(seed: int, *values: str) -> str:
    return hashlib.sha256(":".join([str(seed), *values]).encode("utf-8")).hexdigest()


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
