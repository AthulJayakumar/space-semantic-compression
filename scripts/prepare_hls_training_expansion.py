"""Extract only HLS tiles eligible for training and build a mixed manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.research_wildfire import load_grayscale_image, load_rgb_image  # noqa: E402


GROUP = re.compile(r"HLS\.S30\.T([0-9A-Z]{5})\.")
FIELDS = (
    "sample_id", "dataset", "image_path", "mask_path", "split", "source_split",
    "geographic_group", "event_group", "label_source", "image_unit",
    "cloud_fraction", "smoke_label_status",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def allowed_members(archive: tarfile.TarFile, held_groups: set[str], current_ids: set[str]):
    members = {member.name: member for member in archive.getmembers() if member.isfile()}
    selected = []
    for name, image_member in sorted(members.items()):
        if not (name.startswith(("training/", "validation/")) and name.endswith("_merged.tif")):
            continue
        sample_id = Path(name).stem
        match = GROUP.search(sample_id)
        if match is None:
            raise ValueError(f"Cannot parse HLS geographic group from {name}")
        if match.group(1) in held_groups or sample_id in current_ids:
            continue
        mask_name = name.removesuffix("_merged.tif") + ".mask.tif"
        if mask_name not in members:
            raise ValueError(f"Missing paired mask: {mask_name}")
        selected.append((image_member, members[mask_name], sample_id, match.group(1)))
    return selected


def extract_member(archive: tarfile.TarFile, member: tarfile.TarInfo, destination: Path) -> Path:
    root = destination.resolve()
    target = (root / member.name).resolve()
    if not target.is_relative_to(root):
        raise ValueError(f"Unsafe archive member path: {member.name}")
    if not target.exists() or target.stat().st_size != member.size:
        target.parent.mkdir(parents=True, exist_ok=True)
        stream = archive.extractfile(member)
        if stream is None:
            raise ValueError(f"Cannot read archive member: {member.name}")
        temporary = target.with_name(target.name + ".part")
        with stream, temporary.open("wb") as handle:
            shutil.copyfileobj(stream, handle)
        if temporary.stat().st_size != member.size:
            raise ValueError(f"Archive member extraction was incomplete: {member.name}")
        temporary.replace(target)
    if target.stat().st_size != member.size:
        raise ValueError(f"Extracted size mismatch: {target}")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("datasets/research_wildfire/hls_burn_scars/hls_burn_scars.tar.gz"))
    parser.add_argument("--hls-manifest", type=Path, default=Path("results/validated_hls_500/validated_scene_manifest.csv"))
    parser.add_argument("--selector-manifest", type=Path, default=Path("results/token_sus_selector/internal_geographic_split.csv"))
    parser.add_argument("--cems-manifest", type=Path, default=Path("datasets/research_wildfire/cems_hls/cems_hls_manifest.csv"))
    parser.add_argument("--destination", type=Path, default=Path("datasets/research_wildfire/hls_train_expansion"))
    parser.add_argument("--output", type=Path, default=Path("results/mixed_satellite_training/training_manifest.csv"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    hls = read_csv(args.hls_manifest)
    selector = read_csv(args.selector_manifest)
    cems = read_csv(args.cems_manifest)
    held = {row["geographic_group"] for row in hls if row["split"] != "train"}
    held |= {row["geographic_group"] for row in selector if row["split"] == "selector_validation"}
    existing = [row for row in selector if row["split"] == "selector_train"]
    existing_ids = {row["sample_id"] for row in existing}
    if len(existing) != 259 or len(existing_ids) != len(existing):
        raise ValueError("Expected the frozen 259-tile HLS selector-training partition")
    if any(row["geographic_group"] in held for row in existing):
        raise ValueError("HLS training/validation geographic overlap")
    with tarfile.open(args.archive, "r:gz") as archive:
        additions = allowed_members(archive, held, existing_ids)
        if args.dry_run:
            print(json.dumps({"new_hls_pairs": len(additions), "existing_hls": len(existing),
                              "cems_training": sum(row["split"] == "train" for row in cems)}, indent=2))
            return
        rows = []
        for row in existing:
            rows.append({field: row.get(field, "") for field in FIELDS} | {"split": "train"})
        selected_names = {
            member.name for image_member, mask_member, _, _ in additions
            for member in (image_member, mask_member)
        }
        extracted = 0
        for member in archive.getmembers():
            if member.name not in selected_names:
                continue
            extract_member(archive, member, args.destination)
            extracted += 1
            if extracted % 50 == 0 or extracted == len(selected_names):
                print(f"extracted {extracted}/{len(selected_names)} archive members", flush=True)
        for index, (image_member, mask_member, sample_id, group) in enumerate(additions, 1):
            image = (args.destination / image_member.name).resolve()
            mask = (args.destination / mask_member.name).resolve()
            if load_rgb_image(image).size != (512, 512) or load_grayscale_image(mask).size != (512, 512):
                raise ValueError(f"HLS image/mask shape mismatch: {sample_id}")
            rows.append({
                "sample_id": sample_id, "dataset": "ibm_nasa_hls_burn_scars",
                "image_path": str(image), "mask_path": str(mask), "split": "train",
                "source_split": image_member.name.split("/", 1)[0],
                "geographic_group": group, "event_group": "unavailable",
                "label_source": "IBM/NASA HLS Burn Scars mask",
                "image_unit": "independent_georeferenced_512x512_tile",
                "cloud_fraction": "", "smoke_label_status": "unavailable",
            })
            if index % 50 == 0 or index == len(additions):
                print(f"validated {index}/{len(additions)} HLS image/mask pairs", flush=True)
    for row in cems:
        if row["split"] != "train":
            continue
        image = (ROOT / row["image_path"]).resolve()
        mask = (ROOT / row["mask_path"]).resolve()
        if not image.is_file() or not mask.is_file():
            raise FileNotFoundError(f"Missing CEMS training pair: {image}, {mask}")
        rows.append({
            "sample_id": row["source_id"], "dataset": "cems_hls",
            "image_path": str(image), "mask_path": str(mask), "split": "train",
            "source_split": "train", "geographic_group": "unverified_cems",
            "event_group": "unavailable", "label_source": "CEMS burn-scar mask",
            "image_unit": "512x512_tile", "cloud_fraction": "",
            "smoke_label_status": "unavailable",
        })
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate training sample IDs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    hasher = hashlib.sha256()
    with args.archive.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    metadata = {
        "archive_sha256": hasher.hexdigest(),
        "existing_hls_training": len(existing),
        "new_hls_training": len(additions),
        "cems_training": sum(row["dataset"] == "cems_hls" for row in rows),
        "total_training_tiles": len(rows),
        "heldout_hls_geographic_groups": len(held),
        "heldout_geographic_overlap": 0,
        "cems_geographic_status": "unknown; source training partition only",
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
