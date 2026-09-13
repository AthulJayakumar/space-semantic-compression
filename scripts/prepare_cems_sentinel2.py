"""scripts.prepare_cems_sentinel2

Plain-English purpose: Command-line runners for datasets, benchmarks, reports, and exports.

This file is part of the public Space Semantic Compression research demo.
It is documented at module level so researchers, supervisors, and non-specialist
readers can understand where the file fits before reading implementation details.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tarfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def safe_member_name(name: str) -> str:
    return Path(name).name.replace(" ", "_")


def combine_parts(parts: list[Path], target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        return target
    with target.open("wb") as output:
        for part in parts:
            with part.open("rb") as handle:
                shutil.copyfileobj(handle, output, length=1024 * 1024 * 16)
    return target


def download_parts(repo_id: str, split: str, work_dir: Path) -> list[Path]:
    api = HfApi()
    siblings = api.dataset_info(repo_id, files_metadata=True).siblings
    part_files = sorted(
        sibling.rfilename
        for sibling in siblings
        if sibling.rfilename.startswith(f"data/{split}/") and sibling.rfilename.endswith(".part")
    )
    if not part_files:
        raise RuntimeError(f"No split archive parts found for split={split}")
    work_dir.mkdir(parents=True, exist_ok=True)
    return [
        Path(hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset", local_dir=work_dir))
        for filename in part_files
    ]


def extract_sentinel_images(archive: Path, output_root: Path, max_images: int | None) -> dict[str, object]:
    images_dir = output_root / "images"
    metadata_dir = output_root / "metadata"
    labels_dir = output_root / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    image_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    extracted_images = 0
    extracted_metadata = 0
    extracted_masks = 0
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            suffix = Path(member.name).suffix.lower()
            name_lower = member.name.lower()
            source = tar.extractfile(member)
            if source is None:
                continue
            if suffix in image_exts and ("s2" in name_lower or "sentinel" in name_lower or "l2a" in name_lower):
                target = images_dir / f"cems_{extracted_images:06d}_{safe_member_name(member.name)}"
                if not target.exists():
                    with target.open("wb") as handle:
                        shutil.copyfileobj(source, handle)
                extracted_images += 1
            elif suffix == ".json":
                target = metadata_dir / safe_member_name(member.name)
                if not target.exists():
                    with target.open("wb") as handle:
                        shutil.copyfileobj(source, handle)
                extracted_metadata += 1
            elif suffix in image_exts and any(token in name_lower for token in ("mask", "del", "cm", "burn")):
                target = labels_dir / safe_member_name(member.name)
                if not target.exists():
                    with target.open("wb") as handle:
                        shutil.copyfileobj(source, handle)
                extracted_masks += 1
            if max_images is not None and extracted_images >= max_images:
                break
    return {
        "images": extracted_images,
        "metadata": extracted_metadata,
        "masks": extracted_masks,
        "archive": str(archive),
        "output_root": str(output_root),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare 100+/500 Sentinel-2 CEMS wildfire scenes.")
    parser.add_argument("--repo-id", default="9334hq/wildfires-cems")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--output-root", type=Path, default=Path("datasets/sentinel2"))
    parser.add_argument("--work-dir", type=Path, default=Path("datasets/sentinel2/_hf_parts"))
    parser.add_argument("--max-images", type=int, default=500)
    args = parser.parse_args()

    parts = download_parts(args.repo_id, args.split, args.work_dir)
    archive = combine_parts(parts, args.work_dir / f"{args.split}.tar.gz")
    manifest = extract_sentinel_images(archive, args.output_root, args.max_images)
    manifest.update({"repo_id": args.repo_id, "split": args.split, "parts": [str(part) for part in parts]})
    path = args.output_root / "cems_prepare_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
