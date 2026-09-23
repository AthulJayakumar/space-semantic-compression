"""Fetch pinned EcoFireBias inputs/labels, without running an evaluator.

The probe uses a train-split chip. Eligible mode uses only the geographic
subset fixed by audit_ecofirebias_replication.py; rejected pairs are untouched.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from PIL import Image
import tifffile


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "datasets/wildfire_global_v1"
AUDIT = ROOT / "results/future_satellite_cohort_audit/ecofirebias_replication_audit.json"
REPO_ID = "moritzrengert1/wildfire_global"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metadata_rows() -> dict[str, dict[str, str]]:
    with (SOURCE / "metadata.csv").open(newline="", encoding="utf-8") as handle:
        return {row["example_id"]: row for row in csv.DictReader(handle)}


def inspect_file(path: Path, kind: str, expected_size: int) -> dict[str, object]:
    if kind == "rgb":
        with Image.open(path) as image:
            shape = [image.height, image.width, len(image.getbands())]
            dtype = image.mode
    else:
        with tifffile.TiffFile(path) as image:
            page = image.pages[0]
            shape = list(page.shape)
            dtype = str(page.dtype)
            if len(image.pages) != 1:
                raise ValueError(f"Expected one raw dNBR plane: {path}")
    if shape[:2] != [expected_size, expected_size]:
        raise ValueError(f"Unexpected {kind} shape for {path}: {shape}")
    return {"path": str(path.relative_to(SOURCE)), "sha256": sha256(path), "bytes": path.stat().st_size, "shape": shape, "dtype_or_mode": dtype}


def fetch(mode: str) -> dict[str, object]:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    if audit["status"] != "pre_evaluation_geographic_and_asset_audit":
        raise ValueError("Geographic audit is not frozen")
    rows = metadata_rows()
    if mode == "probe":
        ids = [next(row["example_id"] for row in rows.values() if row["split"] == "train" and row["kind"] == "burn")]
    elif mode == "development":
        gate = json.loads((ROOT / "results/future_satellite_cohort_audit/DEVELOPMENT_GATE_FROZEN.json").read_text(encoding="utf-8"))
        ids = gate["development_ids"]
        if any(rows[sample_id]["split"] != "train" for sample_id in ids):
            raise ValueError("Development selection contains a non-training example")
    elif mode == "official_validation":
        plan = json.loads((ROOT / "results/future_satellite_cohort_audit/ecofirebias_official_validation/VALIDATION_PLAN_FROZEN.json").read_text(encoding="utf-8"))
        ids = plan["sample_ids"]
        if any(rows[sample_id]["split"] != "val" for sample_id in ids):
            raise ValueError("Official-validation selection contains a non-validation example")
    else:
        ids = [sample_id for event in audit["events"] if event["eligible"] for sample_id in event["example_ids"]]
    revision = json.loads((ROOT / "results/external_lockbox_ecofirebias_replication_120/SELECTION_FROZEN.json").read_text(encoding="utf-8"))["dataset_revision"]
    if mode == "official_validation" and plan["dataset_revision"] != revision:
        raise ValueError("Official-validation revision differs from frozen source")
    remote = HfApi().list_repo_files(REPO_ID, repo_type="dataset", revision=revision)
    kinds = ("rgb", "dnbr", "post_geotiff") if mode in ("probe", "development", "official_validation") else ("rgb", "dnbr")
    prefixes = {"rgb": "images/post/", "dnbr": "geotiff/dnbr/", "post_geotiff": "geotiff/post/"}
    resolved = {(Path(path).stem, kind): path for path in remote for kind in kinds if path.startswith(prefixes[kind])}
    missing = [(sample_id, kind) for sample_id in ids for kind in kinds if (sample_id, kind) not in resolved]
    if missing:
        raise FileNotFoundError(f"Missing pinned remote assets: {missing[:5]}")
    assets = []
    for index, sample_id in enumerate(ids, start=1):
        row = rows[sample_id]
        expected_size = int(row["chip_size_px"])
        entry = {"example_id": sample_id, "event_id": row["event_id"], "kind": row["kind"], "assets": {}}
        for kind in kinds:
            path = hf_hub_download(
                repo_id=REPO_ID, filename=resolved[(sample_id, kind)], repo_type="dataset",
                revision=revision, local_dir=SOURCE,
            )
            entry["assets"][kind] = inspect_file(Path(path), kind, expected_size)
        assets.append(entry)
        print(f"validated {index}/{len(ids)} paired assets", flush=True)
    result = {
        "mode": mode,
        "dataset_revision": revision,
        "geographic_audit_sha256": sha256(AUDIT),
        "validated_images": len(assets),
        "assets": assets,
        "model_outputs_accessed": False,
    }
    output = ROOT / f"results/future_satellite_cohort_audit/ecofirebias_assets_{mode}.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return {"output": str(output), "validated_images": len(assets)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("probe", "development", "official_validation", "eligible"), required=True)
    args = parser.parse_args()
    print(json.dumps(fetch(args.mode), indent=2))


if __name__ == "__main__":
    main()
