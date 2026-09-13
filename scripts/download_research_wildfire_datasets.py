"""Download and prepare larger wildfire Earth Observation datasets.

This command is intentionally conservative. By default it prints the recommended
dataset plan without downloading raw data. Use ``--execute`` when you are ready
to acquire the selected Hugging Face dataset snapshots.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.research_wildfire import (  # noqa: E402
    allow_patterns_for,
    build_manifest,
    dataset_plan,
    selected_specs,
    validate_manifest,
    write_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare large wildfire EO datasets for supervised token-selector training.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["cems_hls", "hls_burn_scars"],
        help="Dataset keys to prepare. Choices include cems_hls, hls_burn_scars, firescope_small, eo4wildfires.",
    )
    parser.add_argument("--output-root", type=Path, default=Path("datasets/research_wildfire"))
    parser.add_argument("--profile", choices=["small", "standard", "full"], default="small")
    parser.add_argument("--execute", action="store_true", help="Actually download snapshots. Without this, only print the plan.")
    parser.add_argument("--manifest-name", default="wildfire_research_manifest")
    args = parser.parse_args()

    specs = selected_specs(args.datasets)
    args.output_root.mkdir(parents=True, exist_ok=True)

    if not args.execute:
        payload = {
            "mode": "plan_only",
            "message": "Run again with --execute to download and prepare these datasets.",
            "profile": args.profile,
            "datasets": dataset_plan(args.datasets),
        }
        print(json.dumps(payload, indent=2))
        return

    download_records: list[dict[str, Any]] = []
    all_rows = []
    for spec in specs:
        local_dir = args.output_root / spec.local_dir
        record = download_snapshot(spec.repo_id, local_dir, allow_patterns_for(spec, args.profile))
        record.update({"dataset": spec.name, "task": spec.task, "license": spec.license, "notes": spec.notes})
        download_records.append(record)

        rows = build_manifest(local_dir, spec.name)
        all_rows.extend(rows)
        dataset_manifest = local_dir / f"{spec.name}_manifest.csv"
        write_manifest(rows, dataset_manifest, dataset_manifest.with_suffix(".json"))
        record["paired_items"] = len(rows)
        record["integrity"] = validate_manifest(rows)

    manifest_csv = args.output_root / f"{args.manifest_name}.csv"
    manifest_json = args.output_root / f"{args.manifest_name}.json"
    write_manifest(all_rows, manifest_csv, manifest_json)

    summary = {
        "profile": args.profile,
        "output_root": str(args.output_root),
        "combined_manifest_csv": str(manifest_csv),
        "combined_manifest_json": str(manifest_json),
        "paired_items": len(all_rows),
        "integrity": validate_manifest(all_rows),
        "downloads": download_records,
    }
    summary_path = args.output_root / "research_dataset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def download_snapshot(repo_id: str, local_dir: Path, allow_patterns: tuple[str, ...]) -> dict[str, Any]:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("Install huggingface_hub before downloading research datasets.") from exc

    local_dir.mkdir(parents=True, exist_ok=True)
    resolved_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        allow_patterns=list(allow_patterns) if allow_patterns else None,
    )
    return {
        "repo_id": repo_id,
        "local_dir": str(local_dir),
        "resolved_path": resolved_path,
        "allow_patterns": list(allow_patterns),
    }


if __name__ == "__main__":
    main()
