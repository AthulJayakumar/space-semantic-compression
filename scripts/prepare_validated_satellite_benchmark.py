"""Prepare and audit a leakage-safe HLS wildfire benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.dataset_validity import (  # noqa: E402
    discover_extracted_hls_pairs,
    extract_hls_pairs,
    validate_hls_pairs,
    write_validity_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract and validate independent HLS burn-scar tiles.")
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("datasets/research_wildfire/hls_burn_scars/hls_burn_scars.tar.gz"),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("datasets/research_wildfire/hls_burn_scars_500"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/validated_hls_500"),
    )
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--reuse-extracted", action="store_true")
    args = parser.parse_args()

    if args.reuse_extracted:
        pairs = discover_extracted_hls_pairs(args.data_root)[: args.limit]
    else:
        if not args.archive.exists():
            raise FileNotFoundError(f"Official HLS archive not found: {args.archive}")
        pairs = extract_hls_pairs(args.archive, args.data_root, limit=args.limit)
    rows, audit = validate_hls_pairs(pairs)
    write_validity_outputs(rows, audit, args.output_dir)
    print(json.dumps({"pairs": len(pairs), "output_dir": str(args.output_dir), "audit": audit}, indent=2))


if __name__ == "__main__":
    main()
