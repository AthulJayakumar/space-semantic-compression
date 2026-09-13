"""scripts.run_wildfire_validation

Plain-English purpose: Command-line runners for datasets, benchmarks, reports, and exports.

This file is part of the public Space Semantic Compression research demo.
It is documented at module level so researchers, supervisors, and non-specialist
readers can understand where the file fits before reading implementation details.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.factory import get_compression_service
from datasets.wildfire_datasets import DFireDataset, FLAMEDataset, WildfireImageDataset, discover_wildfire_datasets
from evaluation.wildfire_validation import WildfireValidationPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run dataset-level wildfire validation for CompressAI.")
    parser.add_argument("--datasets-root", type=Path, default=Path("datasets"))
    parser.add_argument("--dfire-root", type=Path)
    parser.add_argument("--flame-root", type=Path)
    parser.add_argument("--fallback-image-dir", type=Path)
    parser.add_argument("--allow-fallback", action="store_true")
    parser.add_argument("--split", default="benchmark")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-root", type=Path, default=Path("results/wildfire_validation"))
    parser.add_argument("--mission", default="wildfire_detection")
    args = parser.parse_args()

    datasets = {}
    if args.dfire_root and args.dfire_root.exists():
        datasets["dfire"] = DFireDataset(args.dfire_root, split=args.split)
    if args.flame_root and args.flame_root.exists():
        datasets["flame"] = FLAMEDataset(args.flame_root, split=args.split)
    datasets.update({key: value for key, value in discover_wildfire_datasets(args.datasets_root, args.split).items() if key not in datasets})

    if not datasets and args.allow_fallback and args.fallback_image_dir and args.fallback_image_dir.exists():
        datasets["local_fallback_not_dfire_or_flame"] = WildfireImageDataset(args.fallback_image_dir, split=args.split)

    if not datasets:
        raise SystemExit(
            "No DFire/FLAME datasets found. Place data under datasets/dfire and datasets/flame, "
            "or pass --dfire-root/--flame-root. Use --allow-fallback only for pipeline smoke tests."
        )

    outputs = WildfireValidationPipeline(get_compression_service(), args.output_root).run(
        datasets=datasets,
        mission=args.mission,
        limit=args.limit,
    )
    for key, value in outputs.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
