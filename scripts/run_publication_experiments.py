"""scripts.run_publication_experiments

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
from evaluation.publication_experiments import PublicationExperimentRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Run publishable CompressAI wildfire semantic-compression experiments.")
    parser.add_argument("--image", action="append", help="Image path. Can be passed multiple times.")
    parser.add_argument("--image-dir", help="Directory of images.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--mission", default="wildfire_detection")
    parser.add_argument("--output-root", default="results")
    args = parser.parse_args()

    paths: list[Path] = []
    if args.image:
        paths.extend(Path(path) for path in args.image)
    if args.image_dir:
        root = Path(args.image_dir)
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
            paths.extend(root.rglob(ext))
    paths = sorted(set(paths))
    if args.limit is not None:
        paths = paths[: args.limit]
    if not paths:
        raise SystemExit("No images provided.")

    runner = PublicationExperimentRunner(get_compression_service(), Path(args.output_root))
    outputs = runner.run_all(paths, mission=args.mission)
    for key, value in outputs.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
