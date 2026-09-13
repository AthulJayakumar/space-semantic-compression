"""scripts.run_sentinel2_validation

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
from evaluation.sentinel2_validation import Sentinel2ValidationPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Sentinel-2 Earth Observation validation.")
    parser.add_argument("--sentinel2-root", type=Path, default=Path("datasets/sentinel2"))
    parser.add_argument("--wildfire-results-root", type=Path, default=Path("results/wildfire_publication_evidence"))
    parser.add_argument("--output-root", type=Path, default=Path("results/earth_observation_validation"))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    outputs = Sentinel2ValidationPipeline(get_compression_service(), args.output_root).run(
        sentinel2_root=args.sentinel2_root,
        wildfire_results_root=args.wildfire_results_root,
        limit=args.limit,
    )
    for key, value in outputs.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
