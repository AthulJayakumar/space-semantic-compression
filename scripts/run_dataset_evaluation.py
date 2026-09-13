"""scripts.run_dataset_evaluation

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
from evaluation.dataset_evaluator import DatasetEvaluator


def main() -> None:
    parser = argparse.ArgumentParser(description="Run dataset-level semantic communication evaluation.")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--keep-ratio", type=float, default=0.45)
    parser.add_argument("--mission", default="wildfire_detection")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    evaluator = DatasetEvaluator(get_compression_service(), Path("results"))
    result = evaluator.evaluate_directory(Path(args.dataset_dir), args.keep_ratio, args.mission, args.limit)
    print(result)


if __name__ == "__main__":
    main()
