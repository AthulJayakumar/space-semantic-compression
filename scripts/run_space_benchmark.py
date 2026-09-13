"""scripts.run_space_benchmark

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

from backend.services.factory import get_benchmark_service


def main() -> None:
    parser = argparse.ArgumentParser(description="Run semantic space-compression benchmark for one image.")
    parser.add_argument("--image", required=True, help="Path to image file")
    args = parser.parse_args()

    image_path = Path(args.image)
    result = get_benchmark_service().run_image_benchmark(image_path.read_bytes(), image_path.name)
    print(f"benchmark_id={result.benchmark_id}")
    print(f"csv_path={result.csv_path}")
    print(f"json_path={result.json_path}")


if __name__ == "__main__":
    main()
