"""scripts.run_edge_benchmark

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

from deployment.edge_benchmark import EdgeDeploymentBenchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CompressAI edge deployment benchmark.")
    parser.add_argument("--image", required=True, type=Path, help="Input image path.")
    parser.add_argument("--checkpoint", default=Path("checkpoints/vqvae_s16k8.pt"), type=Path)
    parser.add_argument("--output-dir", default=Path("results/edge"), type=Path)
    parser.add_argument("--repeats", default=3, type=int)
    args = parser.parse_args()

    rows = EdgeDeploymentBenchmark(args.checkpoint, args.output_dir).run(args.image, repeats=args.repeats)
    print(f"Wrote {len(rows)} edge benchmark rows to {args.output_dir}")


if __name__ == "__main__":
    main()
