"""Refresh matched-rate figures and text from saved measurements without rerunning inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.matched_rate import MatchedRateBenchmark  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    args = parser.parse_args()
    metadata = json.loads((args.results_dir / "run_metadata.json").read_text(encoding="utf-8"))
    benchmark = MatchedRateBenchmark.__new__(MatchedRateBenchmark)
    benchmark.output_dir = args.results_dir
    benchmark.methods = tuple(metadata["methods"])
    benchmark.image_size = int(metadata["image_size"])
    benchmark.skip_lpips = bool(metadata["skip_lpips"])
    summary = benchmark._read_csv(args.results_dir / "matched_rate_summary.csv")
    statistics = benchmark._read_csv(args.results_dir / "matched_rate_statistics.csv")
    confirmatory = benchmark._read_csv(args.results_dir / "confirmatory_source_validation_summary.csv")
    confirmatory_statistics = benchmark._read_csv(args.results_dir / "confirmatory_source_validation_statistics.csv")
    pareto = benchmark._read_csv(args.results_dir / "rate_utility_pareto.csv")
    exclusions = benchmark._read_csv(args.results_dir / "excluded_samples.csv")
    benchmark.generate_figures(summary, pareto)
    benchmark.write_report(metadata, summary, statistics, confirmatory, confirmatory_statistics, exclusions)
    print(f"Refreshed figures and report in {args.results_dir}")


if __name__ == "__main__":
    main()
