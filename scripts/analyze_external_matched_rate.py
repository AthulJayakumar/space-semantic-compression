"""Summarize matched-byte codec results with wildfire events as sampling units."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.clustered_matched_rate import analyze, compare_checkpoints, load_rows, write_csv  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--baseline-dir", type=Path)
    args = parser.parse_args()
    rows = load_rows(args.results_dir / "matched_rate_rows.csv")
    comparisons = analyze(rows)
    destination = args.results_dir / "event_level_paired_statistics.csv"
    write_csv(destination, comparisons)
    print(f"Wrote {len(comparisons)} comparisons to {destination}")
    if args.baseline_dir is not None:
        baseline_rows = load_rows(args.baseline_dir / "matched_rate_rows.csv")
        checkpoint = compare_checkpoints(baseline_rows, rows)
        destination = args.results_dir / "event_level_checkpoint_comparison.csv"
        write_csv(destination, checkpoint)
        print(f"Wrote {len(checkpoint)} matched-budget checkpoint comparisons to {destination}")


if __name__ == "__main__":
    main()
