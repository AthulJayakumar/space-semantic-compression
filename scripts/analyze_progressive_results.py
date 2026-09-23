"""Regenerate event-level progressive comparisons and an early-contact curve."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.progressive_downlink import METHODS, paired_comparisons, write_rows  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    args = parser.parse_args()
    with (args.results_dir / "progressive_rows.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    comparisons = paired_comparisons(rows)
    if comparisons:
        write_rows(args.results_dir / "progressive_event_paired_tests.csv", comparisons)
    with (args.results_dir / "progressive_event_summary.csv").open(newline="", encoding="utf-8") as handle:
        summary = list(csv.DictReader(handle))
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.6), dpi=220)
    for method in METHODS:
        points = sorted((row for row in summary if row["method"] == method), key=lambda row: float(row["cumulative_bytes_mean"]))
        ax.plot(
            [float(row["cumulative_bytes_mean"]) for row in points],
            [float(row["sus_mean"]) for row in points],
            marker="o",
            linewidth=1.8,
            label=method.replace("vqvae_", ""),
        )
    ax.set_xlabel("Cumulative bytes received before contact ends")
    ax.set_ylabel("Semantic Utility Score")
    ax.grid(alpha=0.25)
    ax.legend(title="Token ordering")
    fig.tight_layout()
    figure = args.results_dir / "progressive_rate_vs_sus.png"
    fig.savefig(figure, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {len(comparisons)} paired tests and {figure}")


if __name__ == "__main__":
    main()
