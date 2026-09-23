"""Summarize independent-mask specificity for the fixed 1200-byte screen."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.decoder_aware_decision import paired_group_comparison, read_rows  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, default=Path("results/mask_aware_specificity_internal_512/extreme_rate_rows.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/mask_aware_specificity_internal_512"))
    args = parser.parse_args()
    rows = read_rows(args.rows)
    if {int(row["target_bytes"]) for row in rows} != {1200}:
        raise ValueError("Specificity screen requires only the predeclared 1200-byte budget")
    methods = ("vqvae_fixed_utility", "jpeg_rdo", "jpeg2000_rdo")
    metrics = (
        "label_dice", "label_precision", "label_recall", "label_false_positive_rate",
        "predicted_positive_fraction", "truth_positive_fraction", "sus", "encoded_bytes",
    )
    summary = []
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        if not selected or len(selected) != len({row["sample_id"] for row in selected}):
            raise ValueError(f"Missing or duplicate samples for {method}")
        negative = [row for row in selected if float(row["truth_positive_fraction"]) == 0]
        summary.append({
            "method": method,
            "n_images": len(selected),
            "n_negative_images": len(negative),
            **{f"{metric}_mean": sum(float(row[metric]) for row in selected) / len(selected) for metric in metrics},
            "negative_scene_false_positive_rate_mean": (
                sum(float(row["label_false_positive_rate"]) for row in negative) / len(negative)
                if negative else None
            ),
        })
    comparisons = [
        paired_group_comparison(rows, "vqvae_fixed_utility", baseline, metric, 0)
        for baseline in methods[1:]
        for metric in ("label_dice", "label_precision", "label_recall", "label_false_positive_rate", "sus")
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "specificity_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (args.output_dir / "specificity_comparisons.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
