"""Apply the frozen mixed-data checkpoint gate to paired severe-rate results."""

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


def compare(old_path: Path, new_path: Path) -> dict[str, object]:
    old = [row for row in read_rows(old_path) if row["method"] == "vqvae_fixed_utility"]
    new = [row for row in read_rows(new_path) if row["method"] == "vqvae_fixed_utility"]
    key = lambda row: (row["sample_id"], int(row["budget_index"]))
    old_by_key = {key(row): row for row in old}
    new_by_key = {key(row): row for row in new}
    if len(old_by_key) != len(old) or len(new_by_key) != len(new):
        raise ValueError("Duplicate sample/budget rows")
    if not new_by_key.keys() <= old_by_key.keys():
        raise ValueError("Every candidate image and budget must exist in the baseline run")
    budget_indices = sorted({budget for _, budget in new_by_key})
    if 1 not in budget_indices:
        raise ValueError("The predeclared 1200-byte budget (index 1) is required")
    old = [row for row in old if key(row) in new_by_key]
    old_by_key = {key(row): row for row in old}
    if old_by_key.keys() != new_by_key.keys():
        raise ValueError("Checkpoint runs are not paired")
    for sample_key in new_by_key:
        prior, current = old_by_key[sample_key], new_by_key[sample_key]
        for field in ("target_bytes", "geographic_group", "label_source"):
            if prior[field] != current[field]:
                raise ValueError(f"Mismatched {field} for {sample_key}")
    combined = [dict(row, method="old_vqvae") for row in old]
    combined.extend(dict(row, method="mixed_wire_vqvae") for row in new)
    metrics = ("sus", "label_dice", "detector_retention", "psnr", "ssim", "encoded_bytes")
    comparisons = [
        paired_group_comparison(combined, "mixed_wire_vqvae", "old_vqvae", metric, budget_index)
        for budget_index in budget_indices
        for metric in metrics
    ]
    primary = {row["metric"]: row for row in comparisons if row["budget_index"] == 1}
    summaries = [
        {
            "method": label,
            "budget_index": budget_index,
            "metric": metric,
            "mean": sum(float(row[metric]) for row in source if int(row["budget_index"]) == budget_index)
            / sum(int(row["budget_index"]) == budget_index for row in source),
        }
        for label, source in (("old_vqvae", old), ("mixed_wire_vqvae", new))
        for budget_index in budget_indices
        for metric in metrics
    ]
    return {
        "decision": "pass_development_gate" if primary["sus"]["ci_low"] > 0 and primary["label_dice"]["ci_low"] > 0 else "no_go",
        "primary_budget_bytes": 1200,
        "sampling_unit": "HLS geographic group",
        "n_images": len({sample for sample, _ in new_by_key}),
        "n_groups": primary["sus"]["n_groups"],
        "comparisons": comparisons,
        "summaries": summaries,
        "limitations": "Internal development validation only; classical codec superiority requires separate comparison and independent replication.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, default=Path("results/extreme_rate_internal_512/extreme_rate_rows.csv"))
    parser.add_argument("--new", type=Path, default=Path("results/extreme_rate_mixed_wire_internal_512/extreme_rate_rows.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/extreme_rate_mixed_wire_internal_512"))
    args = parser.parse_args()
    result = compare(args.old, args.new)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "checkpoint_decision.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (args.output_dir / "checkpoint_comparisons.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result["comparisons"][0]))
        writer.writeheader()
        writer.writerows(result["comparisons"])
    print(json.dumps({key: value for key, value in result.items() if key not in ("comparisons", "summaries")}, indent=2))
    for row in result["comparisons"]:
        if row["budget_index"] == 1:
            print(f"{row['metric']}: {row['mean_difference']:+.4f} CI [{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]")


if __name__ == "__main__":
    main()
