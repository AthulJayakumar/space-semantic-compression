"""Apply the frozen advancement rule to the task-aligned selector validation."""

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


CANDIDATE = "vqvae_utility"
CONTROLS = ("vqvae_fixed_utility", "vqvae_random", "vqvae_entropy")


def decide(rows: list[dict[str, str]], metadata: dict[str, object]) -> dict[str, object]:
    complete = (
        int(metadata["requested_items"]) == int(metadata["completed_items"])
        and int(metadata["excluded_items"]) == 0
        and len(rows) == int(metadata["requested_items"]) * len(metadata["methods"]) * len(metadata["budget_levels"])
    )
    available_metrics = (
        "sus", "detector_retention", "label_dice", "label_iou", "psnr", "ssim", "lpips", "actual_bytes"
    )
    comparisons = [
        paired_group_comparison(rows, CANDIDATE, baseline, metric)
        for metric in available_metrics
        if all(row.get(metric) not in (None, "") for row in rows if int(row["budget_index"]) == 0)
        for baseline in CONTROLS
    ]
    sus = [row for row in comparisons if row["metric"] == "sus"]
    detector_random = next(
        row for row in comparisons
        if row["metric"] == "detector_retention" and row["baseline"] == "vqvae_random"
    )
    pass_rule = (
        complete
        and all(float(row["ci_low"]) > 0 for row in sus)
        and float(detector_random["ci_low"]) >= -0.02
    )
    return {
        "decision": "advance_to_reserved_replication" if pass_rule else "no_go",
        "candidate": CANDIDATE,
        "complete_validation": complete,
        "requested_items": int(metadata["requested_items"]),
        "completed_items": int(metadata["completed_items"]),
        "excluded_items": int(metadata["excluded_items"]),
        "primary_budget_index": 0,
        "sampling_unit": "HLS geographic group",
        "rule": "SUS paired geographic-cluster bootstrap 95% CI lower > 0 against fixed utility, random and entropy; detector retention lower CI versus random >= -0.02; complete validation set",
        "comparisons": comparisons,
        "caveat": "This official HLS validation partition was used in prior selector development and is not an untouched external test.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("results/task_aligned_selector_validation"))
    args = parser.parse_args()
    rows = read_rows(args.run_dir / "matched_rate_rows.csv")
    metadata = json.loads((args.run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    decision = decide(rows, metadata)
    (args.run_dir / "ADVANCEMENT_DECISION.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    with (args.run_dir / "ADVANCEMENT_COMPARISONS.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(decision["comparisons"][0]))
        writer.writeheader()
        writer.writerows(decision["comparisons"])
    print(json.dumps({key: value for key, value in decision.items() if key != "comparisons"}, indent=2))


if __name__ == "__main__":
    main()
