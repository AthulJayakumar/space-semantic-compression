"""Apply the predeclared advancement rule to a learned selector benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.matched_rate import bootstrap_mean_ci  # noqa: E402
from evaluation.statistics import StatisticalValidator  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate learned-selector advancement without test leakage.")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--partition", default="internal geographic validation")
    parser.add_argument("--candidate", default="vqvae_utility")
    args = parser.parse_args()

    rows = read_csv(args.results_dir / "matched_rate_rows.csv")
    comparisons = paired_average_rate_tests(rows, args.candidate)
    fixed = next(row for row in comparisons if row["baseline"] == "vqvae_fixed_utility")
    entropy = next(row for row in comparisons if row["baseline"] == "vqvae_entropy")
    go = float(fixed["difference_ci_low"]) > 0.0 and float(entropy["difference_ci_low"]) > 0.0
    decision = {
        "decision": "go_for_official_validation" if go else "no_go_for_official_validation",
        "selection_partition": args.partition,
        "official_validation_used": False,
        "test_partition_used": False,
        "candidate": args.candidate,
        "go_rule": "paired average low/middle SUS bootstrap CI must be above zero versus fixed and entropy selectors",
        "comparison_vs_fixed": fixed,
        "comparison_vs_entropy": entropy,
    }
    write_csv(args.results_dir / "advancement_statistics.csv", comparisons)
    (args.results_dir / "advancement_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    (args.results_dir / "advancement_report.md").write_text(build_report(decision), encoding="utf-8")
    print(json.dumps(decision, indent=2))


def paired_average_rate_tests(rows, candidate: str) -> list[dict[str, object]]:
    validator = StatisticalValidator()
    values: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        if int(row["budget_index"]) not in {0, 1}:
            continue
        values.setdefault(row["method"], {}).setdefault(row["sample_id"], []).append(float(row["sus"]))
    candidate_values = {sample_id: float(np.mean(items)) for sample_id, items in values[candidate].items()}
    output = []
    for baseline in ("vqvae_fixed_utility", "vqvae_entropy", "vqvae_random"):
        baseline_values = {sample_id: float(np.mean(items)) for sample_id, items in values[baseline].items()}
        ids = sorted(set(candidate_values) & set(baseline_values))
        base = np.asarray([baseline_values[sample_id] for sample_id in ids], dtype="float64")
        selected = np.asarray([candidate_values[sample_id] for sample_id in ids], dtype="float64")
        differences = selected - base
        t_result = validator.paired_t_test(base.tolist(), selected.tolist())
        w_result = validator.wilcoxon(base.tolist(), selected.tolist())
        low, high = bootstrap_mean_ci(differences, seed=20260922)
        output.append(
            {
                "baseline": baseline,
                "candidate": candidate,
                "n": len(ids),
                "baseline_mean_low_middle_sus": float(base.mean()),
                "candidate_mean_low_middle_sus": float(selected.mean()),
                "mean_paired_difference": float(differences.mean()),
                "difference_ci_low": low,
                "difference_ci_high": high,
                "paired_t_p_value": t_result.p_value,
                "wilcoxon_p_value": w_result.p_value,
                "cohens_dz": t_result.effect_size,
            }
        )
    return output


def build_report(decision: dict[str, object]) -> str:
    fixed = decision["comparison_vs_fixed"]
    entropy = decision["comparison_vs_entropy"]
    return "\n".join(
        [
            "# SUS-Aligned Selector Advancement Decision",
            "",
            "The learned selector was evaluated at the exact same per-image byte budgets as all baselines. The decision statistic is each scene's mean SUS across the low and middle operating points.",
            "",
            f"- Versus fixed utility: {fixed['mean_paired_difference']:+.2f} SUS, 95% CI {fixed['difference_ci_low']:+.2f} to {fixed['difference_ci_high']:+.2f}, paired t-test p={fixed['paired_t_p_value']:.4g}",
            f"- Versus entropy: {entropy['mean_paired_difference']:+.2f} SUS, 95% CI {entropy['difference_ci_low']:+.2f} to {entropy['difference_ci_high']:+.2f}, paired t-test p={entropy['paired_t_p_value']:.4g}",
            "",
            f"**{str(decision['decision']).replace('_', ' ').title()}.**",
            "",
            "The official validation and final test partitions remain unused by this decision. Advancement requires both confidence-interval lower bounds to exceed zero.",
        ]
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
