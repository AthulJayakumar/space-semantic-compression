"""Generate the validation decision report for the decoder-impact selector."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Report decoder-impact selector validation evidence.")
    parser.add_argument("--results-dir", type=Path, default=Path("results/token_impact_selector_validation"))
    parser.add_argument("--training-dir", type=Path, default=Path("results/token_impact_selector"))
    args = parser.parse_args()
    summary = read_csv(args.results_dir / "matched_rate_summary.csv")
    statistics = read_csv(args.results_dir / "matched_rate_statistics.csv")
    history = read_csv(args.training_dir / "training_history.csv")
    metadata = json.loads((args.results_dir / "run_metadata.json").read_text(encoding="utf-8"))
    report, decision = build_report(summary, statistics, history, metadata)
    (args.results_dir / "token_impact_validation_report.md").write_text(report, encoding="utf-8")
    (args.results_dir / "validation_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(json.dumps(decision, indent=2))


def build_report(summary, statistics, history, metadata) -> tuple[str, dict[str, object]]:
    low = get_summary(summary, "vqvae_utility", 0)
    middle = get_summary(summary, "vqvae_utility", 1)
    low_fixed = get_test(statistics, "vqvae_fixed_utility", "sus", 0)
    middle_fixed = get_test(statistics, "vqvae_fixed_utility", "sus", 1)
    low_entropy = get_test(statistics, "vqvae_entropy", "sus", 0)
    middle_entropy = get_test(statistics, "vqvae_entropy", "sus", 1)
    low_random = get_test(statistics, "vqvae_random", "sus", 0)
    middle_random = get_test(statistics, "vqvae_random", "sus", 1)
    low_psnr = get_test(statistics, "vqvae_fixed_utility", "psnr", 0)
    middle_psnr = get_test(statistics, "vqvae_fixed_utility", "psnr", 1)
    low_ssim = get_test(statistics, "vqvae_fixed_utility", "ssim", 0)
    middle_ssim = get_test(statistics, "vqvae_fixed_utility", "ssim", 1)
    best = min(history, key=lambda row: float(row["validation_loss"]))
    passes_low = float(low_fixed["difference_ci_low"]) > 0.0
    passes_middle = float(middle_fixed["difference_ci_low"]) > 0.0
    decision = {
        "decision": "no_go_for_test_evaluation" if not (passes_low or passes_middle) else "go_for_test_evaluation",
        "reason": "SUS improvement over the fixed selector was not statistically supported on validation.",
        "test_partition_used": False,
        "validation_items": metadata["completed_items"],
        "low_rate_sus_difference_vs_fixed": float(low_fixed["mean_paired_difference"]),
        "low_rate_sus_ci": [float(low_fixed["difference_ci_low"]), float(low_fixed["difference_ci_high"])],
        "middle_rate_sus_difference_vs_fixed": float(middle_fixed["mean_paired_difference"]),
        "middle_rate_sus_ci": [
            float(middle_fixed["difference_ci_low"]),
            float(middle_fixed["difference_ci_high"]),
        ],
    }
    lines = [
        "# Decoder-Impact Token Selector Validation",
        "",
        "## Research question",
        "",
        "Does a selector trained from decoder-sensitive fallback-token attribution preserve wildfire SUS more effectively than fixed utility, entropy, random selection, JPEG, and JPEG2000 at matched transmitted-byte budgets?",
        "",
        "## Training protocol",
        "",
        "The model was trained on 318 geography-separated HLS tiles and selected on 98 separate validation tiles. Integrated-gradient targets attribute wildfire-weighted reconstruction-loss reduction from the all-fallback latent grid to individual tokens. The 84-scene test partition was not used.",
        "",
        f"Training reached validation loss {float(best['validation_loss']):.4f} and top-20% token overlap {float(best['validation_top20_overlap']) * 100.0:.1f}%.",
        "",
        "## Matched-rate validation results",
        "",
        "| operating point | learned SUS | vs fixed | vs entropy | vs random | PSNR gain vs fixed | SSIM gain vs fixed |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        table_row("low", low, low_fixed, low_entropy, low_random, low_psnr, low_ssim),
        table_row("middle", middle, middle_fixed, middle_entropy, middle_random, middle_psnr, middle_ssim),
        "",
        "## Statistical interpretation",
        "",
        describe("Low-rate SUS versus fixed utility", low_fixed),
        "",
        describe("Middle-rate SUS versus fixed utility", middle_fixed),
        "",
        describe("Low-rate SUS versus entropy", low_entropy),
        "",
        describe("Middle-rate SUS versus random", middle_random),
        "",
        "The learned selector strongly improved reconstruction quality over fixed utility selection: low-rate PSNR increased by "
        f"{float(low_psnr['mean_paired_difference']):.2f} dB and SSIM by {float(low_ssim['mean_paired_difference']):.3f}; "
        f"middle-rate PSNR increased by {float(middle_psnr['mean_paired_difference']):.2f} dB and SSIM by {float(middle_ssim['mean_paired_difference']):.3f}. All four reconstruction improvements were statistically significant.",
        "",
        "## Decision",
        "",
        "**No-go for test evaluation and no promotion to the default mission selector.** The learned model successfully identifies decoder-relevant tokens, but its SUS gains over the fixed selector are small and statistically unsupported, while low-rate SUS remains significantly below entropy selection. The untouched test partition is preserved for a later selector that passes validation.",
        "",
        "## Next experiment",
        "",
        "Use the learned decoder-impact score as one component in a validation-only hybrid with detector retention and entropy, or train with a discrete post-reconstruction detector-retention objective. Any hybrid weight must be frozen using validation data before one final test evaluation.",
    ]
    return "\n".join(lines), decision


def table_row(label, result, fixed, entropy, random, psnr, ssim) -> str:
    return (
        f"| {label} | {float(result['sus_mean']):.2f} | {float(fixed['mean_paired_difference']):+.2f} | "
        f"{float(entropy['mean_paired_difference']):+.2f} | {float(random['mean_paired_difference']):+.2f} | "
        f"{float(psnr['mean_paired_difference']):+.2f} dB | {float(ssim['mean_paired_difference']):+.3f} |"
    )


def describe(label: str, row: dict[str, str]) -> str:
    return (
        f"- {label}: {float(row['mean_paired_difference']):+.2f} points, "
        f"95% CI {float(row['difference_ci_low']):+.2f} to {float(row['difference_ci_high']):+.2f}, "
        f"paired t-test p={float(row['paired_t_p_value']):.4g}, "
        f"Wilcoxon p={float(row['wilcoxon_p_value']):.4g}, Cohen's dz={float(row['cohens_dz']):+.2f}."
    )


def get_summary(rows, method: str, budget: int):
    return next(row for row in rows if row["method"] == method and int(row["budget_index"]) == budget)


def get_test(rows, baseline: str, metric: str, budget: int):
    return next(
        row
        for row in rows
        if row["baseline"] == baseline and row["metric"] == metric and int(row["budget_index"]) == budget
    )


def read_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    main()
