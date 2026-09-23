"""Compare detector and selector iterations without conflating changed SUS backends."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.matched_rate import bootstrap_mean_ci  # noqa: E402
from evaluation.statistics import StatisticalValidator  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare successive semantic detector and selector experiments.")
    parser.add_argument("--heuristic-run", type=Path, default=Path("results/matched_rate_hls_500"))
    parser.add_argument(
        "--supervised-run", type=Path, default=Path("results/matched_rate_hls_500_supervised_detector")
    )
    parser.add_argument(
        "--calibrated-run", type=Path, default=Path("results/matched_rate_hls_500_calibrated_selector")
    )
    parser.add_argument(
        "--detector-results", type=Path, default=Path("results/burn_scar_utility_detector")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/selector_iteration_comparison"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    runs = {
        "heuristic_detector_original_selector": read_csv(args.heuristic_run / "matched_rate_rows.csv"),
        "supervised_detector_original_selector": read_csv(args.supervised_run / "matched_rate_rows.csv"),
        "supervised_detector_calibrated_selector": read_csv(args.calibrated_run / "matched_rate_rows.csv"),
    }
    overview = build_overview(runs)
    selector_tests = compare_same_detector_selectors(
        runs["supervised_detector_original_selector"],
        runs["supervised_detector_calibrated_selector"],
    )
    calibrated_statistics = read_csv(args.calibrated_run / "matched_rate_statistics.csv")
    detector_summary = read_csv(args.detector_results / "detector_comparison_summary.csv")
    detector_statistics = read_csv(args.detector_results / "detector_comparison_statistics.csv")

    write_csv(args.output_dir / "iteration_overview.csv", overview)
    write_csv(args.output_dir / "selector_generalisation_tests.csv", selector_tests)
    report = build_report(overview, selector_tests, calibrated_statistics, detector_summary, detector_statistics)
    (args.output_dir / "selector_iteration_report.md").write_text(report, encoding="utf-8")
    print(args.output_dir / "selector_iteration_report.md")


def build_overview(runs: dict[str, list[dict[str, str]]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for run_name, rows in runs.items():
        for cohort, cohort_rows in (
            ("full_test", rows),
            ("source_validation_confirmatory", [row for row in rows if row["source_split"] == "validation"]),
        ):
            for budget in (0, 1, 2):
                subset = [
                    row
                    for row in cohort_rows
                    if row["method"] == "vqvae_utility" and int(row["budget_index"]) == budget
                ]
                output.append(
                    {
                        "run": run_name,
                        "cohort": cohort,
                        "budget_index": budget,
                        "n": len(subset),
                        **{
                            f"{metric}_mean": mean(subset, metric)
                            for metric in ("actual_bytes", "sus", "detector_retention", "psnr", "ssim", "lpips")
                        },
                    }
                )
    return output


def compare_same_detector_selectors(
    original_rows: list[dict[str, str]], calibrated_rows: list[dict[str, str]]
) -> list[dict[str, object]]:
    validator = StatisticalValidator()
    output: list[dict[str, object]] = []
    for cohort in ("full_test", "source_validation_confirmatory"):
        original = original_rows if cohort == "full_test" else [
            row for row in original_rows if row["source_split"] == "validation"
        ]
        calibrated = calibrated_rows if cohort == "full_test" else [
            row for row in calibrated_rows if row["source_split"] == "validation"
        ]
        for budget in (0, 1, 2):
            for metric in ("sus", "detector_retention", "psnr", "ssim", "lpips"):
                left = {
                    row["sample_id"]: float(row[metric])
                    for row in original
                    if row["method"] == "vqvae_utility" and int(row["budget_index"]) == budget
                }
                right = {
                    row["sample_id"]: float(row[metric])
                    for row in calibrated
                    if row["method"] == "vqvae_utility" and int(row["budget_index"]) == budget
                }
                ids = sorted(set(left) & set(right))
                baseline = [left[sample_id] for sample_id in ids]
                candidate = [right[sample_id] for sample_id in ids]
                differences = np.asarray(candidate) - np.asarray(baseline)
                t_result = validator.paired_t_test(baseline, candidate)
                w_result = validator.wilcoxon(baseline, candidate)
                low, high = bootstrap_mean_ci(differences, seed=20260921)
                output.append(
                    {
                        "cohort": cohort,
                        "budget_index": budget,
                        "metric": metric,
                        "n": len(ids),
                        "original_mean": float(np.mean(baseline)),
                        "calibrated_mean": float(np.mean(candidate)),
                        "mean_paired_difference": float(differences.mean()),
                        "difference_ci_low": low,
                        "difference_ci_high": high,
                        "paired_t_p_value": t_result.p_value,
                        "wilcoxon_p_value": w_result.p_value,
                        "cohens_dz": t_result.effect_size,
                    }
                )
    return output


def build_report(
    overview: list[dict[str, object]],
    selector_tests: list[dict[str, object]],
    calibrated_statistics: list[dict[str, str]],
    detector_summary: list[dict[str, str]],
    detector_statistics: list[dict[str, str]],
) -> str:
    dice = next(row for row in detector_statistics if row["metric"] == "dice")
    balanced = next(row for row in detector_statistics if row["metric"] == "balanced_accuracy")
    low = get_selector_test(selector_tests, "full_test", 0, "sus")
    middle = get_selector_test(selector_tests, "full_test", 1, "sus")
    low_baselines = [
        row
        for row in calibrated_statistics
        if int(row["budget_index"]) == 0 and row["metric"] == "sus"
    ]
    calibrated_low = next(
        row
        for row in overview
        if row["run"] == "supervised_detector_calibrated_selector"
        and row["cohort"] == "full_test"
        and int(row["budget_index"]) == 0
    )
    lines = [
        "# Detector and Selector Iteration Report",
        "",
        "## Scope",
        "",
        "This report separates perception quality from end-to-end transmission quality. Absolute SUS values from the heuristic-detector run and supervised-detector runs are not treated as directly interchangeable because the detector used inside SUS changed. Selector comparisons between the two supervised runs are directly paired because their detector, dataset, codec, and byte budgets are identical.",
        "",
        "## Detector improvement",
        "",
        f"The supervised burn-scar model improved mean per-image Dice by {float(dice['mean_difference']):.3f} "
        f"(95% CI {float(dice['difference_ci_low']):.3f} to {float(dice['difference_ci_high']):.3f}; "
        f"paired t-test p={float(dice['paired_t_p_value']):.6f}). Balanced accuracy improved by "
        f"{float(balanced['mean_difference']):.3f}. This validates the detector improvement, but not the compression selector.",
        "",
        "## Selector calibration and test generalisation",
        "",
        f"At the severe low-rate point, context calibration changed full-test SUS from {float(low['original_mean']):.2f} "
        f"to {float(low['calibrated_mean']):.2f}, a paired difference of {float(low['mean_paired_difference']):+.2f} "
        f"(95% CI {float(low['difference_ci_low']):+.2f} to {float(low['difference_ci_high']):+.2f}; "
        f"p={float(low['paired_t_p_value']):.4f}). The validation gain therefore did not generalise at the primary low-rate operating point.",
        "",
        f"At the middle-rate point, SUS increased from {float(middle['original_mean']):.2f} to "
        f"{float(middle['calibrated_mean']):.2f}, a paired improvement of {float(middle['mean_paired_difference']):+.2f} "
        f"(95% CI {float(middle['difference_ci_low']):+.2f} to {float(middle['difference_ci_high']):+.2f}; "
        f"p={float(middle['paired_t_p_value']):.6f}). This is a secondary positive result and requires replication.",
        "",
        "## Matched-rate baseline result",
        "",
        f"The calibrated selector achieved full-test low-rate SUS {float(calibrated_low['sus_mean']):.2f} at "
        f"{float(calibrated_low['actual_bytes_mean']):.1f} mean transmitted bytes. Its paired SUS differences versus the matched-rate baselines were:",
        "",
    ]
    for row in low_baselines:
        lines.append(
            f"- {row['baseline']}: {float(row['mean_paired_difference']):+.2f} points "
            f"(paired t-test p={float(row['paired_t_p_value']):.6f})."
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "Do not claim low-rate superiority for the present utility-aware selector and do not replace the default selector on the basis of this experiment. Retain the context-expanded selector as an experimental middle-rate configuration. The next model change should learn token value from downstream loss or token-removal impact, because direct spatial ranking of a segmentation map is not aligned strongly enough with the VQ-VAE latent representation.",
            "",
            "## Scientific value",
            "",
            "The experiment produced a useful negative result: a more accurate semantic detector does not automatically produce a better semantic communication system. The perception model, latent representation, and transmission controller must be trained or calibrated as a coupled decision pipeline.",
        ]
    )
    return "\n".join(lines)


def get_selector_test(
    rows: list[dict[str, object]], cohort: str, budget: int, metric: str
) -> dict[str, object]:
    return next(
        row
        for row in rows
        if row["cohort"] == cohort and int(row["budget_index"]) == budget and row["metric"] == metric
    )


def mean(rows: list[dict[str, str]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows])) if rows else float("nan")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
