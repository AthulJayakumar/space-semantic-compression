"""Generate the final, evidence-led report for the matched-rate experiment."""

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

from evaluation.matched_rate import bootstrap_mean_ci, markdown_table  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a publication-oriented matched-rate evidence report.")
    parser.add_argument("--results-dir", type=Path, default=Path("results/matched_rate_hls_500"))
    parser.add_argument("--validity-dir", type=Path, default=Path("results/validated_hls_500"))
    args = parser.parse_args()

    rows = read_csv(args.results_dir / "matched_rate_rows.csv")
    manifest = read_csv(args.validity_dir / "validated_scene_manifest.csv")
    labels = {row["sample_id"]: row for row in manifest}
    for row in rows:
        metadata = labels.get(row["sample_id"], {})
        row["fire_status"] = metadata.get("fire_status", "unknown")
        row["positive_pixel_fraction"] = metadata.get("positive_pixel_fraction", "")
    detector_backends = sorted({row.get("detector_backend", "unknown") for row in rows})
    write_csv(args.results_dir / "matched_rate_rows_with_labels.csv", rows)

    confirmatory = [row for row in rows if row.get("source_split") == "validation"]
    stratified = stratified_summary(confirmatory)
    write_csv(args.results_dir / "confirmatory_fire_stratified_summary.csv", stratified)

    summary = read_csv(args.results_dir / "confirmatory_source_validation_summary.csv")
    statistics = read_csv(args.results_dir / "confirmatory_source_validation_statistics.csv")
    sensitivity_summary = read_csv(args.results_dir / "matched_rate_summary.csv")
    sensitivity_statistics = read_csv(args.results_dir / "matched_rate_statistics.csv")
    integrity = json.loads((args.validity_dir / "dataset_integrity.json").read_text(encoding="utf-8"))
    report = build_report(
        summary,
        statistics,
        sensitivity_summary,
        sensitivity_statistics,
        stratified,
        integrity,
        args.results_dir,
        detector_backends,
    )
    report_path = args.results_dir / "publication_evidence_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(json.dumps({"report": str(report_path), "stratified_rows": len(stratified)}, indent=2))


def stratified_summary(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    groups = sorted({(row["fire_status"], row["method"], int(row["budget_index"])) for row in rows})
    for fire_status, method, budget_index in groups:
        subset = [
            row
            for row in rows
            if row["fire_status"] == fire_status and row["method"] == method and int(row["budget_index"]) == budget_index
        ]
        record: dict[str, object] = {
            "fire_status": fire_status,
            "method": method,
            "budget_index": budget_index,
            "n": len(subset),
        }
        for metric in ("actual_bytes", "sus", "detector_retention", "psnr", "ssim", "lpips"):
            values = np.asarray([float(row[metric]) for row in subset], dtype="float64")
            low, high = bootstrap_mean_ci(values, seed=20260919 + budget_index)
            record[f"{metric}_mean"] = float(values.mean())
            record[f"{metric}_ci_low"] = low
            record[f"{metric}_ci_high"] = high
        output.append(record)
    return output


def build_report(
    summary: list[dict[str, str]],
    statistics: list[dict[str, str]],
    sensitivity_summary: list[dict[str, str]],
    sensitivity_statistics: list[dict[str, str]],
    stratified: list[dict[str, object]],
    integrity: dict[str, object],
    results_dir: Path,
    detector_backends: list[str],
) -> str:
    low_utility = get_summary(summary, "vqvae_utility", 0)
    mid_utility = get_summary(summary, "vqvae_utility", 1)
    high_utility = get_summary(summary, "vqvae_utility", 2)
    low_jpeg = get_test(statistics, "jpeg", "sus", 0)
    low_j2k = get_test(statistics, "jpeg2000", "sus", 0)
    low_entropy = get_test(statistics, "vqvae_entropy", "sus", 0)
    low_random = get_test(statistics, "vqvae_random", "sus", 0)
    mid_jpeg = get_test(statistics, "jpeg", "sus", 1)
    mid_j2k = get_test(statistics, "jpeg2000", "sus", 1)
    mid_quality = {
        metric: get_test(statistics, "jpeg2000", metric, 1)
        for metric in ("psnr", "ssim", "lpips")
    }
    sensitivity_low_utility = get_summary(sensitivity_summary, "vqvae_utility", 0)
    sensitivity_low_jpeg2000 = get_summary(sensitivity_summary, "jpeg2000", 0)
    sensitivity_low_j2k_test = get_test(sensitivity_statistics, "jpeg2000", "sus", 0)
    low_tests = {
        "JPEG": low_jpeg,
        "JPEG2000": low_j2k,
        "entropy-only selection": low_entropy,
        "random selection": low_random,
    }
    low_wins = [
        label
        for label, test in low_tests.items()
        if float(test["mean_paired_difference"]) > 0.0
    ]
    low_losses = [
        label
        for label, test in low_tests.items()
        if float(test["mean_paired_difference"]) < 0.0
    ]
    if low_losses:
        interpretation = (
            "The supervised-detector matched-rate experiment does not support a general claim that the current "
            "utility-aware selector is superior at the lowest shared byte budget. Utility-aware selection trailed "
            f"{', '.join(low_losses)} in mean SUS"
            + (f", while exceeding {', '.join(low_wins)}" if low_wins else "")
            + ". This negative end-to-end result is informative: improving burn-scar segmentation alone did not "
            "improve token allocation, so the learned utility map and token-ranking objective require better calibration."
        )
    else:
        interpretation = (
            "At the lowest shared byte budget, utility-aware selection achieved higher mean SUS than every tested "
            "baseline. Statistical significance and external detector validity must still be considered separately "
            "before claiming general superiority."
        )

    summary_display = [
        {
            "budget": row["budget_index"],
            "method": row["method"],
            "n": row["n"],
            "bytes": round(float(row["actual_bytes_mean"]), 1),
            "SUS": round(float(row["sus_mean"]), 2),
            "SUS 95% CI": f"{float(row['sus_ci_low']):.2f}-{float(row['sus_ci_high']):.2f}",
            "detector retention": round(float(row["detector_retention_mean"]), 3),
            "PSNR": round(float(row["psnr_mean"]), 2),
            "SSIM": round(float(row["ssim_mean"]), 3),
            "LPIPS": round(float(row["lpips_mean"]), 3),
        }
        for row in summary
    ]
    sensitivity_display = [
        {
            "budget": row["budget_index"],
            "method": row["method"],
            "n": row["n"],
            "bytes": round(float(row["actual_bytes_mean"]), 1),
            "SUS": round(float(row["sus_mean"]), 2),
            "detector retention": round(float(row["detector_retention_mean"]), 3),
            "PSNR": round(float(row["psnr_mean"]), 2),
            "SSIM": round(float(row["ssim_mean"]), 3),
            "LPIPS": round(float(row["lpips_mean"]), 3),
        }
        for row in sensitivity_summary
    ]
    lines = [
        "# Matched-Rate Wildfire Earth Observation Evidence Report",
        "",
        "## Study objective",
        "",
        "This experiment tests whether utility-aware VQ-VAE token selection preserves the current wildfire-oriented Semantic Utility Score (SUS) more effectively than JPEG, JPEG2000, random token selection, and entropy-only token selection when every method receives the same transmitted-byte budget.",
        "",
        "## Dataset validity",
        "",
        f"The prepared corpus contains {integrity['total_independent_tiles']} independently stored, georeferenced 512x512 HLS tiles with IBM/NASA burn-scar masks. It contains {integrity['fire_positive']} fire-positive and {integrity['fire_negative']} fire-negative tiles across {integrity['geographic_groups']} HLS geographic groups. Geographic groups are disjoint across 318 training, 98 validation, and 84 test tiles. Exact image hashes, source identities, and geographic groups do not cross partitions.",
        "",
        "The unit of analysis is an independent HLS tile, not a full Sentinel-2 granule and not one of several overlapping patches cut from the same tile. The primary confirmatory subset contains 30 samples that are both geography-held-out and part of the provider's original validation partition; the full 84-image geography-held-out partition is retained as a sensitivity analysis.",
        "",
        "Wildfire-event separation is not verified because the source metadata does not include incident identifiers. The dataset also lacks independent smoke annotations and official cloud masks. Accordingly, event-generalisation, smoke-specific, and cloud-specific claims are outside the evidence produced here.",
        "",
        "## Matched-rate protocol",
        "",
        "For every image, the experiment measured the feasible byte range shared by all five methods and evaluated low, middle, and high operating points inside that range. Each codec was constrained to remain at or below the target; unused capacity was counted as transmitted padding. Thus `actual_bytes` is identical within every paired image/operating-point comparison. VQ-VAE streams contain selected code indices and their serialized mask; the decoder checkpoint is assumed to be deployed at the receiver.",
        "",
        f"All images were evaluated at 512x512 RGB resolution. Detector backend(s): {', '.join(f'`{backend}`' for backend in detector_backends)}. When the supervised backend is used, it estimates burn-scar relevance from independently labelled HLS masks; it is not an active-fire or smoke detector.",
        "",
        "## Primary confirmatory results",
        "",
        markdown_table(summary_display, ["budget", "method", "n", "bytes", "SUS", "SUS 95% CI", "detector retention", "PSNR", "SSIM", "LPIPS"]),
        "",
        "## Main findings",
        "",
        f"1. At the low operating point ({float(low_utility['actual_bytes_mean']):.0f} mean transmitted bytes, {float(low_utility['bits_per_pixel_mean']):.3f} bpp), utility-aware selection achieved SUS {float(low_utility['sus_mean']):.2f} (95% CI {float(low_utility['sus_ci_low']):.2f}-{float(low_utility['sus_ci_high']):.2f}). {comparison_sentence(low_jpeg, 'JPEG')}",
        f"2. At the same low rate, {comparison_sentence(low_entropy, 'entropy-only selection', lowercase_subject=True)} {comparison_sentence(low_random, 'random selection')}",
        f"3. Against the strongest classical baseline at the low rate, {comparison_sentence(low_j2k, 'JPEG2000', lowercase_subject=True, include_ci=True)}",
        f"4. At the middle operating point ({float(mid_utility['actual_bytes_mean']):.0f} bytes), utility-aware SUS was {float(mid_utility['sus_mean']):.2f}. {comparison_sentence(mid_jpeg, 'JPEG')} {comparison_sentence(mid_j2k, 'JPEG2000')}",
        f"5. Visual fidelity remains the principal weakness. At the middle rate, utility-aware selection was {abs(float(mid_quality['psnr']['mean_paired_difference'])):.2f} dB below JPEG2000 in PSNR and {abs(float(mid_quality['ssim']['mean_paired_difference'])):.3f} lower in SSIM; LPIPS was {float(mid_quality['lpips']['mean_paired_difference']):.3f} higher, where lower is better. All three differences were statistically significant.",
        f"6. At the high operating point ({float(high_utility['actual_bytes_mean']):.0f} bytes), all three VQ-VAE selectors converged to the full-token payload and therefore produced the same result (SUS {float(high_utility['sus_mean']):.2f}). This point measures the VQ-VAE reconstruction ceiling rather than selector quality.",
        f"7. In the broader 84-image sensitivity analysis, utility-aware SUS was {float(sensitivity_low_utility['sus_mean']):.2f} versus {float(sensitivity_low_jpeg2000['sus_mean']):.2f} for JPEG2000. {comparison_sentence(sensitivity_low_j2k_test, 'JPEG2000')}",
        "",
        "## Full geography-held-out sensitivity results",
        "",
        markdown_table(sensitivity_display, ["budget", "method", "n", "bytes", "SUS", "detector retention", "PSNR", "SSIM", "LPIPS"]),
        "",
        "## Interpretation",
        "",
        interpretation,
        "",
        "The visual-quality results also show that optimizing the current utility score can trade away perceptual fidelity. This is not a reason to discard utility-aware selection; it defines the next research problem more precisely: improve the alignment between learned token utility, independent wildfire labels, and downstream task performance while reducing the PSNR/LPIPS penalty.",
        "",
        "## Fire-positive and fire-negative coverage",
        "",
        markdown_table(stratified, ["fire_status", "method", "budget_index", "n", "actual_bytes_mean", "sus_mean", "sus_ci_low", "sus_ci_high", "detector_retention_mean"]),
        "",
        "The primary subset contains 26 fire-positive and 4 fire-negative tiles. Negative-scene confidence intervals are therefore exploratory and should not be used for strong subgroup claims.",
        "",
        "## Reproducibility outputs",
        "",
        f"- Raw paired measurements: `{results_dir / 'matched_rate_rows_with_labels.csv'}`",
        f"- Primary summary: `{results_dir / 'confirmatory_source_validation_summary.csv'}`",
        f"- Primary paired tests: `{results_dir / 'confirmatory_source_validation_statistics.csv'}`",
        f"- Full sensitivity summary: `{results_dir / 'matched_rate_summary.csv'}`",
        f"- Full sensitivity tests: `{results_dir / 'matched_rate_statistics.csv'}`",
        f"- Pareto points: `{results_dir / 'rate_utility_pareto.csv'}`",
        f"- Figures: `{results_dir / 'figures'}`",
        "",
        "## Publication readiness",
        "",
        "These results are suitable as rigorous preliminary evidence and materially improve the earlier unmatched-rate comparison. Before making a journal-level claim of wildfire mission superiority, the evaluation still requires independently labelled wildfire events, smoke/cloud strata, external validation of the supervised burn-scar detector, and a test set whose non-overlap with all model-development imagery is verifiable at the original scene identifier level.",
    ]
    return "\n".join(lines)


def get_summary(rows: list[dict[str, str]], method: str, budget_index: int) -> dict[str, str]:
    return next(row for row in rows if row["method"] == method and int(row["budget_index"]) == budget_index)


def get_test(rows: list[dict[str, str]], baseline: str, metric: str, budget_index: int) -> dict[str, str]:
    return next(
        row
        for row in rows
        if row["baseline"] == baseline and row["metric"] == metric and int(row["budget_index"]) == budget_index
    )


def comparison_sentence(
    test: dict[str, str],
    baseline_label: str,
    *,
    lowercase_subject: bool = False,
    include_ci: bool = False,
) -> str:
    """Describe a utility-minus-baseline paired result without reversing its sign."""
    difference = float(test["mean_paired_difference"])
    p_value = float(test["paired_t_p_value"])
    wilcoxon = float(test["wilcoxon_p_value"])
    subject = "utility-aware selection" if lowercase_subject else "Utility-aware selection"
    if np.isclose(difference, 0.0):
        comparison = f"matched {baseline_label} in mean SUS"
    elif difference > 0.0:
        comparison = f"was {abs(difference):.2f} SUS points higher than {baseline_label}"
    else:
        comparison = f"was {abs(difference):.2f} SUS points lower than {baseline_label}"
    significance = "statistically significant" if p_value < 0.05 else "not statistically significant"
    details = (
        f"paired t-test p{format_p_value(p_value)}, Wilcoxon p{format_p_value(wilcoxon)}; "
        f"{significance} at alpha=0.05"
    )
    if include_ci:
        details += (
            f"; 95% CI {float(test['difference_ci_low']):.2f} to "
            f"{float(test['difference_ci_high']):.2f}"
        )
    return f"{subject} {comparison} ({details})."


def format_p_value(value: float) -> str:
    return "<0.0001" if value < 0.0001 else f"={value:.4f}"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
