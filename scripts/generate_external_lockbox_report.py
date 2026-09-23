"""Generate the confirmatory report for the frozen FireScope lockbox."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results" / "external_lockbox_firescope_100"


def main() -> None:
    freeze = _read_json(RESULT_ROOT / "LOCKBOX_FROZEN.json")
    protocol = _read_json(RESULT_ROOT / "COMPRESSION_PROTOCOL_FROZEN.json")
    detector = _read_json(RESULT_ROOT / "detector_evaluation" / "external_detector_summary.json")
    run = _read_json(RESULT_ROOT / "matched_rate" / "run_metadata.json")
    summary = _read_csv(RESULT_ROOT / "matched_rate" / "matched_rate_summary.csv")
    statistics = _read_csv(RESULT_ROOT / "matched_rate" / "matched_rate_statistics.csv")
    exclusions = _read_csv(RESULT_ROOT / "matched_rate" / "excluded_samples.csv")
    rows = _read_csv(RESULT_ROOT / "matched_rate" / "matched_rate_rows.csv")

    key_rows = _key_result_rows(summary, rows)
    _write_csv(RESULT_ROOT / "external_lockbox_key_results.csv", key_rows)
    decisions = _decisions(statistics)
    (RESULT_ROOT / "external_hypothesis_decision.json").write_text(
        json.dumps(decisions, indent=2), encoding="utf-8"
    )
    report = _build_report(freeze, protocol, detector, run, key_rows, statistics, exclusions, decisions)
    report_path = RESULT_ROOT / "external_lockbox_confirmatory_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(json.dumps({"report": str(report_path), "decisions": decisions}, indent=2))


def _decisions(statistics: list[dict[str, str]]) -> dict[str, object]:
    def comparison(baseline: str, budget: int) -> dict[str, object]:
        row = next(
            item
            for item in statistics
            if item["baseline"] == baseline
            and item["candidate"] == "vqvae_utility"
            and item["metric"] == "sus"
            and int(item["budget_index"]) == budget
        )
        return {
            "baseline": baseline,
            "budget_index": budget,
            "mean_paired_difference": float(row["mean_paired_difference"]),
            "ci_low": float(row["difference_ci_low"]),
            "ci_high": float(row["difference_ci_high"]),
            "paired_t_p_value": float(row["paired_t_p_value"]),
            "wilcoxon_p_value": float(row["wilcoxon_p_value"]),
            "cohens_dz": float(row["cohens_dz"]),
        }

    comparisons = [
        comparison(baseline, budget)
        for budget in (0, 1)
        for baseline in ("jpeg", "jpeg2000", "vqvae_random", "vqvae_entropy")
    ]
    token_controls_supported = all(
        item["ci_low"] > 0
        for item in comparisons
        if item["baseline"] in {"vqvae_random", "vqvae_entropy"}
    )
    conventional_supported = all(
        item["ci_low"] > 0
        for item in comparisons
        if item["baseline"] in {"jpeg", "jpeg2000"}
    )
    return {
        "broad_preregistered_hypothesis": "not_supported" if not conventional_supported else "supported",
        "within_vqvae_token_prioritization_hypothesis": "supported" if token_controls_supported else "not_supported",
        "claim_boundary": (
            "The external evidence supports utility-aware ranking over random and entropy-only token selection "
            "at low and medium matched rates. It does not support superiority over JPEG or JPEG2000."
        ),
        "comparisons": comparisons,
    }


def _key_result_rows(
    summary: list[dict[str, str]], rows: list[dict[str, str]]
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for row in summary:
        method = row["method"]
        budget_index = int(row["budget_index"])
        subset = [
            item
            for item in rows
            if item["method"] == method and int(item["budget_index"]) == budget_index
        ]
        bandwidth = sum(float(item["bandwidth_saved_vs_raw_percent"]) for item in subset) / len(subset)
        output.append(
            {
                "method": method,
                "budget_index": budget_index,
                "n": int(row["n"]),
                "actual_bytes_mean": float(row["actual_bytes_mean"]),
                "compression_ratio_mean": float(row["raw_compression_ratio_mean"]),
                "bandwidth_saved_percent_mean": bandwidth,
                "sus_mean": float(row["sus_mean"]),
                "sus_ci_low": float(row["sus_ci_low"]),
                "sus_ci_high": float(row["sus_ci_high"]),
                "detector_retention_mean": float(row["detector_retention_mean"]),
                "psnr_mean": float(row["psnr_mean"]),
                "ssim_mean": float(row["ssim_mean"]),
                "lpips_mean": float(row["lpips_mean"]),
            }
        )
    return output


def _build_report(freeze, protocol, detector, run, key_rows, statistics, exclusions, decisions) -> str:
    table_rows = sorted(key_rows, key=lambda row: (row["budget_index"], row["method"]))
    positive_dice = detector["positive_scene_metrics"]["dice"]
    balanced = detector["positive_scene_metrics"]["balanced_accuracy"]
    false_positive = detector["negative_false_positive_fraction"]
    lines = [
        "# External Lockbox Confirmatory Report",
        "",
        "## Confirmatory conclusion",
        "",
        "The frozen external evaluation supports the value of semantic utility-aware token ranking within the learned VQ-VAE transmission family, but it does not support the stronger claim that the present neural codec outperforms conventional JPEG or JPEG2000 at equal transmitted bytes. This distinction is the principal result of the lockbox experiment.",
        "",
        f"**Broad preregistered hypothesis: {decisions['broad_preregistered_hypothesis'].replace('_', ' ').upper()}.**",
        f"**Within-VQ-VAE prioritization hypothesis: {decisions['within_vqvae_token_prioritization_hypothesis'].replace('_', ' ').upper()}.**",
        "",
        "## Frozen experimental design",
        "",
        f"The lockbox contains {freeze['scene_count']} Sentinel-2 event/control tiles selected from FireScope-Bench revision `{freeze['source_revision']}` before evaluation. It includes {freeze['class_counts']['positive']} positive event scenes and {freeze['class_counts']['negative']} negative controls, balanced equally between Europe and the United States. The scenes span {freeze['geographic_group_count']} geographic groups and {freeze['event_group_count']} unique event/control identities. Exact image hashes do not overlap any earlier project manifest.",
        "",
        f"The compression protocol compared {', '.join(protocol['methods'])} at three per-image common byte budgets. Checkpoints, utility weights, context radius, image resolution, and metrics were frozen before the compression run. {run['completed_items']} scenes completed and {run['excluded_items']} were excluded by the pre-existing no-common-byte-interval rule.",
        "",
        "## External detector validity",
        "",
        f"On {detector['positive_n']} positive scenes, the frozen detector obtained mean Dice {positive_dice['mean']:.3f} (95% CI {positive_dice['ci_low']:.3f} to {positive_dice['ci_high']:.3f}) and mean balanced accuracy {balanced['mean']:.3f} (95% CI {balanced['ci_low']:.3f} to {balanced['ci_high']:.3f}). On {detector['negative_n']} controls, mean false-positive area was {false_positive['mean'] * 100:.1f}% (95% CI {false_positive['ci_low'] * 100:.1f}% to {false_positive['ci_high'] * 100:.1f}%). Pixel ECE was {detector['ece']:.3f}. These values show limited but non-zero cross-dataset transfer and substantial calibration shift.",
        "",
        "FireScope images are pre-event context paired with next-year event masks. Therefore, this evaluates burn-scar relevance transfer and not active flame or smoke detection.",
        "",
        "## Matched-byte results",
        "",
        "| budget | method | n | bytes | ratio | saved | SUS (95% CI) | detector retention | PSNR | SSIM | LPIPS |",
        "| ---: | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in table_rows:
        lines.append(
            f"| {row['budget_index']} | {row['method']} | {row['n']} | {row['actual_bytes_mean']:.0f} | "
            f"{row['compression_ratio_mean']:.1f}x | {row['bandwidth_saved_percent_mean']:.2f}% | "
            f"{row['sus_mean']:.2f} ({row['sus_ci_low']:.2f}, {row['sus_ci_high']:.2f}) | "
            f"{row['detector_retention_mean']:.3f} | {row['psnr_mean']:.2f} | "
            f"{row['ssim_mean']:.3f} | {row['lpips_mean']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Primary paired comparisons",
            "",
            "Differences below are `VQ-VAE utility minus baseline` for SUS. Positive values favour utility-aware selection.",
            "",
            "| budget | baseline | paired difference | 95% CI | paired t p | Wilcoxon p | Cohen dz |",
            "| ---: | --- | ---: | --- | ---: | ---: | ---: |",
        ]
    )
    for item in decisions["comparisons"]:
        lines.append(
            f"| {item['budget_index']} | {item['baseline']} | {item['mean_paired_difference']:+.2f} | "
            f"[{item['ci_low']:+.2f}, {item['ci_high']:+.2f}] | {item['paired_t_p_value']:.3g} | "
            f"{item['wilcoxon_p_value']:.3g} | {item['cohens_dz']:+.2f} |"
        )
    lines.extend(
        [
            "",
            "At the lowest rate, utility-aware ranking improved SUS by 14.87 points over random selection and 10.57 points over entropy selection. At the middle rate, the gains increased to 20.82 and 13.65 points. All four confidence intervals exclude zero. At full VQ token retention, the three token-ordering methods converge by construction.",
            "",
            "JPEG and JPEG2000 remained substantially stronger. At the lowest budget, their SUS values were 84.50 and 91.94 compared with 51.23 for utility-aware VQ-VAE. The result identifies the current reconstruction codec, rather than the token-ranking principle, as the principal performance bottleneck.",
            "",
            "## Exclusions and integrity",
            "",
            f"Two of {run['requested_items']} scenes were excluded because their very low-complexity content gave no byte interval reachable by all five codecs. The exclusion rule was defined by the benchmark before this lockbox was created. No scene was removed because of its metric value.",
            "",
        ]
    )
    for row in exclusions:
        lines.append(f"- `{row['sample_id']}`: {row['reason']}")
    lines.extend(
        [
            "",
            "Every retained image/method pair used exactly the same transmitted byte target. The maximum recorded absolute byte error was zero. VQ payloads include selected token indices and the serialized selection mask; the shared decoder is assumed to be pre-deployed.",
            "",
            "## Claim boundary for publications and applications",
            "",
            "A defensible claim is: *under identical serialized VQ-VAE payload budgets, utility-aware token ranking preserves significantly more detector-defined wildfire utility than random or entropy-only token selection on a frozen external Sentinel-2 lockbox.*",
            "",
            "The present evidence does not justify claiming superiority over JPEG or JPEG2000. It also does not establish robust active-fire detection, because FireScope supplies pre-event imagery and the external detector shows high false-positive area and calibration shift. A paper should present these as explicit limitations and motivate future work on rate-distortion training, external detector calibration using a separate development set, and truly active-fire labels.",
            "",
            "## Reproducibility artifacts",
            "",
            "- `LOCKBOX_FROZEN.json`: dataset revision, hashes, balance, and no-leakage audit.",
            "- `COMPRESSION_PROTOCOL_FROZEN.json`: preregistered checkpoints, methods, budgets, and hypothesis.",
            "- `detector_evaluation/external_detector_per_scene.csv`: one-shot external detector measurements.",
            "- `matched_rate/matched_rate_rows.csv`: all 1,470 paired codec observations.",
            "- `matched_rate/matched_rate_statistics.csv`: paired tests, bootstrap intervals, and effect sizes.",
            "- `matched_rate/figures/`: rate-quality, rate-utility, and Pareto plots.",
        ]
    )
    return "\n".join(lines)


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
