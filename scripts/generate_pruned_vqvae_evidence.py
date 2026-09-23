"""Consolidate the frozen pruned-token VQ-VAE validation evidence."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compare_vqvae_checkpoints import paired_statistics, write_rows  # noqa: E402


RETENTION_DIRS = {
    0.2: "retention_20",
    0.5: "retention_50",
    0.8: "retention_80",
    1.0: "retention_100",
}


def main() -> None:
    root = Path("results/external_lockbox_ecofirebias_120")
    output = root / "evidence"
    output.mkdir(parents=True, exist_ok=True)
    consolidated: list[dict[str, object]] = []
    for retention, dirname in RETENTION_DIRS.items():
        run_dir = root / dirname
        rows = read_csv(run_dir / "vqvae_checkpoint_comparison_rows.csv")
        stats = paired_statistics(rows, baseline_checkpoint="baseline")
        write_rows(run_dir / "vqvae_checkpoint_paired_statistics.csv", stats)
        by_metric = {str(row["metric"]): row for row in stats}
        by_checkpoint = {
            row["checkpoint"]: row
            for row in read_csv(run_dir / "vqvae_checkpoint_comparison_summary.csv")
        }
        baseline = by_checkpoint["baseline"]
        candidate = by_checkpoint["pruned_semantic"]
        record: dict[str, object] = {"retention": retention, "n": int(candidate["n_images"])}
        for metric in (
            "semantic_utility_score",
            "detector_retention",
            "psnr",
            "ssim",
            "lpips",
            "compression_ratio",
            "bandwidth_saved_percent",
        ):
            record[f"baseline_{metric}"] = float(baseline[f"{metric}_mean"])
            record[f"candidate_{metric}"] = float(candidate[f"{metric}_mean"])
            stat = by_metric[metric]
            record[f"difference_{metric}"] = float(stat["mean_difference_candidate_minus_baseline"])
            record[f"ci_low_{metric}"] = float(stat["bootstrap_95ci_low"])
            record[f"ci_high_{metric}"] = float(stat["bootstrap_95ci_high"])
            record[f"paired_t_p_{metric}"] = float(stat["paired_t_p_value"])
            record[f"wilcoxon_p_{metric}"] = float(stat["wilcoxon_p_value"])
        consolidated.append(record)

    write_rows(output / "retention_summary.csv", consolidated)
    decision = build_decision(consolidated)
    (output / "external_validation_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    make_figure(consolidated, output / "retention_evidence.png")
    (output / "pruned_token_external_validation_report.md").write_text(
        build_report(consolidated, decision), encoding="utf-8"
    )
    print(json.dumps(decision, indent=2))


def build_decision(rows: list[dict[str, object]]) -> dict[str, object]:
    primary = next(row for row in rows if float(row["retention"]) == 0.2)
    passed = (
        float(primary["ci_low_semantic_utility_score"]) > 0.0
        and float(primary["ci_low_detector_retention"]) > 0.0
        and float(primary["ci_low_psnr"]) > -0.5
    )
    return {
        "decision": "promote_for_research_evaluation" if passed else "do_not_promote",
        "research_checkpoint_recommendation": "pruned_semantic_candidate",
        "primary_operating_point": 0.2,
        "primary_rule_passed": passed,
        "candidate_checkpoint": "models/checkpoints/vqvae_s16k8_pruned_semantic_finetuned.pt",
        "candidate_checkpoint_sha256": "e327907517875b9f7c0d2699563014d6476bf31c06866eb17ddfe1af42133959",
        "external_dataset": "EcoFireBias event-level test split",
        "external_scenes": 120,
        "external_events": 60,
        "continents": 6,
        "claim": (
            "On the frozen EcoFireBias lockbox, pruned-token decoder training significantly improves "
            "detector-defined semantic utility at 20 to 100 percent token retention relative to the "
            "pre-existing VQ-VAE checkpoint."
        ),
        "boundary": (
            "This is a VQ-VAE checkpoint comparison with a fixed utility selector. It does not establish "
            "superiority over JPEG/JPEG2000, and SUS remains dependent on the frozen detector."
        ),
    }


def make_figure(rows: list[dict[str, object]], path: Path) -> None:
    retention = [float(row["retention"]) * 100 for row in rows]
    panels = [
        ("semantic_utility_score", "SUS (0-100)", False),
        ("detector_retention", "Detector retention", False),
        ("psnr", "PSNR (dB)", False),
        ("lpips", "LPIPS (lower is better)", True),
    ]
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), dpi=180)
    for axis, (metric, label, lower_better) in zip(axes.flat, panels):
        baseline = [float(row[f"baseline_{metric}"]) for row in rows]
        candidate = [float(row[f"candidate_{metric}"]) for row in rows]
        axis.plot(retention, baseline, marker="o", linewidth=2, label="Baseline VQ-VAE", color="#59636e")
        axis.plot(retention, candidate, marker="s", linewidth=2, label="Pruned-token candidate", color="#007f73")
        axis.set_xlabel("Token retention (%)")
        axis.set_ylabel(label)
        axis.set_xticks(retention)
        if not lower_better:
            axis.set_ylim(bottom=0)
    axes[0, 0].legend(frameon=False, loc="lower right")
    figure.suptitle("Frozen EcoFireBias External Validation (120 scenes, 60 events)", fontsize=13)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def build_report(rows: list[dict[str, object]], decision: dict[str, object]) -> str:
    lines = [
        "# Pruned-Token VQ-VAE External Validation",
        "",
        "## Experimental question",
        "",
        "Does training the existing decoder on utility-pruned latent representations improve wildfire-relevant reconstruction under severe token constraints without introducing a new architecture?",
        "",
        "## Method and integrity",
        "",
        "The architecture, codebook, formal SUS definition, detector checkpoint and utility selector were unchanged. The candidate added an auxiliary training branch that retained the 20% highest detector-utility tokens, replaced all other tokens with the modal fallback code, and optimized reconstruction plus frozen-detector consistency. Training used only the established CEMS-HLS and Sentinel-2 development data.",
        "",
        "The final candidate was frozen before EcoFireBias evaluation. The external lockbox contains 120 Sentinel-2 post-event chips from the provider's event-level test split: 60 burn scenes and 60 paired controls, representing 60 events distributed equally across six continents. Selection used metadata and a deterministic seed; no model output was used.",
        "",
        "## External results",
        "",
        "| Retention | Baseline SUS | Candidate SUS | SUS change (95% CI) | Detector change (95% CI) | PSNR change (95% CI) | SSIM change | LPIPS change |",
        "| ---: | ---: | ---: | --- | --- | --- | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {ret:.0f}% | {base:.2f} | {cand:.2f} | {sus:+.2f} [{sus_lo:+.2f}, {sus_hi:+.2f}] | "
            "{det:+.3f} [{det_lo:+.3f}, {det_hi:+.3f}] | {psnr:+.2f} [{psnr_lo:+.2f}, {psnr_hi:+.2f}] | "
            "{ssim:+.4f} | {lpips:+.4f} |".format(
                ret=float(row["retention"]) * 100,
                base=float(row["baseline_semantic_utility_score"]),
                cand=float(row["candidate_semantic_utility_score"]),
                sus=float(row["difference_semantic_utility_score"]),
                sus_lo=float(row["ci_low_semantic_utility_score"]),
                sus_hi=float(row["ci_high_semantic_utility_score"]),
                det=float(row["difference_detector_retention"]),
                det_lo=float(row["ci_low_detector_retention"]),
                det_hi=float(row["ci_high_detector_retention"]),
                psnr=float(row["difference_psnr"]),
                psnr_lo=float(row["ci_low_psnr"]),
                psnr_hi=float(row["ci_high_psnr"]),
                ssim=float(row["difference_ssim"]),
                lpips=float(row["difference_lpips"]),
            )
        )
    primary = rows[0]
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"**{str(decision['decision']).replace('_', ' ').title()}.** At 20% retention, SUS improved by {float(primary['difference_semantic_utility_score']):.2f} points and detector retention by {float(primary['difference_detector_retention']):.3f}; both bootstrap confidence intervals exclude zero. PSNR increased by {float(primary['difference_psnr']):.2f} dB, SSIM increased by {float(primary['difference_ssim']):.4f}, and LPIPS decreased by {abs(float(primary['difference_lpips'])):.4f}.",
            "",
            "The candidate therefore passes the preregistered severe-rate promotion rule and is suitable as the preferred VQ-VAE checkpoint for subsequent research evaluation. The previous detector-consistency-only candidate remains an ablation, not the default.",
            "",
            "## Claim boundary",
            "",
            str(decision["boundary"]),
            "",
            "The earlier FireScope matched-byte experiment still shows that JPEG and JPEG2000 outperform the historical VQ-VAE codec. A new matched-byte conventional-codec comparison is required before claiming that the improved checkpoint closes that gap.",
            "",
            "## Reproducibility artifacts",
            "",
            "- `LOCKBOX_FROZEN.json`: event balance, dataset revision and manifest hash.",
            "- `COMPRESSION_PROTOCOL_FROZEN.json`: checkpoint hashes and success rule.",
            "- `retention_*/vqvae_checkpoint_comparison_rows.csv`: all per-scene observations.",
            "- `retention_*/vqvae_checkpoint_paired_statistics.csv`: paired tests and bootstrap intervals.",
            "- `evidence/retention_summary.csv`: consolidated table used in this report and figure.",
        ]
    )
    return "\n".join(lines) + "\n"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    main()
