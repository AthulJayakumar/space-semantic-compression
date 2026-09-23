"""Compare historical and retrained detectors on the official validation split."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.research_wildfire import load_grayscale_image, load_rgb_image  # noqa: E402
from evaluation.matched_rate import bootstrap_mean_ci  # noqa: E402
from evaluation.statistics import StatisticalValidator  # noqa: E402
from scripts.evaluate_burn_scar_detector import confusion_counts, metrics_from_counts  # noqa: E402
from semantic_ai.burn_scar_training import expected_calibration_error  # noqa: E402
from semantic_ai.wildfire_detector import WildfireDetector  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Official-validation comparison of two detector checkpoints.")
    parser.add_argument(
        "--manifest", type=Path, default=Path("results/validated_hls_500/validated_scene_manifest.csv")
    )
    parser.add_argument(
        "--historical", type=Path, default=Path("models/checkpoints/wildfire_utility_segmentation.pt")
    )
    parser.add_argument(
        "--candidate",
        type=Path,
        default=Path("models/checkpoints/wildfire_utility_segmentation_retrained.pt"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/retrained_detector_official_validation"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260923)
    args = parser.parse_args()

    rows = [row for row in read_csv(args.manifest) if row["split"] == "validation"]
    if args.limit is not None:
        rows = rows[: args.limit]
    historical = WildfireDetector(args.historical, output_dir=None, save_visualizations=False)
    candidate = WildfireDetector(args.candidate, output_dir=None, save_visualizations=False)
    historical_config = torch.load(args.historical, map_location="cpu", weights_only=True)["config"]
    candidate_config = torch.load(args.candidate, map_location="cpu", weights_only=True)["config"]
    result_rows = []
    probability_store = {"historical": [], "candidate": [], "target": []}
    for index, row in enumerate(rows, start=1):
        print(f"official detector validation {index}/{len(rows)}: {row['sample_id']}", flush=True)
        image = load_rgb_image(Path(row["image_path"]))
        target = np.asarray(load_grayscale_image(Path(row["mask_path"])), dtype="uint8") > 0
        record = {
            "sample_id": row["sample_id"],
            "geographic_group": row["geographic_group"],
            "positive_pixel_fraction": float(row["positive_pixel_fraction"]),
        }
        for name, detector, config in (
            ("historical", historical, historical_config),
            ("candidate", candidate, candidate_config),
        ):
            probability = detector.detect(image).confidence_map
            threshold = float(config["threshold"])
            metrics = metrics_from_counts(*confusion_counts(probability >= threshold, target))
            for metric, value in metrics.items():
                record[f"{name}_{metric}"] = value
            probability_store[name].append(probability.astype("float32"))
        probability_store["target"].append(target.astype("float32"))
        result_rows.append(record)

    summary = summarize(result_rows, probability_store)
    statistics = paired_statistics(result_rows, args.seed)
    dice_test = next(row for row in statistics if row["metric"] == "dice")
    historical_summary = next(row for row in summary if row["detector"] == "historical")
    candidate_summary = next(row for row in summary if row["detector"] == "candidate")
    go = (
        float(dice_test["difference_ci_low"]) > 0.0
        and float(candidate_summary["ece"]) <= float(historical_summary["ece"])
    )
    decision = {
        "decision": "go_for_future_compression_validation" if go else "no_go_for_future_compression_validation",
        "partition": "official 98-scene geographic validation",
        "test_partition_used": False,
        "candidate_was_selected_on_this_partition": False,
        "historical_checkpoint_was_selected_on_this_partition": True,
        "go_rule": "paired scene Dice bootstrap CI above zero and candidate ECE no worse than historical checkpoint",
        "historical": historical_summary,
        "candidate": candidate_summary,
        "paired_dice": dice_test,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "official_validation_per_scene.csv", result_rows)
    write_csv(args.output_dir / "official_validation_summary.csv", summary)
    write_csv(args.output_dir / "official_validation_statistics.csv", statistics)
    (args.output_dir / "detector_validation_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    (args.output_dir / "detector_validation_report.md").write_text(build_report(decision), encoding="utf-8")
    print(json.dumps(decision, indent=2))


def summarize(rows, probability_store):
    output = []
    targets = np.concatenate([value.reshape(-1) for value in probability_store["target"]])
    for detector in ("historical", "candidate"):
        record = {"detector": detector, "n": len(rows)}
        for metric in ("dice", "iou", "precision", "recall", "specificity", "balanced_accuracy"):
            values = np.asarray([float(row[f"{detector}_{metric}"]) for row in rows])
            low, high = bootstrap_mean_ci(values, seed=20260923)
            record[f"{metric}_mean"] = float(values.mean())
            record[f"{metric}_ci_low"] = low
            record[f"{metric}_ci_high"] = high
        probabilities = np.concatenate([value.reshape(-1) for value in probability_store[detector]])
        record["ece"] = expected_calibration_error(probabilities, targets)
        record["brier"] = float(np.mean((probabilities - targets) ** 2))
        output.append(record)
    return output


def paired_statistics(rows, seed):
    validator = StatisticalValidator()
    output = []
    for metric in ("dice", "iou", "precision", "recall", "specificity", "balanced_accuracy"):
        baseline = np.asarray([float(row[f"historical_{metric}"]) for row in rows])
        candidate = np.asarray([float(row[f"candidate_{metric}"]) for row in rows])
        differences = candidate - baseline
        t_result = validator.paired_t_test(baseline.tolist(), candidate.tolist())
        w_result = validator.wilcoxon(baseline.tolist(), candidate.tolist())
        low, high = bootstrap_mean_ci(differences, seed=seed)
        output.append(
            {
                "metric": metric,
                "n": len(rows),
                "historical_mean": float(baseline.mean()),
                "candidate_mean": float(candidate.mean()),
                "mean_paired_difference": float(differences.mean()),
                "difference_ci_low": low,
                "difference_ci_high": high,
                "paired_t_p_value": t_result.p_value,
                "wilcoxon_p_value": w_result.p_value,
                "cohens_dz": t_result.effect_size,
            }
        )
    return output


def build_report(decision):
    historical, candidate, dice = decision["historical"], decision["candidate"], decision["paired_dice"]
    return "\n".join(
        [
            "# Official Validation: Retrained Burn-Scar Detector",
            "",
            "The retrained detector was frozen before this comparison. It was evaluated once on the 98-scene official geographic validation partition against the historical checkpoint. The historical model had previously used this partition for model selection, so the comparison is conservative in its favor. The final test partition remains untouched.",
            "",
            "| detector | mean scene Dice | IoU | balanced accuracy | ECE | Brier |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
            f"| historical | {historical['dice_mean']:.4f} | {historical['iou_mean']:.4f} | {historical['balanced_accuracy_mean']:.4f} | {historical['ece']:.4f} | {historical['brier']:.4f} |",
            f"| retrained candidate | {candidate['dice_mean']:.4f} | {candidate['iou_mean']:.4f} | {candidate['balanced_accuracy_mean']:.4f} | {candidate['ece']:.4f} | {candidate['brier']:.4f} |",
            "",
            f"Paired Dice difference: {dice['mean_paired_difference']:+.4f}, 95% CI {dice['difference_ci_low']:+.4f} to {dice['difference_ci_high']:+.4f}, paired t-test p={dice['paired_t_p_value']:.4g}, Wilcoxon p={dice['wilcoxon_p_value']:.4g}.",
            "",
            f"**{decision['decision'].replace('_', ' ').title()}.** This decision authorizes the checkpoint only for future compression validation. It does not retrospectively alter registered results or consume the final test set.",
        ]
    )


def read_csv(path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
