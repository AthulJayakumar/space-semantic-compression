"""Compare the supervised burn-scar detector with the handcrafted baseline."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.research_wildfire import load_grayscale_image, load_rgb_image  # noqa: E402
from evaluation.matched_rate import bootstrap_mean_ci  # noqa: E402
from evaluation.statistics import StatisticalValidator  # noqa: E402
from semantic_ai.wildfire_detector import WildfireDetector  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate supervised and handcrafted burn-scar utility maps.")
    parser.add_argument("--manifest", type=Path, default=Path("results/validated_hls_500/validated_scene_manifest.csv"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/checkpoints/wildfire_utility_segmentation.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/burn_scar_utility_detector"))
    args = parser.parse_args()

    rows = read_csv(args.manifest)
    validation = [row for row in rows if row["split"] == "validation"]
    test = [row for row in rows if row["split"] == "test"]
    supervised = WildfireDetector(args.checkpoint, output_dir=None, save_visualizations=False)
    handcrafted = WildfireDetector(None, output_dir=None, save_visualizations=False)
    checkpoint = __import__("torch").load(args.checkpoint, map_location="cpu", weights_only=True)
    supervised_threshold = float(checkpoint["config"]["threshold"])

    validation_cache = collect_outputs(validation, supervised, handcrafted)
    handcrafted_threshold = select_threshold(validation_cache, "handcrafted_probability")
    test_cache = collect_outputs(test, supervised, handcrafted)
    result_rows = evaluate_rows(test_cache, supervised_threshold, handcrafted_threshold)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "paired_test_metrics.csv", result_rows)
    summary = summarize(result_rows)
    statistics = paired_statistics(result_rows)
    write_csv(args.output_dir / "detector_comparison_summary.csv", summary)
    write_csv(args.output_dir / "detector_comparison_statistics.csv", statistics)
    report = build_report(summary, statistics, supervised_threshold, handcrafted_threshold)
    (args.output_dir / "detector_comparison_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"summary": summary, "statistics": statistics}, indent=2))


def collect_outputs(rows, supervised, handcrafted) -> list[dict[str, object]]:
    outputs: list[dict[str, object]] = []
    for index, row in enumerate(rows, start=1):
        print(f"detector evaluation {index}/{len(rows)}: {row['sample_id']}", flush=True)
        image = load_rgb_image(Path(row["image_path"]))
        mask = np.asarray(load_grayscale_image(Path(row["mask_path"])), dtype="uint8") > 0
        supervised_output = supervised.detect(image)
        handcrafted_output = handcrafted.detect(image)
        outputs.append(
            {
                "sample_id": row["sample_id"],
                "fire_status": row["fire_status"],
                "target": mask,
                "supervised_probability": supervised_output.utility_map,
                "handcrafted_probability": handcrafted_output.utility_map,
            }
        )
    return outputs


def select_threshold(rows: list[dict[str, object]], probability_key: str) -> float:
    candidates = np.linspace(0.1, 0.9, 17)
    scores = []
    for threshold in candidates:
        counts = aggregate_counts(rows, probability_key, float(threshold))
        scores.append((float(threshold), metrics_from_counts(*counts)["dice"]))
    return max(scores, key=lambda item: item[1])[0]


def evaluate_rows(rows, supervised_threshold: float, handcrafted_threshold: float) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for row in rows:
        record = {"sample_id": row["sample_id"], "fire_status": row["fire_status"]}
        for method, key, threshold in (
            ("supervised", "supervised_probability", supervised_threshold),
            ("handcrafted", "handcrafted_probability", handcrafted_threshold),
        ):
            counts = confusion_counts(np.asarray(row[key]) >= threshold, np.asarray(row["target"]))
            metrics = metrics_from_counts(*counts)
            for metric, value in metrics.items():
                record[f"{method}_{metric}"] = value
        output.append(record)
    return output


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for method in ("handcrafted", "supervised"):
        record: dict[str, object] = {"method": method, "n": len(rows)}
        for metric in ("dice", "iou", "precision", "recall", "specificity", "balanced_accuracy"):
            values = np.asarray([float(row[f"{method}_{metric}"]) for row in rows])
            low, high = bootstrap_mean_ci(values, seed=20260921)
            record[f"{metric}_mean"] = float(values.mean())
            record[f"{metric}_ci_low"] = low
            record[f"{metric}_ci_high"] = high
        output.append(record)
    return output


def paired_statistics(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    validator = StatisticalValidator()
    output: list[dict[str, object]] = []
    for metric in ("dice", "iou", "precision", "recall", "specificity", "balanced_accuracy"):
        baseline = [float(row[f"handcrafted_{metric}"]) for row in rows]
        candidate = [float(row[f"supervised_{metric}"]) for row in rows]
        differences = np.asarray(candidate) - np.asarray(baseline)
        t_test = validator.paired_t_test(baseline, candidate)
        wilcoxon = validator.wilcoxon(baseline, candidate)
        low, high = bootstrap_mean_ci(differences, seed=20260921)
        output.append(
            {
                "metric": metric,
                "n": len(rows),
                "handcrafted_mean": float(np.mean(baseline)),
                "supervised_mean": float(np.mean(candidate)),
                "mean_difference": float(differences.mean()),
                "paired_t_p_value": t_test.p_value,
                "wilcoxon_p_value": wilcoxon.p_value,
                "cohens_dz": t_test.effect_size,
                "difference_ci_low": low,
                "difference_ci_high": high,
            }
        )
    return output


def aggregate_counts(rows, probability_key: str, threshold: float) -> tuple[float, float, float, float]:
    counts = np.zeros(4, dtype="float64")
    for row in rows:
        counts += confusion_counts(np.asarray(row[probability_key]) >= threshold, np.asarray(row["target"]))
    return tuple(float(value) for value in counts)


def confusion_counts(predicted: np.ndarray, target: np.ndarray) -> tuple[float, float, float, float]:
    predicted = predicted.astype(bool)
    target = target.astype(bool)
    return (
        float(np.logical_and(predicted, target).sum()),
        float(np.logical_and(predicted, ~target).sum()),
        float(np.logical_and(~predicted, target).sum()),
        float(np.logical_and(~predicted, ~target).sum()),
    )


def metrics_from_counts(tp: float, fp: float, fn: float, tn: float) -> dict[str, float]:
    epsilon = 1e-9
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    specificity = tn / (tn + fp + epsilon)
    return {
        "dice": 2.0 * tp / (2.0 * tp + fp + fn + epsilon),
        "iou": tp / (tp + fp + fn + epsilon),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": 0.5 * (recall + specificity),
    }


def build_report(summary, statistics, supervised_threshold: float, handcrafted_threshold: float) -> str:
    dice = next(row for row in statistics if row["metric"] == "dice")
    lines = [
        "# Burn-Scar Detector Comparison",
        "",
        "Both thresholds were selected without using the test split. The supervised threshold came from model validation; the handcrafted threshold was optimized on the same geographic validation partition.",
        "",
        f"- Supervised threshold: {supervised_threshold:.3f}",
        f"- Handcrafted threshold: {handcrafted_threshold:.3f}",
        f"- Test images: {dice['n']}",
        f"- Handcrafted mean per-image Dice: {dice['handcrafted_mean']:.4f}",
        f"- Supervised mean per-image Dice: {dice['supervised_mean']:.4f}",
        f"- Paired Dice improvement: {dice['mean_difference']:.4f}",
        f"- Paired t-test p-value: {dice['paired_t_p_value']:.6g}",
        f"- Wilcoxon p-value: {dice['wilcoxon_p_value']:.6g}",
        f"- Cohen's dz: {dice['cohens_dz']:.4f}",
        f"- Bootstrap 95% CI: {dice['difference_ci_low']:.4f} to {dice['difference_ci_high']:.4f}",
        "",
        "The supervised model is promoted only if its mask agreement improves materially. It remains a burn-scar relevance model, not an active-fire or smoke detector.",
    ]
    return "\n".join(lines)


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
