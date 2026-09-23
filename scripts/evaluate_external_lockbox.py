"""Run the one-shot detector evaluation on the frozen FireScope lockbox."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.research_wildfire import load_grayscale_image, load_rgb_image  # noqa: E402
from evaluation.matched_rate import bootstrap_mean_ci  # noqa: E402
from scripts.evaluate_burn_scar_detector import confusion_counts, metrics_from_counts  # noqa: E402
from semantic_ai.wildfire_detector import WildfireDetector  # noqa: E402


class CalibrationAccumulator:
    """Compute pixel calibration statistics without retaining full-size maps."""

    def __init__(self, bins: int = 15) -> None:
        self.edges = np.linspace(0.0, 1.0, bins + 1)
        self.count = np.zeros(bins, dtype="float64")
        self.confidence_sum = np.zeros(bins, dtype="float64")
        self.target_sum = np.zeros(bins, dtype="float64")
        self.brier_sum = 0.0
        self.total = 0

    def update(self, probability: np.ndarray, target: np.ndarray) -> None:
        probability = np.asarray(probability, dtype="float32").reshape(-1)
        target = np.asarray(target, dtype="float32").reshape(-1)
        indices = np.clip(np.digitize(probability, self.edges[1:-1]), 0, len(self.count) - 1)
        for index in range(len(self.count)):
            selected = indices == index
            if selected.any():
                self.count[index] += selected.sum()
                self.confidence_sum[index] += probability[selected].sum()
                self.target_sum[index] += target[selected].sum()
        self.brier_sum += float(np.square(probability - target).sum())
        self.total += probability.size

    def metrics(self) -> tuple[float, float]:
        if self.total == 0:
            return float("nan"), float("nan")
        populated = self.count > 0
        mean_confidence = np.divide(
            self.confidence_sum,
            self.count,
            out=np.zeros_like(self.confidence_sum),
            where=populated,
        )
        mean_target = np.divide(
            self.target_sum,
            self.count,
            out=np.zeros_like(self.target_sum),
            where=populated,
        )
        ece = float((self.count * np.abs(mean_confidence - mean_target)).sum() / self.total)
        return ece, self.brier_sum / self.total


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the frozen detector on the FireScope lockbox once.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("results/external_lockbox_firescope_100/firescope_100_lockbox_manifest.csv"),
    )
    parser.add_argument(
        "--freeze-record",
        type=Path,
        default=Path("results/external_lockbox_firescope_100/LOCKBOX_FROZEN.json"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/checkpoints/wildfire_utility_segmentation_retrained.pt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/external_lockbox_firescope_100/detector_evaluation"),
    )
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260924)
    args = parser.parse_args()

    freeze = json.loads(args.freeze_record.read_text(encoding="utf-8"))
    if freeze.get("status") != "FROZEN_BEFORE_EVALUATION":
        raise ValueError("The external lockbox is not frozen")
    if freeze.get("manifest_sha256") != _sha256(args.manifest):
        raise ValueError("The lockbox manifest changed after freezing")

    rows = _read_csv(args.manifest)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    threshold = float(checkpoint["config"]["threshold"])
    detector = WildfireDetector(args.checkpoint, output_dir=None, save_visualizations=False)
    calibration = CalibrationAccumulator()
    results: list[dict[str, object]] = []
    aggregate = np.zeros(4, dtype="float64")
    for index, row in enumerate(rows, start=1):
        print(f"external lockbox detector {index}/{len(rows)}: {row['sample_id']}", flush=True)
        image = load_rgb_image(Path(row["image_path"])).convert("RGB").resize(
            (args.image_size, args.image_size), Image.Resampling.LANCZOS
        )
        target_image = load_grayscale_image(Path(row["mask_path"])).resize(
            (args.image_size, args.image_size), Image.Resampling.NEAREST
        )
        target = np.asarray(target_image, dtype="uint8") > 0
        probability = detector.detect(image).confidence_map
        predicted = probability >= threshold
        counts = confusion_counts(predicted, target)
        aggregate += counts
        metrics = metrics_from_counts(*counts)
        calibration.update(probability, target)
        results.append(
            {
                "sample_id": row["sample_id"],
                "region": row["geographic_group"].split(":", 1)[0],
                "geographic_group": row["geographic_group"],
                "fire_status": row["fire_status"],
                "positive_pixel_fraction": float(target.mean()),
                "predicted_positive_fraction": float(predicted.mean()),
                **metrics,
            }
        )

    ece, brier = calibration.metrics()
    aggregate_metrics = metrics_from_counts(*aggregate)
    positive = [row for row in results if row["fire_status"] == "positive"]
    negative = [row for row in results if row["fire_status"] == "negative"]
    summary = {
        "evaluation_status": "completed_once_after_freeze",
        "n": len(results),
        "positive_n": len(positive),
        "negative_n": len(negative),
        "image_size": args.image_size,
        "frozen_threshold": threshold,
        "checkpoint_sha256": _sha256(args.checkpoint),
        "manifest_sha256": _sha256(args.manifest),
        "positive_scene_metrics": {
            metric: _scene_summary(positive, metric, args.seed)
            for metric in ("dice", "iou", "precision", "recall", "specificity", "balanced_accuracy")
        },
        "negative_false_positive_fraction": _scene_summary(
            negative, "predicted_positive_fraction", args.seed
        ),
        "aggregate_pixel_metrics": aggregate_metrics,
        "ece": ece,
        "brier": brier,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "external_detector_per_scene.csv", results)
    (args.output_dir / "external_detector_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (args.output_dir / "external_detector_report.md").write_text(
        _report(summary), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


def _scene_summary(rows: list[dict[str, object]], metric: str, seed: int) -> dict[str, float]:
    values = np.asarray([float(row[metric]) for row in rows], dtype="float64")
    low, high = bootstrap_mean_ci(values, seed=seed)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "ci_low": low,
        "ci_high": high,
    }


def _report(summary: dict[str, object]) -> str:
    positive = summary["positive_scene_metrics"]
    dice = positive["dice"]
    balanced = positive["balanced_accuracy"]
    negative = summary["negative_false_positive_fraction"]
    aggregate = summary["aggregate_pixel_metrics"]
    return "\n".join(
        [
            "# Frozen FireScope External Detector Evaluation",
            "",
            "This is the first and only evaluation of the frozen retrained burn-scar detector on the 100-scene FireScope lockbox. No threshold, model, or selector was fitted using these scenes.",
            "",
            f"- Positive scenes: {summary['positive_n']}; negative controls: {summary['negative_n']}.",
            f"- Mean positive-scene Dice: {dice['mean']:.4f} (95% CI {dice['ci_low']:.4f} to {dice['ci_high']:.4f}).",
            f"- Mean positive-scene balanced accuracy: {balanced['mean']:.4f} (95% CI {balanced['ci_low']:.4f} to {balanced['ci_high']:.4f}).",
            f"- Mean negative-control false-positive area: {negative['mean']:.4f} (95% CI {negative['ci_low']:.4f} to {negative['ci_high']:.4f}).",
            f"- Aggregate pixel Dice: {aggregate['dice']:.4f}; aggregate IoU: {aggregate['iou']:.4f}.",
            f"- Pixel ECE: {summary['ece']:.4f}; Brier score: {summary['brier']:.4f}.",
            "",
            "FireScope images are pre-event Sentinel-2 context associated with next-year event masks. These results therefore measure cross-dataset burn-scar relevance transfer, not active flame or smoke detection.",
        ]
    )


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
