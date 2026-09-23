"""Summarize the one frozen official-validation comparison without opening test data."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.decide_ecofirebias_development_gate import paired_bootstrap_ci, sha256  # noqa: E402

OUT = ROOT / "results/future_satellite_cohort_audit/ecofirebias_official_validation"
METHODS = ("vqvae_fixed_utility", "jpeg_rdo", "jpeg2000_rdo")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate_rows(
    rows: list[dict[str, str]], plan: dict[str, object],
    manifest: dict[str, dict[str, str]], kinds: dict[str, str],
) -> dict[str, dict[str, dict[str, str]]]:
    expected_ids = set(plan["sample_ids"])
    if len(rows) != len(expected_ids) * len(METHODS):
        raise ValueError("Incomplete or extra benchmark rows")
    indexed: dict[str, dict[str, dict[str, str]]] = {}
    for method in METHODS:
        method_rows = [row for row in rows if row["method"] == method]
        if len(method_rows) != len(expected_ids) or {row["sample_id"] for row in method_rows} != expected_ids:
            raise ValueError(f"Missing or duplicated {method} rows")
        indexed[method] = {row["sample_id"]: row for row in method_rows}
    for row in rows:
        sample_id = row["sample_id"]
        expected = manifest[sample_id]
        if row["event_group"] != expected["event_group"] or row["source_split"] != "val":
            raise ValueError("Benchmark event or source split differs from frozen manifest")
        if row["split"] != "official_validation" or row["label_source"] != expected["label_source"]:
            raise ValueError("Benchmark label or split differs from frozen manifest")
        if int(row["target_bytes"]) != plan["maximum_wire_bytes_per_image"]:
            raise ValueError("Benchmark byte ceiling differs from frozen plan")
        if not 0 < int(row["encoded_bytes"]) <= plan["maximum_wire_bytes_per_image"]:
            raise ValueError("Serialized payload exceeds frozen byte ceiling")
        if not np.isfinite(float(row["sus"])) or not np.isfinite(float(row["label_dice"])):
            raise ValueError("Nonfinite utility or proxy Dice")
        if kinds[sample_id] not in ("burn", "neg"):
            raise ValueError("Unknown source kind")
    return indexed


def paired_difference(
    candidate: dict[str, dict[str, str]], reference: dict[str, dict[str, str]],
    sample_ids: list[str], metric: str,
) -> dict[str, float | int]:
    differences = np.asarray(
        [float(candidate[sample_id][metric]) - float(reference[sample_id][metric]) for sample_id in sample_ids],
        dtype=float,
    )
    low, high = paired_bootstrap_ci(differences, 10000, 20260922)
    return {"mean_difference": float(differences.mean()), "ci_low": low, "ci_high": high,
            "n_events": len(sample_ids)}


def summarize(method: dict[str, dict[str, str]], positive_ids: list[str], negative_ids: list[str]) -> dict[str, float]:
    def mean(ids: list[str], metric: str) -> float:
        return float(np.mean([float(method[sample_id][metric]) for sample_id in ids]))

    return {
        "burn_sus": mean(positive_ids, "sus"),
        "burn_proxy_dice": mean(positive_ids, "label_dice"),
        "burn_psnr_db": mean(positive_ids, "psnr"),
        "burn_ssim": mean(positive_ids, "ssim"),
        "burn_detector_retention": mean(positive_ids, "detector_retention"),
        "negative_predicted_positive_fraction": mean(negative_ids, "predicted_positive_fraction"),
        "mean_serialized_bytes": mean(positive_ids + negative_ids, "encoded_bytes"),
        "max_serialized_bytes": max(int(row["encoded_bytes"]) for row in method.values()),
    }


def report() -> dict[str, object]:
    plan_path = OUT / "VALIDATION_PLAN_FROZEN.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    prep = json.loads((OUT / "PREPARATION.json").read_text(encoding="utf-8"))
    manifest_path = OUT / "validation_manifest.csv"
    metadata_path = ROOT / "datasets/wildfire_global_v1/metadata.csv"
    if prep["plan_sha256"] != sha256(plan_path) or prep["manifest_sha256"] != sha256(manifest_path):
        raise ValueError("Preparation differs from the pre-scoring frozen selection")
    if plan["metadata_sha256"] != sha256(metadata_path):
        raise ValueError("Source metadata changed after selection")
    if plan["sealed_test_scoring_authorized"] or plan["images"] != 120 or plan["events"] != 60:
        raise ValueError("Unexpected validation plan")
    for name, checkpoint in (
        ("base", ROOT / "models/checkpoints/vqvae_mixed_wire_finetuned.pt"),
        ("adapted", ROOT / "models/checkpoints/vqvae_ecofirebias_train_adapted.pt"),
    ):
        if sha256(checkpoint) != plan["candidate_checkpoint_sha256"][name]:
            raise ValueError(f"{name} checkpoint changed after freezing")
    manifest_rows = read_csv(manifest_path)
    manifest = {row["sample_id"]: row for row in manifest_rows}
    if len(manifest) != 120 or set(manifest) != set(plan["sample_ids"]):
        raise ValueError("Manifest differs from frozen sample IDs")
    metadata = {row["example_id"]: row for row in read_csv(metadata_path)}
    kinds = {sample_id: metadata[sample_id]["kind"] for sample_id in manifest}
    positive_ids = [sample_id for sample_id in plan["sample_ids"] if kinds[sample_id] == "burn"]
    negative_ids = [sample_id for sample_id in plan["sample_ids"] if kinds[sample_id] == "neg"]
    if len(positive_ids) != 60 or len(negative_ids) != 60:
        raise ValueError("Validation must have 60 burn/negative pairs")
    for event_id in plan["event_ids"]:
        if sum(manifest[sample_id]["event_group"] == event_id for sample_id in positive_ids) != 1:
            raise ValueError("Burn event pairing differs from frozen plan")
        if sum(manifest[sample_id]["event_group"] == event_id for sample_id in negative_ids) != 1:
            raise ValueError("Negative event pairing differs from frozen plan")
    paths = {name: OUT / f"{name}_1200/extreme_rate_rows.csv" for name in ("base", "adapted")}
    runs = {
        name: validate_rows(read_csv(path), plan, manifest, kinds)
        for name, path in paths.items()
    }
    for method in ("jpeg_rdo", "jpeg2000_rdo"):
        if runs["base"][method] != runs["adapted"][method]:
            raise ValueError(f"{method} changed between checkpoint runs")
    named = {
        "JPEG": runs["base"]["jpeg_rdo"],
        "JPEG2000 RDO": runs["base"]["jpeg2000_rdo"],
        "Base VQ-VAE": runs["base"]["vqvae_fixed_utility"],
        "Adapted VQ-VAE": runs["adapted"]["vqvae_fixed_utility"],
    }
    summaries = {name: summarize(rows, positive_ids, negative_ids) for name, rows in named.items()}
    differences = {}
    for reference_name in ("Base VQ-VAE", "JPEG2000 RDO"):
        differences[reference_name] = {
            metric: paired_difference(named["Adapted VQ-VAE"], named[reference_name], ids, metric)
            for metric, ids in (
                ("sus", positive_ids), ("label_dice", positive_ids),
                ("psnr", positive_ids), ("ssim", positive_ids),
                ("detector_retention", positive_ids),
                ("predicted_positive_fraction", negative_ids),
            )
        }
    result: dict[str, object] = {
        "status": "one_off_official_validation_not_sealed_test",
        "frozen_plan_sha256": sha256(plan_path),
        "preparation_sha256": sha256(OUT / "PREPARATION.json"),
        "rows_sha256": {name: sha256(path) for name, path in paths.items()},
        "events": 60, "burn_images": 60, "negative_images": 60,
        "byte_ceiling": 1200, "sealed_test_scored": False,
        "summaries": summaries, "paired_adapted_minus": differences,
        "interpretation": "One event-disjoint official-validation comparison, but countries overlap selected training, labels are quantized dNBR proxies, and the prior frozen development gate failed; this cannot establish independent superiority or authorize sealed testing.",
    }
    (OUT / "VALIDATION_REPORT.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# EcoFireBias official-validation comparison", "",
        "**Status:** one frozen, previously unscored official-validation comparison; sealed test remains unscored. The earlier 18-event development gate failed and is not superseded.", "",
        "60 event pairs (60 burn, 60 matched negative images), native 224 x 224, maximum 1,200 serialized bytes per image. Selection, checkpoints, methods, and quantized dNBR >85 proxy rule were frozen before model scoring. JPEG and JPEG2000 are distortion-optimized under the same ceiling; controls are byte-identical between checkpoint runs.", "",
        "| Method | Burn SUS | Burn proxy Dice | Burn PSNR (dB) | Burn SSIM | Detector retention | Negative predicted-positive area | Mean serialized bytes |",
        "|:--|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for name, summary in summaries.items():
        lines.append(f"| {name} | {summary['burn_sus']:.2f} | {summary['burn_proxy_dice']:.3f} | {summary['burn_psnr_db']:.2f} | {summary['burn_ssim']:.3f} | {summary['burn_detector_retention']:.3f} | {summary['negative_predicted_positive_fraction']:.3f} | {summary['mean_serialized_bytes']:.1f} |")
    lines.extend(["", "## Event-paired adapted VQ-VAE differences", "", "95% percentile bootstrap confidence intervals, 10,000 paired-event resamples; burn metrics use one burn image per event, negative area uses one negative image per event.", ""])
    for reference_name, metrics in differences.items():
        lines.extend([f"### Versus {reference_name}", ""])
        for metric, comparison in metrics.items():
            lines.append(f"- {metric}: {comparison['mean_difference']:+.3f} [95% CI {comparison['ci_low']:+.3f}, {comparison['ci_high']:+.3f}].")
        lines.append("")
    lines.extend([
        "## Interpretation and limits", "",
        "This is a validation-split check, not a replacement for the predeclared failed development gate. All 60 validation events are distinct from selected train, internal-validation, label-calibration, reused-development and sealed-test events, but every selected validation event is in a country already represented in selected training. Event identifiers are patch-derived and do not guarantee scene or spatial-footprint independence. Labels are quantized dNBR spectral proxies, not manually verified burn-scar ground truth; original dNBR invalid pixels are not fully recoverable from the quantized export. Five images required grid reprojection; uncovered edge pixels were excluded. SUS and detector retention share a detector and are not independent evidence. Classical encoders choose highest original-image PSNR among predefined settings, using the original at the sender; selection overhead and encoder compute are not included in wire bytes. No LPIPS was run in this frozen comparison. No publication-grade superiority claim or sealed-test opening is justified.", "",
    ])
    (OUT / "VALIDATION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return result


if __name__ == "__main__":
    result = report()
    print(json.dumps({"events": result["events"], "sealed_test_scored": result["sealed_test_scored"], "summaries": result["summaries"]}, indent=2))
