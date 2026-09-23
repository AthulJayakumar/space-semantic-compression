"""Apply the already-frozen joint gate to EcoFireBias training development rows."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/future_satellite_cohort_audit"
RUNS = {
    "mixed_wire": OUT / "development_mixed_wire_1200/extreme_rate_rows.csv",
    "mask_aware": OUT / "development_mask_aware_1200/extreme_rate_rows.csv",
}
METHODS = ("vqvae_fixed_utility", "jpeg_rdo", "jpeg2000_rdo")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def paired_bootstrap_ci(differences: np.ndarray, resamples: int, seed: int) -> tuple[float, float]:
    if differences.ndim != 1 or not len(differences) or not np.all(np.isfinite(differences)):
        raise ValueError("Finite one-dimensional paired differences are required")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(resamples, len(differences)))
    means = differences[indices].mean(axis=1)
    low, high = np.quantile(means, (0.025, 0.975))
    return float(low), float(high)


def paired_result(differences: list[float], gate: dict[str, object]) -> dict[str, float]:
    values = np.asarray(differences, dtype=float)
    low, high = paired_bootstrap_ci(values, int(gate["bootstrap_resamples"]), int(gate["bootstrap_seed"]))
    return {
        "mean_difference": float(values.mean()),
        "ci_low": low,
        "ci_high": high,
        "paired_t_p": float(stats.ttest_1samp(values, 0).pvalue) if np.std(values) else 1.0,
        "wilcoxon_p": float(stats.wilcoxon(values, zero_method="zsplit").pvalue),
        "n_events": len(values),
    }


def evaluate_run(rows: list[dict[str, str]], gate: dict[str, object], metadata: dict[str, dict[str, str]]) -> dict[str, object]:
    expected_ids = set(gate["development_ids"])
    by_method = {}
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        if len(selected) != len(expected_ids) or {row["sample_id"] for row in selected} != expected_ids:
            raise ValueError(f"Incomplete or duplicated {method} development rows")
        by_method[method] = {row["sample_id"]: row for row in selected}
    if len(rows) != len(METHODS) * len(expected_ids):
        raise ValueError("Unexpected extra benchmark rows")
    for row in rows:
        if row["source_split"] != "train" or row["split"] != "development":
            raise ValueError("A non-development source was evaluated")
        if row["event_group"] != metadata[row["sample_id"]]["event_id"]:
            raise ValueError("Event grouping differs from frozen metadata")
        if int(row["target_bytes"]) != gate["maximum_wire_bytes_per_image"]:
            raise ValueError("Wire byte ceiling differs from frozen gate")
        if int(row["encoded_bytes"]) > gate["maximum_wire_bytes_per_image"]:
            raise ValueError("Serialized payload exceeds frozen byte ceiling")

    positive_ids = [sample_id for sample_id in gate["development_ids"] if metadata[sample_id]["kind"] == "burn"]
    negative_ids = [sample_id for sample_id in gate["development_ids"] if metadata[sample_id]["kind"] == "neg"]
    if len(positive_ids) != 18 or len(negative_ids) != 18:
        raise ValueError("Expected 18 positive and 18 negative development images")
    reference = by_method["jpeg2000_rdo"]
    candidate = by_method["vqvae_fixed_utility"]
    metrics = {
        metric: paired_result(
            [float(candidate[sample_id][metric]) - float(reference[sample_id][metric]) for sample_id in ids],
            gate["gate"],
        )
        for metric, ids in (("sus", positive_ids), ("label_dice", positive_ids), ("predicted_positive_fraction", negative_ids))
    }
    checks = {
        "sus_mean_advantage": metrics["sus"]["mean_difference"] >= gate["gate"]["burn_positive_sus_mean_difference_at_least"],
        "proxy_dice_mean_advantage": metrics["label_dice"]["mean_difference"] >= gate["gate"]["burn_positive_proxy_dice_mean_difference_at_least"],
        "sus_ci_positive": metrics["sus"]["ci_low"] > gate["gate"]["paired_event_bootstrap_95_percent_lower_bound_for_both_differences_above"],
        "proxy_dice_ci_positive": metrics["label_dice"]["ci_low"] > gate["gate"]["paired_event_bootstrap_95_percent_lower_bound_for_both_differences_above"],
        "negative_false_positive_increase": metrics["predicted_positive_fraction"]["mean_difference"] <= gate["gate"]["negative_pair_false_positive_area_mean_increase_at_most"],
    }
    summaries = {}
    for method, samples in by_method.items():
        summaries[method] = {
            "mean_positive_sus": float(np.mean([float(samples[sample_id]["sus"]) for sample_id in positive_ids])),
            "mean_positive_proxy_dice": float(np.mean([float(samples[sample_id]["label_dice"]) for sample_id in positive_ids])),
            "mean_positive_psnr": float(np.mean([float(samples[sample_id]["psnr"]) for sample_id in positive_ids])),
            "mean_positive_ssim": float(np.mean([float(samples[sample_id]["ssim"]) for sample_id in positive_ids])),
            "mean_positive_detector_retention": float(np.mean([float(samples[sample_id]["detector_retention"]) for sample_id in positive_ids])),
            "mean_negative_predicted_positive_fraction": float(np.mean([float(samples[sample_id]["predicted_positive_fraction"]) for sample_id in negative_ids])),
            "mean_encoded_bytes": float(np.mean([float(row["encoded_bytes"]) for row in samples.values()])),
            "max_encoded_bytes": max(int(row["encoded_bytes"]) for row in samples.values()),
        }
    return {"advance": all(checks.values()), "checks": checks, "paired_differences_vs_jpeg2000": metrics, "summaries": summaries}


def decide() -> dict[str, object]:
    gate_path = OUT / "DEVELOPMENT_GATE_FROZEN.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    preparation_path = OUT / "ecofirebias_development_preparation.json"
    preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
    if preparation["gate_sha256"] != sha256(gate_path) or preparation["images"] != 36:
        raise ValueError("Prepared set is inconsistent with the frozen gate")
    metadata = {row["example_id"]: row for row in read_csv(ROOT / "datasets/wildfire_global_v1/metadata.csv")}
    runs = {}
    controls = None
    for name, path in RUNS.items():
        rows = read_csv(path)
        current = {row["sample_id"]: (row["encoded_bytes"], row["sus"], row["label_dice"]) for row in rows if row["method"] == "jpeg2000_rdo"}
        if controls is not None and current != controls:
            raise ValueError("JPEG2000 reference differs across checkpoint runs")
        controls = current
        runs[name] = {"results_sha256": sha256(path), **evaluate_run(rows, gate, metadata)}
    decision = {
        "status": "development_screen_only",
        "gate_sha256": sha256(gate_path),
        "preparation_sha256": sha256(preparation_path),
        "candidate_checkpoints": list(RUNS),
        "eligible_for_sealed_test": [name for name, result in runs.items() if result["advance"]],
        "sealed_test_scored": False,
        "interpretation": "Quantized dNBR proxy, not manually verified burn-scar truth; development evidence is not confirmatory",
        "runs": runs,
    }
    (OUT / "development_gate_decision.json").write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# EcoFireBias training-development decision", "",
        "**Status: no sealed test data scored.** This is an 18-event training-split development screen at 1,200 bytes per native 224x224 image.", "",
        "| Candidate | Burn SUS | JPEG2000 burn SUS | Burn proxy Dice | JPEG2000 proxy Dice | Mean serialized bytes | Gate |",
        "|:--|--:|--:|--:|--:|--:|:--|",
    ]
    for name, run in runs.items():
        summary = run["summaries"]
        vq = summary["vqvae_fixed_utility"]
        jp2 = summary["jpeg2000_rdo"]
        lines.append(
            f"| {name} | {vq['mean_positive_sus']:.2f} | {jp2['mean_positive_sus']:.2f} | "
            f"{vq['mean_positive_proxy_dice']:.3f} | {jp2['mean_positive_proxy_dice']:.3f} | "
            f"{vq['mean_encoded_bytes']:.1f} | {'PASS' if run['advance'] else 'NO-GO'} |"
        )
    lines.extend(["", "## Paired event differences versus JPEG2000", ""])
    for name, run in runs.items():
        for metric, value in run["paired_differences_vs_jpeg2000"].items():
            lines.append(f"- {name} {metric}: mean {value['mean_difference']:+.3f}; 95% paired-event bootstrap CI [{value['ci_low']:+.3f}, {value['ci_high']:+.3f}].")
    lines.extend(["", "## Descriptive quality on burn chips", "", "| Method | PSNR (dB) | SSIM | Detector retention | Mean serialized bytes |", "|:--|--:|--:|--:|--:|"])
    first = runs["mixed_wire"]["summaries"]
    for method in ("jpeg_rdo", "jpeg2000_rdo"):
        summary = first[method]
        lines.append(f"| {method} | {summary['mean_positive_psnr']:.2f} | {summary['mean_positive_ssim']:.3f} | {summary['mean_positive_detector_retention']:.3f} | {summary['mean_encoded_bytes']:.1f} |")
    for name, run in runs.items():
        summary = run["summaries"]["vqvae_fixed_utility"]
        lines.append(f"| {name} VQ-VAE | {summary['mean_positive_psnr']:.2f} | {summary['mean_positive_ssim']:.3f} | {summary['mean_positive_detector_retention']:.3f} | {summary['mean_encoded_bytes']:.1f} |")
    lines.extend(["", "All five predeclared gate checks must pass. The dNBR mask is a quantized spectral proxy; some pixels with invalid original dNBR cannot be distinguished from low byte values. Two development pairs required UTM-grid reprojection, and uncovered edge pixels were excluded. This screen cannot support independent or publication-grade superiority claims.", ""])
    (OUT / "DEVELOPMENT_DECISION.md").write_text("\n".join(lines), encoding="utf-8")
    return decision


def main() -> None:
    decision = decide()
    print(json.dumps({"eligible_for_sealed_test": decision["eligible_for_sealed_test"], "sealed_test_scored": False}, indent=2))


if __name__ == "__main__":
    main()
