"""Run dataset evidence for the detail-aware token scoring improvement.

This is Step 3 of the incremental model-improvement process. It compares the
current full selector against the same selector with the detail term removed.
The goal is to quantify whether boundary/detail preservation improves
mission-oriented metrics on real Sentinel-2 patches.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.schemas.response_schema import TransmissionConfig
from backend.services.factory import get_compression_service
from evaluation.statistics import StatisticalValidator
from token_selection.utility_pruner import TokenSelectionWeights, UtilityAwareTokenPruner


METRICS = [
    "semantic_utility_score",
    "detector_retention",
    "compression_ratio",
    "bandwidth_saved_percent",
    "psnr",
    "ssim",
    "lpips",
]


VARIANTS = {
    "full_detail_aware": TokenSelectionWeights(
        alpha_utility=0.55,
        beta_entropy=0.20,
        gamma_cost=0.05,
        delta_detail=0.20,
    ),
    "without_detail_term": TokenSelectionWeights(
        alpha_utility=0.65,
        beta_entropy=0.25,
        gamma_cost=0.10,
        delta_detail=0.0,
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare detail-aware token scoring against no-detail ablation.")
    parser.add_argument("--dataset-dir", default="datasets/sentinel2_500_patch/images")
    parser.add_argument("--output-dir", default="results/model_improvement_step3_detail_ablation")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--keep-ratios", nargs="+", type=float, default=[0.45, 0.80])
    parser.add_argument("--mission", default="wildfire_detection")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = discover_images(dataset_dir, args.limit)
    if not image_paths:
        raise ValueError(f"No images found in {dataset_dir}")

    service = get_compression_service()
    original_pruner = service.utility_pruner
    rows: list[dict[str, object]] = []
    try:
        for image_index, image_path in enumerate(image_paths, start=1):
            payload = image_path.read_bytes()
            for keep_ratio in args.keep_ratios:
                for variant, weights in VARIANTS.items():
                    service.utility_pruner = UtilityAwareTokenPruner(weights)
                    result = service.compress_image(
                        payload,
                        filename=image_path.name,
                        transmission_config=TransmissionConfig(semantic_keep_ratio=keep_ratio),
                        mission=args.mission,
                    )
                    rows.append(row_from_result(image_path, image_index, variant, keep_ratio, result))
                    print(f"[{image_index}/{len(image_paths)}] {variant} keep={keep_ratio:.2f} {image_path.name}")
    finally:
        service.utility_pruner = original_pruner

    summary_rows = summarize(rows)
    comparison_rows = paired_comparisons(rows)
    write_csv(output_dir / "detail_term_ablation_rows.csv", rows)
    write_csv(output_dir / "detail_term_ablation_summary.csv", summary_rows)
    write_csv(output_dir / "detail_term_ablation_paired_statistics.csv", comparison_rows)
    write_json(output_dir / "detail_term_ablation_manifest.json", args, rows, summary_rows, comparison_rows)
    figure_path = write_figures(output_dir, summary_rows)
    report_path = write_report(output_dir, image_paths, args.keep_ratios, summary_rows, comparison_rows, figure_path)
    print(report_path)


def discover_images(dataset_dir: Path, limit: int | None) -> list[Path]:
    images = sorted(
        [
            *dataset_dir.rglob("*.png"),
            *dataset_dir.rglob("*.jpg"),
            *dataset_dir.rglob("*.jpeg"),
            *dataset_dir.rglob("*.webp"),
        ]
    )
    return images[:limit] if limit else images


def row_from_result(image_path: Path, image_index: int, variant: str, keep_ratio: float, result) -> dict[str, object]:
    return {
        "image_index": image_index,
        "image": str(image_path),
        "variant": variant,
        "keep_ratio": keep_ratio,
        "semantic_utility_score": result.semantic_utility_score,
        "detector_retention": result.detector_retention,
        "compression_ratio": result.compression_ratio,
        "bandwidth_saved_percent": result.bandwidth_saved_percent,
        "psnr": result.psnr,
        "ssim": result.ssim,
        "lpips": result.lpips,
        "object_retention": result.object_retention,
        "relevance_retention": result.relevance_retention,
        "region_preservation": result.region_preservation,
        "semantic_token_count": result.semantic_token_count,
        "total_token_count": result.total_token_count,
        "inference_latency_ms": result.inference_latency_ms,
    }


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary_rows: list[dict[str, object]] = []
    keys = sorted({(row["variant"], row["keep_ratio"]) for row in rows}, key=lambda item: (float(item[1]), str(item[0])))
    for variant, keep_ratio in keys:
        group = [row for row in rows if row["variant"] == variant and row["keep_ratio"] == keep_ratio]
        summary: dict[str, object] = {
            "variant": variant,
            "keep_ratio": keep_ratio,
            "n_images": len({row["image"] for row in group}),
        }
        for metric in METRICS:
            values = np.asarray([row[metric] for row in group if row.get(metric) is not None], dtype="float64")
            if values.size == 0:
                continue
            ci = confidence_interval(values)
            summary[f"{metric}_mean"] = round(float(values.mean()), 6)
            summary[f"{metric}_std"] = round(float(values.std(ddof=1)) if values.size > 1 else 0.0, 6)
            summary[f"{metric}_ci_low"] = round(ci[0], 6)
            summary[f"{metric}_ci_high"] = round(ci[1], 6)
        summary_rows.append(summary)
    return summary_rows


def paired_comparisons(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    stats = StatisticalValidator()
    comparison_rows: list[dict[str, object]] = []
    keep_ratios = sorted({float(row["keep_ratio"]) for row in rows})
    for keep_ratio in keep_ratios:
        for metric in METRICS:
            baseline = metric_by_image(rows, "without_detail_term", keep_ratio, metric)
            candidate = metric_by_image(rows, "full_detail_aware", keep_ratio, metric)
            shared = sorted(set(baseline) & set(candidate))
            if not shared:
                continue
            baseline_values = [baseline[key] for key in shared]
            candidate_values = [candidate[key] for key in shared]
            differences = np.asarray(candidate_values, dtype="float64") - np.asarray(baseline_values, dtype="float64")
            ttest = stats.paired_t_test(baseline_values, candidate_values)
            wilcoxon = stats.wilcoxon(baseline_values, candidate_values)
            ci_low, ci_high = bootstrap_ci(differences)
            comparison_rows.append(
                {
                    "comparison": "full_detail_aware_vs_without_detail_term",
                    "keep_ratio": keep_ratio,
                    "metric": metric,
                    "n": len(shared),
                    "mean_difference": round(float(differences.mean()), 6),
                    "bootstrap_ci_low": round(ci_low, 6),
                    "bootstrap_ci_high": round(ci_high, 6),
                    "paired_t_p": round(float(ttest.p_value), 8) if np.isfinite(ttest.p_value) else "",
                    "wilcoxon_p": round(float(wilcoxon.p_value), 8) if np.isfinite(wilcoxon.p_value) else "",
                    "cohens_d": round(float(ttest.effect_size), 6),
                    "interpretation": interpret_difference(metric, float(differences.mean())),
                }
            )
    return comparison_rows


def metric_by_image(rows: list[dict[str, object]], variant: str, keep_ratio: float, metric: str) -> dict[str, float]:
    return {
        str(row["image"]): float(row[metric])
        for row in rows
        if row["variant"] == variant and float(row["keep_ratio"]) == keep_ratio and row.get(metric) is not None
    }


def confidence_interval(values: np.ndarray) -> tuple[float, float]:
    if values.size < 2:
        value = float(values.mean()) if values.size else float("nan")
        return value, value
    half_width = 1.96 * float(values.std(ddof=1)) / float(np.sqrt(values.size))
    mean = float(values.mean())
    return mean - half_width, mean + half_width


def bootstrap_ci(values: Iterable[float], n_bootstrap: int = 2000) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype="float64")
    if arr.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(1234)
    means = [float(rng.choice(arr, size=arr.size, replace=True).mean()) for _ in range(n_bootstrap)]
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def interpret_difference(metric: str, mean_difference: float) -> str:
    if abs(mean_difference) < 1e-9:
        return "No average change"
    direction = "higher" if mean_difference > 0 else "lower"
    if metric == "lpips":
        return f"Detail-aware is {direction}; lower LPIPS is usually better"
    return f"Detail-aware is {direction}"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_json(
    path: Path,
    args: argparse.Namespace,
    rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
    comparison_rows: list[dict[str, object]],
) -> None:
    payload = {
        "dataset_dir": args.dataset_dir,
        "limit": args.limit,
        "keep_ratios": args.keep_ratios,
        "mission": args.mission,
        "n_rows": len(rows),
        "summary": summary_rows,
        "paired_statistics": comparison_rows,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_figures(output_dir: Path, summary_rows: list[dict[str, object]]) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    path = output_dir / "detail_term_ablation_sus_detector.png"
    keep_ratios = sorted({float(row["keep_ratio"]) for row in summary_rows})
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), dpi=220)
    for variant in VARIANTS:
        variant_rows = sorted([row for row in summary_rows if row["variant"] == variant], key=lambda row: float(row["keep_ratio"]))
        x = [float(row["keep_ratio"]) for row in variant_rows]
        axes[0].plot(x, [float(row["semantic_utility_score_mean"]) for row in variant_rows], marker="o", label=variant)
        axes[1].plot(x, [float(row["detector_retention_mean"]) for row in variant_rows], marker="o", label=variant)
    axes[0].set_title("SUS")
    axes[0].set_xlabel("Token retention")
    axes[0].set_ylabel("Semantic Utility Score")
    axes[1].set_title("Detector Retention")
    axes[1].set_xlabel("Token retention")
    axes[1].set_ylabel("Retention")
    for axis in axes:
        axis.set_xticks(keep_ratios)
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def write_report(
    output_dir: Path,
    image_paths: list[Path],
    keep_ratios: list[float],
    summary_rows: list[dict[str, object]],
    comparison_rows: list[dict[str, object]],
    figure_path: Path | None,
) -> Path:
    path = output_dir / "detail_term_ablation_report.md"
    lines = [
        "# Model Improvement Step 3: Sentinel-2 Detail-Term Ablation",
        "",
        "## Purpose",
        "This experiment tests whether the detail-aware token selector improves real Sentinel-2 wildfire patch results compared with an ablated selector that removes the structural detail term.",
        "",
        "## Experimental Setup",
        f"- Images evaluated: {len(image_paths)}",
        f"- Token retention ratios: {', '.join(f'{ratio:.2f}' for ratio in keep_ratios)}",
        "- Full selector: utility + token entropy + detail + cost.",
        "- Ablation: utility + token entropy + cost, with detail removed.",
        "- Metrics: SUS, detector retention, compression ratio, bandwidth saved, PSNR, SSIM, and LPIPS.",
        "",
        "## Summary",
        summary_table(summary_rows),
        "",
        "## Paired Statistical Comparison",
        comparison_table(comparison_rows),
        "",
        "## Interpretation",
        "Positive differences mean the detail-aware selector improved the metric relative to the no-detail ablation. LPIPS should be interpreted separately because lower values are usually better.",
        "",
        "This experiment is the first dataset-level check after the controlled token-grid result. If the gains are small or mixed, the scientific conclusion is still useful: detail-aware scoring improves boundary selection behavior, but its dataset-level value depends on detector sensitivity, token resolution, and reconstruction quality.",
    ]
    if figure_path is not None:
        lines.extend(["", f"Figure: `{figure_path}`"])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def summary_table(rows: list[dict[str, object]]) -> str:
    lines = [
        "| Variant | Keep Ratio | Images | SUS | Detector Retention | Compression Ratio | Bandwidth Saved | PSNR | SSIM | LPIPS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: (float(item["keep_ratio"]), str(item["variant"]))):
        lines.append(
            "| {variant} | {keep:.2f} | {n} | {sus:.2f} | {det:.3f} | {cr:.2f}x | {bw:.2f}% | {psnr:.2f} | {ssim:.3f} | {lpips:.4f} |".format(
                variant=row["variant"],
                keep=float(row["keep_ratio"]),
                n=row["n_images"],
                sus=float(row.get("semantic_utility_score_mean", 0.0)),
                det=float(row.get("detector_retention_mean", 0.0)),
                cr=float(row.get("compression_ratio_mean", 0.0)),
                bw=float(row.get("bandwidth_saved_percent_mean", 0.0)),
                psnr=float(row.get("psnr_mean", 0.0)),
                ssim=float(row.get("ssim_mean", 0.0)),
                lpips=float(row.get("lpips_mean", 0.0)),
            )
        )
    return "\n".join(lines)


def comparison_table(rows: list[dict[str, object]]) -> str:
    preferred = {"semantic_utility_score", "detector_retention", "psnr", "ssim", "lpips"}
    selected = [row for row in rows if row["metric"] in preferred]
    lines = [
        "| Keep Ratio | Metric | Mean Difference | 95% Bootstrap CI | Paired t p | Wilcoxon p | Cohen's d |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(selected, key=lambda item: (float(item["keep_ratio"]), str(item["metric"]))):
        lines.append(
            "| {keep:.2f} | {metric} | {diff:.4f} | [{lo:.4f}, {hi:.4f}] | {tp} | {wp} | {d:.3f} |".format(
                keep=float(row["keep_ratio"]),
                metric=row["metric"],
                diff=float(row["mean_difference"]),
                lo=float(row["bootstrap_ci_low"]),
                hi=float(row["bootstrap_ci_high"]),
                tp=row["paired_t_p"],
                wp=row["wilcoxon_p"],
                d=float(row["cohens_d"]),
            )
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
