"""scripts.generate_sentinel2_publication_evidence

Plain-English purpose: Command-line runners for datasets, benchmarks, reports, and exports.

This file is part of the public Space Semantic Compression research demo.
It is documented at module level so researchers, supervisors, and non-specialist
readers can understand where the file fits before reading implementation details.
"""

from __future__ import annotations

import csv
import argparse
import json
import math
import sys
import textwrap
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


METRICS = [
    "semantic_utility_score",
    "detector_retention",
    "psnr",
    "ssim",
    "lpips",
    "compression_ratio",
    "bandwidth_saved_percent",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: object) -> float | None:
    try:
        if value in {"", None}:
            return None
        return float(value)
    except Exception:
        return None


def mean_ci(values: list[float], n_bootstrap: int = 1000) -> tuple[float, float, float, float]:
    arr = np.asarray(values, dtype="float64")
    if arr.size == 0:
        return math.nan, math.nan, math.nan, math.nan
    rng = np.random.default_rng(2026)
    means = [float(rng.choice(arr, size=arr.size, replace=True).mean()) for _ in range(n_bootstrap)]
    return (
        round(float(arr.mean()), 6),
        round(float(arr.std(ddof=1)) if arr.size > 1 else 0.0, 6),
        round(float(np.quantile(means, 0.025)), 6),
        round(float(np.quantile(means, 0.975)), 6),
    )


def robust_stats(rows: list[dict[str, str]], group_key: str) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row.get(group_key, "")].append(row)
    output: list[dict[str, object]] = []
    for group, group_rows in grouped.items():
        out: dict[str, object] = {group_key: group, "n_images": len({row.get("image") for row in group_rows})}
        for metric in METRICS:
            values = [value for row in group_rows if (value := as_float(row.get(metric))) is not None]
            if not values:
                continue
            mean, std, low, high = mean_ci(values)
            out[f"{metric}_mean"] = mean
            out[f"{metric}_std"] = std
            out[f"{metric}_ci_low"] = low
            out[f"{metric}_ci_high"] = high
        output.append(out)
    return sorted(output, key=lambda row: str(row.get(group_key, "")))


def operating_points(retention_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    summary = robust_stats(retention_rows, "keep_ratio")
    candidates: list[dict[str, object]] = []
    for row in summary:
        utility = float(row.get("semantic_utility_score_mean", 0.0))
        bandwidth = float(row.get("bandwidth_saved_percent_mean", 0.0))
        compression = float(row.get("compression_ratio_mean", 0.0))
        detector = float(row.get("detector_retention_mean", 0.0))
        score = 0.45 * utility + 0.25 * bandwidth + 0.20 * min(compression, 250.0) / 250.0 * 100.0 + 0.10 * detector * 100.0
        out = dict(row)
        out["operating_score"] = round(score, 6)
        candidates.append(out)
    if not candidates:
        return []
    best = max(candidates, key=lambda row: float(row["operating_score"]))
    for row in candidates:
        row["recommendation"] = "recommended" if row is best else ""
    return candidates


def pareto_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    frontier: list[dict[str, object]] = []
    for row in rows:
        util = float(row.get("semantic_utility_score_mean", 0.0))
        bw = float(row.get("bandwidth_saved_percent_mean", 0.0))
        dominated = False
        for other in rows:
            if other is row:
                continue
            other_util = float(other.get("semantic_utility_score_mean", 0.0))
            other_bw = float(other.get("bandwidth_saved_percent_mean", 0.0))
            if other_util >= util and other_bw >= bw and (other_util > util or other_bw > bw):
                dominated = True
                break
        if not dominated:
            out = dict(row)
            out["pareto_frontier"] = True
            frontier.append(out)
    return frontier


def make_figures(output_dir: Path, operating: list[dict[str, object]], frontier: list[dict[str, object]]) -> list[str]:
    import matplotlib.pyplot as plt

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    if not operating:
        return paths
    keep = [float(row["keep_ratio"]) for row in operating]
    sus = [float(row.get("semantic_utility_score_mean", 0.0)) for row in operating]
    bw = [float(row.get("bandwidth_saved_percent_mean", 0.0)) for row in operating]
    comp = [float(row.get("compression_ratio_mean", 0.0)) for row in operating]
    det = [float(row.get("detector_retention_mean", 0.0)) for row in operating]

    for y, ylabel, name in [
        (sus, "SUS", "pareto_retention_sus.png"),
        (bw, "Bandwidth saved (%)", "pareto_retention_bandwidth.png"),
        (det, "Detector retention", "detector_retention_curve.png"),
        (comp, "Compression ratio", "compression_ratio_curve.png"),
    ]:
        fig, ax = plt.subplots(figsize=(6.8, 4.2))
        ax.plot(keep, y, marker="o", linewidth=2)
        ax.set_xlabel("Token retention ratio")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = figure_dir / name
        fig.savefig(path, dpi=300)
        fig.savefig(path.with_suffix(".pdf"))
        plt.close(fig)
        paths.append(str(path))

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.scatter(bw, sus, s=60, label="Retention points")
    if frontier:
        fx = [float(row.get("bandwidth_saved_percent_mean", 0.0)) for row in frontier]
        fy = [float(row.get("semantic_utility_score_mean", 0.0)) for row in frontier]
        ax.plot(fx, fy, marker="o", linewidth=2, label="Pareto frontier")
    for row in operating:
        ax.annotate(str(row.get("keep_ratio")), (float(row.get("bandwidth_saved_percent_mean", 0.0)), float(row.get("semantic_utility_score_mean", 0.0))), fontsize=7)
    ax.set_xlabel("Bandwidth saved (%)")
    ax.set_ylabel("SUS")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = figure_dir / "pareto_bandwidth_vs_sus.png"
    fig.savefig(path, dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    paths.append(str(path))
    return paths


def make_table_markdown(title: str, rows: list[dict[str, object]], keys: list[str], max_rows: int = 20) -> str:
    lines = [f"## {title}", "", "| " + " | ".join(keys) + " |", "| " + " | ".join(["---"] * len(keys)) + " |"]
    for row in rows[:max_rows]:
        lines.append("| " + " | ".join(str(row.get(key, "")) for key in keys) + " |")
    lines.append("")
    return "\n".join(lines)


def make_pdf(path: Path, title: str, markdown: str, figures: list[str]) -> None:
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    with PdfPages(path) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.07, 0.94, title, fontsize=16, weight="bold")
        y = 0.90
        for paragraph in markdown.splitlines():
            for line in textwrap.wrap(paragraph, width=98) or [""]:
                fig.text(0.07, y, line, fontsize=8.2)
                y -= 0.019
                if y < 0.07:
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)
                    fig = plt.figure(figsize=(8.27, 11.69))
                    y = 0.94
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        for figure in figures:
            source = Path(figure)
            if not source.exists() or source.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            fig, ax = plt.subplots(figsize=(8.27, 5.8))
            ax.imshow(mpimg.imread(source))
            ax.axis("off")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Sentinel-2 publication evidence tables and PDFs.")
    parser.add_argument("--output-root", type=Path, default=Path("results/earth_observation_validation_100_patch"))
    args = parser.parse_args()
    output_root = args.output_root
    sentinel_root = output_root / "sentinel2"
    tables_dir = output_root / "publication_tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    retention = read_csv(sentinel_root / "sentinel2" / "retention_study.csv")
    if not retention:
        retention = read_csv(sentinel_root / "retention_study.csv")
    baseline = read_csv(sentinel_root / "sentinel2" / "baseline_comparison.csv")
    if not baseline:
        baseline = read_csv(sentinel_root / "baseline_comparison.csv")
    ablation = read_csv(sentinel_root / "sentinel2" / "ablation_study.csv")
    if not ablation:
        ablation = read_csv(sentinel_root / "ablation_study.csv")
    stats = read_csv(sentinel_root / "statistical_significance.csv")
    cross = read_csv(output_root / "cross_dataset_comparison.csv")

    write_csv(output_root / "sentinel2_100_scene_results.csv", retention)
    baseline_stats = robust_stats(baseline, "method")
    retention_stats = robust_stats(retention, "keep_ratio")
    ablation_stats = robust_stats(ablation, "method")
    operating = operating_points(retention)
    frontier = pareto_rows(operating)

    write_csv(tables_dir / "table1_cross_dataset_comparison.csv", cross)
    write_csv(tables_dir / "table2_sentinel2_results.csv", baseline_stats)
    write_csv(tables_dir / "table3_ablation_study.csv", ablation_stats)
    write_csv(tables_dir / "table4_statistical_significance.csv", stats)
    write_csv(tables_dir / "table5_operating_point_analysis.csv", operating)
    write_csv(output_root / "operating_point_analysis.csv", operating)
    write_csv(output_root / "pareto_frontier.csv", frontier)

    figures = make_figures(output_root, operating, frontier)
    recommended = next((row for row in operating if row.get("recommendation") == "recommended"), {})
    report = "\n".join(
        [
            "# Sentinel-2 100-Scene Publication Evidence",
            "",
            f"Recommended operating point: token retention {recommended.get('keep_ratio', 'n/a')}.",
            f"At this point, SUS={recommended.get('semantic_utility_score_mean', 'n/a')}, detector retention={recommended.get('detector_retention_mean', 'n/a')}, bandwidth saved={recommended.get('bandwidth_saved_percent_mean', 'n/a')}%, compression ratio={recommended.get('compression_ratio_mean', 'n/a')}.",
            "",
            make_table_markdown("Table 1: Cross-Dataset Comparison", cross, ["dataset", "n_images", "sus_mean", "detector_retention_mean", "bandwidth_saved_percent_mean", "compression_ratio_mean"]),
            make_table_markdown("Table 2: Sentinel-2 Results", baseline_stats, ["method", "n_images", "semantic_utility_score_mean", "semantic_utility_score_std", "detector_retention_mean", "bandwidth_saved_percent_mean", "compression_ratio_mean", "psnr_mean", "ssim_mean", "lpips_mean"]),
            make_table_markdown("Table 3: Ablation Study", ablation_stats, ["method", "n_images", "semantic_utility_score_mean", "detector_retention_mean", "bandwidth_saved_percent_mean", "compression_ratio_mean"]),
            make_table_markdown("Table 4: Statistical Significance", stats, ["comparison", "metric", "n", "paired_t_p", "wilcoxon_p", "cohens_d"], max_rows=30),
            make_table_markdown("Table 5: Operating Point Analysis", operating, ["keep_ratio", "semantic_utility_score_mean", "detector_retention_mean", "bandwidth_saved_percent_mean", "compression_ratio_mean", "operating_score", "recommendation"]),
        ]
    )
    report_path = output_root / "sentinel2_100_publication_tables.md"
    report_path.write_text(report, encoding="utf-8")
    make_pdf(output_root / "semantic_utility_wildfire_earth_observation_preprint.pdf", "Revised Preprint: Sentinel-2 100-Scene Evidence", report, figures)
    make_pdf(output_root / "supervisor_package.pdf", "Supervisor Package: Sentinel-2 100-Scene Evidence", report, figures)
    print(f"results={output_root / 'sentinel2_100_scene_results.csv'}")
    print(f"tables={tables_dir}")
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
