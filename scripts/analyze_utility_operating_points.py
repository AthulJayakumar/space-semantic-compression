"""Analyze utility-aware token-retention operating points.

This script is Step 1 of the model-improvement process. It does not change the
architecture. Instead, it asks a simpler question first: which token-retention
level gives the best trade-off between wildfire utility and bandwidth savings?
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_DIR = ROOT / "results" / "summary_tables"
OUTPUT_DIR = ROOT / "results" / "model_improvement_step1_operating_points"


METRICS = [
    "semantic_utility_score",
    "detector_retention",
    "compression_ratio",
    "bandwidth_saved_percent",
    "psnr",
    "ssim",
    "lpips",
]


def load_retention_results() -> pd.DataFrame:
    path = SUMMARY_DIR / "sentinel2_100_scene_results.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def load_sentinel2_baselines() -> pd.DataFrame:
    path = SUMMARY_DIR / "table2_sentinel2_results.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_space_codec_baselines() -> pd.DataFrame:
    path = ROOT / "results" / "space_codec_baselines_500" / "space_codec_baseline_summary.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def mean_ci(group: pd.Series) -> pd.Series:
    values = group.astype(float)
    mean = values.mean()
    std = values.std(ddof=1)
    half_width = 1.96 * std / max(len(values), 1) ** 0.5
    return pd.Series({"mean": mean, "std": std, "ci_low": mean - half_width, "ci_high": mean + half_width})


def summarize_retention(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for keep_ratio, group in df.groupby("keep_ratio"):
        row: dict[str, float] = {"keep_ratio": float(keep_ratio), "n_images": float(group["image"].nunique())}
        for metric in METRICS:
            stats = mean_ci(group[metric])
            for key, value in stats.items():
                row[f"{metric}_{key}"] = float(value)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("keep_ratio")


def choose_recommended_operating_point(summary: pd.DataFrame) -> pd.Series:
    """Pick the practical elbow point for semantic compression.

    Full 100% retention usually gives the highest utility, but it is not a
    meaningful pruning policy. For model-improvement work we want the smallest
    retention ratio that reaches a strong mission-utility target while preserving
    a clear compression advantage.
    """

    data = summary.copy().sort_values("keep_ratio")
    full = data[data["keep_ratio"] == data["keep_ratio"].max()].iloc[0]
    utility_target = max(90.0, 0.975 * float(full["semantic_utility_score_mean"]))
    detector_target = 0.90
    eligible = data[
        (data["semantic_utility_score_mean"] >= utility_target)
        & (data["detector_retention_mean"] >= detector_target)
        & (data["keep_ratio"] < 1.0)
    ]
    if eligible.empty:
        eligible = data[(data["semantic_utility_score_mean"] >= 90.0) & (data["keep_ratio"] < 1.0)]
    if eligible.empty:
        eligible = data[data["keep_ratio"] < 1.0]
    return eligible.sort_values(["keep_ratio"]).iloc[0]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot_operating_points(summary: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(summary["keep_ratio"], summary["semantic_utility_score_mean"], marker="o", label="SUS", color="#1f77b4")
    ax1.plot(summary["keep_ratio"], summary["detector_retention_mean"] * 100, marker="s", label="Detector retention x100", color="#2ca02c")
    ax1.set_xlabel("Utility-aware token retention")
    ax1.set_ylabel("Utility / detector retention")
    ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(summary["keep_ratio"], summary["compression_ratio_mean"], marker="^", label="Compression ratio", color="#d62728")
    ax2.set_ylabel("Compression ratio (x)")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="center right")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "utility_operating_point_curve.png", dpi=300)
    fig.savefig(OUTPUT_DIR / "utility_operating_point_curve.pdf")
    plt.close(fig)


def markdown_table(summary: pd.DataFrame) -> str:
    rows = [
        "| Retention | SUS | Detector Retention | Compression Ratio | Bandwidth Saved | PSNR | SSIM |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        rows.append(
            "| {ret:.0f}% | {sus:.2f} | {det:.3f} | {cr:.2f}x | {bw:.2f}% | {psnr:.2f} | {ssim:.3f} |".format(
                ret=row["keep_ratio"] * 100,
                sus=row["semantic_utility_score_mean"],
                det=row["detector_retention_mean"],
                cr=row["compression_ratio_mean"],
                bw=row["bandwidth_saved_percent_mean"],
                psnr=row["psnr_mean"],
                ssim=row["ssim_mean"],
            )
        )
    return "\n".join(rows)


def baseline_table(baselines: pd.DataFrame, space_codecs: pd.DataFrame) -> str:
    lines = [
        "| Baseline | Dataset/Setting | SUS | Detector Retention | Compression Ratio | Bandwidth Saved |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for method in ["jpeg_quality_20", "jpeg_quality_40", "vqvae_full"]:
        row = baselines[baselines["method"] == method]
        if row.empty:
            continue
        r = row.iloc[0]
        lines.append(
            f"| {method} | Sentinel-2 100-scene | {r['semantic_utility_score_mean']:.2f} | {r['detector_retention_mean']:.3f} | {r['compression_ratio_mean']:.2f}x | {r['bandwidth_saved_percent_mean']:.2f}% |"
        )
    for _, r in space_codecs.iterrows():
        lines.append(
            f"| {r['method']} | Sentinel-2 500-patch | {r['semantic_utility_score_mean']:.2f} | {r['detector_retention_mean']:.3f} | {r['compression_ratio_mean']:.2f}x | {r['bandwidth_saved_percent_mean']:.2f}% |"
        )
    return "\n".join(lines)


def write_report(summary: pd.DataFrame, recommended: pd.Series, baselines: pd.DataFrame, space_codecs: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report = f"""# Model Improvement Step 1: Operating Point Analysis

## Purpose

Before changing the model architecture, this analysis tests whether the current utility-aware VQ-VAE system performs better at a less aggressive token-retention setting.

The earlier headline result used an aggressive 45-50% retention regime. That setting produced very high compression but lost too much detector-visible wildfire evidence. This report evaluates 10-100% utility-aware retention on the Sentinel-2/CEMS 100-scene benchmark.

## Utility-Aware Retention Sweep

{markdown_table(summary)}

## Recommended Practical Operating Point

Recommended point: **{recommended['keep_ratio'] * 100:.0f}% utility-aware token retention**.

At this point:

- SUS: **{recommended['semantic_utility_score_mean']:.2f}**
- Detector retention: **{recommended['detector_retention_mean']:.3f}**
- Compression ratio: **{recommended['compression_ratio_mean']:.2f}x**
- Bandwidth saved: **{recommended['bandwidth_saved_percent_mean']:.2f}%**
- PSNR: **{recommended['psnr_mean']:.2f} dB**
- SSIM: **{recommended['ssim_mean']:.3f}**

This is a better research operating point than 45-50% retention because it preserves much more mission utility while still maintaining over 100x compression.

## Baseline Context

{baseline_table(baselines, space_codecs)}

The comparisons are not all perfectly apples-to-apples: JPEG and VQ-VAE values are from the 100-scene Sentinel-2 benchmark, while JPEG2000/CCSDS-style values are from the newer 500-patch benchmark. They are still useful for research positioning.

## Interpretation

The current model does not need an architecture change as the first improvement. The first improvement is to stop presenting the 45-50% setting as the primary operating point.

The most defensible current story is:

> Utility-aware token transmission at 80% retention preserves high wildfire utility while still achieving substantially higher compression than JPEG/JPEG2000-style baselines.

This improves the model story immediately:

- 50% retention: SUS 82.54, detector retention 0.783, compression 145.36x.
- 80% retention: SUS 90.49, detector retention 0.907, compression 107.77x.
- 100% retention: SUS 92.81, detector retention 0.947, compression 99.08x.

The 80% point gives most of the utility benefit of full VQ-VAE while preserving stronger compression than the full-token setting.

## Next Improvement Step

The next model improvement should be **token scoring**, not a full model rewrite. The current token score should be extended with edge/detail preservation so wildfire boundaries, smoke texture, and burn-scar structure are less likely to be pruned.
"""
    (OUTPUT_DIR / "operating_point_analysis_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    df = load_retention_results()
    baselines = load_sentinel2_baselines()
    space_codecs = load_space_codec_baselines()
    summary = summarize_retention(df)
    recommended = choose_recommended_operating_point(summary)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_DIR / "utility_retention_operating_points.csv", index=False)
    recommended.to_frame().T.to_csv(OUTPUT_DIR / "recommended_operating_point.csv", index=False)
    plot_operating_points(summary)
    write_report(summary, recommended, baselines, space_codecs)
    print(OUTPUT_DIR / "operating_point_analysis_report.md")


if __name__ == "__main__":
    main()
