"""Generate project diagrams and result graphs for CompressAI.

The figures created here are designed for PhD proposals, supervisor outreach,
README documentation, and investor/incubator explanations. The metric plots use
the latest 500-patch Sentinel-2 benchmark results when available.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = ROOT / "results/model_improvement_step20_sentinel2_500_promotion_check/vqvae_checkpoint_comparison_summary.csv"
DEFAULT_STATS = ROOT / "results/model_improvement_step20_sentinel2_500_promotion_check/vqvae_checkpoint_paired_statistics.csv"
DEFAULT_OUTPUT = ROOT / "figures/project_diagrams"

COLORS = {
    "satellite": "#2E86AB",
    "semantic": "#41A368",
    "token": "#6C63FF",
    "transmission": "#F28E2B",
    "evaluation": "#C44536",
    "neutral": "#3D405B",
    "light": "#F6F8FB",
    "line": "#263238",
    "original": "#2E86AB",
    "candidate": "#F28E2B",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CompressAI diagrams and benchmark graphs.")
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    create_system_architecture(args.output_dir)
    create_working_pipeline(args.output_dir)
    create_token_transmission_diagram(args.output_dir)
    create_experiment_pipeline(args.output_dir)
    create_career_positioning_diagram(args.output_dir)

    if args.summary.exists() and args.stats.exists():
        summary = pd.read_csv(args.summary)
        stats = pd.read_csv(args.stats)
        create_metric_comparison(summary, args.output_dir)
        create_paired_effects(stats, args.output_dir)
        create_model_decision_matrix(summary, args.output_dir)

    print(args.output_dir)


def save(fig: plt.Figure, output_dir: Path, name: str) -> None:
    for suffix in ("png", "svg"):
        fig.savefig(output_dir / f"{name}.{suffix}", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def setup_canvas(width: float = 13, height: float = 7, title: str | None = None) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    if title:
        ax.text(0.5, 0.96, title, ha="center", va="center", fontsize=18, fontweight="bold", color=COLORS["line"])
    return fig, ax


def box(ax: plt.Axes, xy: tuple[float, float], size: tuple[float, float], text: str, color: str, fontsize: int = 10) -> None:
    x, y = xy
    w, h = size
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.4,
        edgecolor=COLORS["line"],
        facecolor=color,
    )
    ax.add_patch(patch)
    wrapped = "\n".join(textwrap.wrap(text, width=24))
    ax.text(x + w / 2, y + h / 2, wrapped, ha="center", va="center", fontsize=fontsize, color="white", fontweight="bold")


def label_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    size: tuple[float, float],
    text: str,
    fontsize: int = 9,
    wrap_width: int = 26,
) -> None:
    x, y = xy
    w, h = size
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.01,rounding_size=0.01",
        linewidth=1.0,
        edgecolor="#B7C0CC",
        facecolor=COLORS["light"],
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, "\n".join(textwrap.wrap(text, width=wrap_width)), ha="center", va="center", fontsize=fontsize, color=COLORS["line"])


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = "#263238", rad: float = 0.0) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=1.6,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
        )
    )


def create_system_architecture(output_dir: Path) -> None:
    fig, ax = setup_canvas(title="CompressAI System Architecture")
    box(ax, (0.04, 0.72), (0.18, 0.11), "Streamlit Mission Dashboard", COLORS["satellite"])
    box(ax, (0.04, 0.48), (0.18, 0.11), "FastAPI Backend", COLORS["neutral"])
    box(ax, (0.30, 0.72), (0.18, 0.11), "Semantic Utility Service", COLORS["semantic"])
    box(ax, (0.30, 0.48), (0.18, 0.11), "Encoder / Decoder Services", COLORS["token"])
    box(ax, (0.30, 0.24), (0.18, 0.11), "Metrics + Transmission Services", COLORS["evaluation"])
    box(ax, (0.58, 0.72), (0.18, 0.11), "VQ-VAE Checkpoints", COLORS["token"])
    box(ax, (0.58, 0.48), (0.18, 0.11), "Token Prioritisation", COLORS["transmission"])
    box(ax, (0.58, 0.24), (0.18, 0.11), "Benchmark + Statistics", COLORS["evaluation"])
    box(ax, (0.81, 0.48), (0.15, 0.11), "Reports, Figures, Evidence Pack", COLORS["semantic"])

    arrow(ax, (0.22, 0.535), (0.30, 0.535))
    arrow(ax, (0.22, 0.775), (0.30, 0.775))
    arrow(ax, (0.48, 0.775), (0.58, 0.775))
    arrow(ax, (0.48, 0.535), (0.58, 0.535))
    arrow(ax, (0.48, 0.295), (0.58, 0.295))
    arrow(ax, (0.76, 0.535), (0.81, 0.535))
    arrow(ax, (0.67, 0.72), (0.67, 0.59))
    arrow(ax, (0.67, 0.48), (0.67, 0.35))
    arrow(ax, (0.13, 0.72), (0.13, 0.59))

    label_box(
        ax,
        (0.04, 0.09),
        (0.92, 0.08),
        "Goal: transmit satellite imagery according to wildfire mission utility, then evaluate reconstruction quality and detector retention.",
        wrap_width=95,
    )
    save(fig, output_dir, "01_system_architecture")


def create_working_pipeline(output_dir: Path) -> None:
    fig, ax = setup_canvas(title="Working Pipeline: Semantic Utility-Aware Compression")
    stages = [
        ("Sentinel-2 / Wildfire Image", COLORS["satellite"]),
        ("Wildfire Utility Detector", COLORS["semantic"]),
        ("Semantic Utility Map", COLORS["semantic"]),
        ("VQ-VAE Token Encoder", COLORS["token"]),
        ("Utility-Aware Token Ranking", COLORS["transmission"]),
        ("Bandwidth-Constrained Transmission", COLORS["transmission"]),
        ("Decoder Reconstruction", COLORS["token"]),
        ("SUS + Detector Retention Evaluation", COLORS["evaluation"]),
    ]
    x_positions = [0.04, 0.28, 0.52, 0.76]
    y_positions = [0.67, 0.36]
    coords: list[tuple[float, float]] = []
    for row, y in enumerate(y_positions):
        xs = x_positions if row == 0 else list(reversed(x_positions))
        for x in xs:
            coords.append((x, y))
    for (text, color), (x, y) in zip(stages, coords):
        box(ax, (x, y), (0.18, 0.13), text, color)
    for a, b in zip(coords, coords[1:]):
        arrow(ax, (a[0] + 0.18, a[1] + 0.065), (b[0], b[1] + 0.065), rad=0.0 if abs(a[1] - b[1]) < 0.05 else -0.18)
    save(fig, output_dir, "02_working_pipeline")


def create_token_transmission_diagram(output_dir: Path) -> None:
    fig, ax = setup_canvas(title="Token-Based Semantic Transmission")
    label_box(ax, (0.05, 0.77), (0.20, 0.09), "Image is encoded into a grid of learned tokens")
    label_box(ax, (0.40, 0.77), (0.20, 0.09), "Utility map assigns mission value to each region")
    label_box(ax, (0.74, 0.77), (0.20, 0.09), "Only high-value tokens are sent under bandwidth limits")

    token_x0, token_y0 = 0.07, 0.48
    cell = 0.035
    for i in range(6):
        for j in range(6):
            value = (i + j) / 10
            color = COLORS["token"] if value > 0.55 else "#C8D1E0"
            ax.add_patch(plt.Rectangle((token_x0 + j * cell, token_y0 + i * cell), cell * 0.9, cell * 0.9, facecolor=color, edgecolor="white"))

    util_x0, util_y0 = 0.42, 0.48
    for i in range(6):
        for j in range(6):
            high = (2 <= i <= 4 and 2 <= j <= 5) or (i == 1 and j == 4)
            color = COLORS["semantic"] if high else "#DDE6DD"
            ax.add_patch(plt.Rectangle((util_x0 + j * cell, util_y0 + i * cell), cell * 0.9, cell * 0.9, facecolor=color, edgecolor="white"))

    send_x0, send_y0 = 0.77, 0.48
    for i in range(6):
        for j in range(6):
            keep = (2 <= i <= 4 and 2 <= j <= 5) or (i == 1 and j == 4) or (i == 5 and j == 5)
            color = COLORS["transmission"] if keep else "#ECEFF3"
            ax.add_patch(plt.Rectangle((send_x0 + j * cell, send_y0 + i * cell), cell * 0.9, cell * 0.9, facecolor=color, edgecolor="white"))

    arrow(ax, (0.29, 0.58), (0.39, 0.58))
    arrow(ax, (0.64, 0.58), (0.74, 0.58))
    label_box(
        ax,
        (0.11, 0.23),
        (0.78, 0.09),
        "Transmission policy: preserve tokens with high wildfire relevance first; prune low-utility background tokens when downlink is constrained.",
        wrap_width=85,
    )
    save(fig, output_dir, "03_token_transmission_working_diagram")


def create_experiment_pipeline(output_dir: Path) -> None:
    fig, ax = setup_canvas(title="Research Benchmark and Validation Pipeline")
    items = [
        ("Datasets\nCEMS-HLS, Sentinel-2", 0.07, 0.70, COLORS["satellite"]),
        ("Model Variants\nOriginal, fine-tuned, selector variants", 0.32, 0.70, COLORS["token"]),
        ("Compression Runs\nRetention and baseline studies", 0.57, 0.70, COLORS["transmission"]),
        ("Metric Computation\nSUS, detector, PSNR, SSIM", 0.32, 0.42, COLORS["evaluation"]),
        ("Statistical Analysis\nCI, t-test, Wilcoxon", 0.57, 0.42, COLORS["evaluation"]),
        ("Research Outputs\nCSV, reports, figures, packages", 0.32, 0.16, COLORS["semantic"]),
    ]
    for text, x, y, color in items:
        box(ax, (x, y), (0.18, 0.12), text, color)
    arrow(ax, (0.25, 0.76), (0.32, 0.76))
    arrow(ax, (0.50, 0.76), (0.57, 0.76))
    arrow(ax, (0.66, 0.70), (0.66, 0.54))
    arrow(ax, (0.57, 0.48), (0.50, 0.48))
    arrow(ax, (0.41, 0.42), (0.41, 0.28))
    arrow(ax, (0.66, 0.42), (0.50, 0.22), rad=-0.20)
    save(fig, output_dir, "04_experiment_validation_pipeline")


def create_career_positioning_diagram(output_dir: Path) -> None:
    fig, ax = setup_canvas(title="Evidence Positioning: Patent, PhD, Innovator Founder, Global Talent")
    box(ax, (0.39, 0.58), (0.22, 0.13), "Core Technology\nSemantic Utility-Aware Satellite Compression", COLORS["neutral"])
    targets = [
        ("Patent/IP\ntechnical process, claims, novelty", 0.08, 0.72, COLORS["transmission"]),
        ("PhD Route\nresearch questions, experiments, publications", 0.70, 0.72, COLORS["satellite"]),
        ("Innovator Founder\ninnovative, viable, scalable business", 0.08, 0.28, COLORS["semantic"]),
        ("Global Talent\ninnovation, evidence, expert validation", 0.70, 0.28, COLORS["token"]),
    ]
    for text, x, y, color in targets:
        box(ax, (x, y), (0.22, 0.13), text, color)
    arrow(ax, (0.39, 0.67), (0.30, 0.77), rad=0.08)
    arrow(ax, (0.61, 0.67), (0.70, 0.77), rad=-0.08)
    arrow(ax, (0.39, 0.61), (0.30, 0.37), rad=-0.08)
    arrow(ax, (0.61, 0.61), (0.70, 0.37), rad=0.08)
    label_box(
        ax,
        (0.20, 0.10),
        (0.60, 0.08),
        "Same project, different framing: keep invention details private; share validated results and research narrative publicly.",
        wrap_width=70,
    )
    save(fig, output_dir, "05_career_evidence_positioning")


def create_metric_comparison(summary: pd.DataFrame, output_dir: Path) -> None:
    summary = summary.set_index("checkpoint").loc[["original", "mixed_regularized"]].reset_index()
    labels = ["Original", "Regularized mixed"]
    metrics = [
        ("semantic_utility_score_mean", "SUS", ""),
        ("detector_retention_mean", "Detector retention", ""),
        ("psnr_mean", "PSNR", "dB"),
        ("ssim_mean", "SSIM", ""),
        ("compression_ratio_mean", "Compression ratio", "x"),
        ("bandwidth_saved_percent_mean", "Bandwidth saved", "%"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    fig.suptitle("Sentinel-2 500-Patch Benchmark: Original vs Regularized VQ-VAE", fontsize=16, fontweight="bold")
    for ax, (column, title, unit) in zip(axes.flatten(), metrics):
        values = summary[column].to_numpy()
        bars = ax.bar(labels, values, color=[COLORS["original"], COLORS["candidate"]], width=0.55)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}{unit}", ha="center", va="bottom", fontsize=9)
        low = min(values) * 0.96 if min(values) > 1 else max(0, min(values) - 0.05)
        high = max(values) * 1.04 if max(values) > 1 else min(1.05, max(values) + 0.05)
        if abs(high - low) < 1e-6:
            high += 1
        ax.set_ylim(low, high)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, output_dir, "06_sentinel2_metric_comparison")


def create_paired_effects(stats: pd.DataFrame, output_dir: Path) -> None:
    selected = stats[stats["metric"].isin(["semantic_utility_score", "detector_retention", "psnr", "ssim"])].copy()
    names = {
        "semantic_utility_score": "SUS",
        "detector_retention": "Detector retention",
        "psnr": "PSNR",
        "ssim": "SSIM",
    }
    selected["label"] = selected["metric"].map(names)
    selected = selected.set_index("metric").loc[["semantic_utility_score", "detector_retention", "psnr", "ssim"]].reset_index()
    y = range(len(selected))
    diff = selected["mean_difference_candidate_minus_baseline"]
    low = selected["bootstrap_95ci_low"]
    high = selected["bootstrap_95ci_high"]
    xerr = [diff - low, high - diff]

    fig, ax = plt.subplots(figsize=(10, 5.4))
    ax.errorbar(diff, list(y), xerr=xerr, fmt="o", color=COLORS["candidate"], ecolor=COLORS["neutral"], elinewidth=2, capsize=5)
    ax.axvline(0, color="#444", linewidth=1.2, linestyle="--")
    ax.set_yticks(list(y), selected["label"])
    ax.set_xlabel("Mean difference: regularized mixed-domain minus original")
    ax.set_title("Paired Effects With Bootstrap 95% Confidence Intervals", fontsize=15, fontweight="bold")
    ax.grid(axis="x", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    for yi, row in zip(y, selected.to_dict("records")):
        ax.text(row["bootstrap_95ci_high"], yi + 0.13, f"p={row['paired_t_p_value']:.2g}", fontsize=9, color=COLORS["line"])
    fig.tight_layout()
    save(fig, output_dir, "07_paired_effects_ci")


def create_model_decision_matrix(summary: pd.DataFrame, output_dir: Path) -> None:
    summary = summary.set_index("checkpoint")
    original = summary.loc["original"]
    candidate = summary.loc["mixed_regularized"]
    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.scatter(original["detector_retention_mean"], original["psnr_mean"], s=220, color=COLORS["original"], label="Original VQ-VAE")
    ax.scatter(candidate["detector_retention_mean"], candidate["psnr_mean"], s=220, color=COLORS["candidate"], label="Regularized mixed-domain VQ-VAE")
    ax.annotate("Safer detector retention\ncurrent default", (original["detector_retention_mean"], original["psnr_mean"]), xytext=(0.855, 21.55), arrowprops={"arrowstyle": "->", "color": COLORS["line"]})
    ax.annotate("Better reconstruction\nexperimental candidate", (candidate["detector_retention_mean"], candidate["psnr_mean"]), xytext=(0.812, 22.95), arrowprops={"arrowstyle": "->", "color": COLORS["line"]})
    ax.set_xlabel("Detector retention")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Model Promotion Trade-Off", fontsize=15, fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0.80, 0.88)
    ax.set_ylim(20.8, 23.2)
    fig.tight_layout()
    save(fig, output_dir, "08_model_decision_tradeoff")


if __name__ == "__main__":
    main()
