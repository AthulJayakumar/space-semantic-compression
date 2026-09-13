"""scripts.generate_publication_ready_revision

Plain-English purpose: Command-line runners for datasets, benchmarks, reports, and exports.

This file is part of the public Space Semantic Compression research demo.
It is documented at module level so researchers, supervisors, and non-specialist
readers can understand where the file fits before reading implementation details.
"""

from __future__ import annotations

import csv
import textwrap
from pathlib import Path


ROOT = Path("results/publication_ready_revision")
FIGURES = ROOT / "figures"
REPORTS = Path("reports")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def method_label(method: str) -> str:
    return {
        "jpeg_quality_20": "JPEG20",
        "jpeg_quality_40": "JPEG40",
        "jpeg_quality_60": "JPEG60",
        "jpeg_quality_80": "JPEG80",
        "vqvae_full": "VQ-VAE Full",
        "vqvae_random_45": "Random Tokens",
        "vqvae_entropy_45": "Entropy Tokens",
        "vqvae_utility_45": "Utility-Aware",
    }.get(method, method)


def sentinel_rows() -> list[dict[str, str]]:
    return read_csv(Path("results/earth_observation_validation_100_patch/publication_tables/table2_sentinel2_results.csv"))


def row_by_method(rows: list[dict[str, str]], method: str) -> dict[str, str]:
    return next(row for row in rows if row["method"] == method)


def generate_figures(rows: list[dict[str, str]]) -> list[Path]:
    import matplotlib.pyplot as plt

    FIGURES.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    colors = {
        "jpeg_quality_20": "#4C78A8",
        "jpeg_quality_40": "#4C78A8",
        "jpeg_quality_60": "#4C78A8",
        "jpeg_quality_80": "#4C78A8",
        "vqvae_full": "#72B7B2",
        "vqvae_random_45": "#B279A2",
        "vqvae_entropy_45": "#F58518",
        "vqvae_utility_45": "#E45756",
    }

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for row in rows:
        method = row["method"]
        ax.scatter(
            f(row, "compression_ratio_mean"),
            f(row, "semantic_utility_score_mean"),
            s=130 if method == "vqvae_utility_45" else 70,
            color=colors.get(method, "gray"),
            marker="*" if method == "vqvae_utility_45" else "o",
            edgecolor="black",
            linewidth=0.6,
            label=method_label(method),
        )
        ax.annotate(method_label(method), (f(row, "compression_ratio_mean"), f(row, "semantic_utility_score_mean")), xytext=(5, 3), textcoords="offset points", fontsize=7)
    ax.axvspan(100, 170, color="#E45756", alpha=0.08, label="Utility-aware high-compression region")
    ax.set_xscale("log")
    ax.set_xlabel("Compression ratio (x, log scale)")
    ax.set_ylabel("Semantic Utility Score (SUS)")
    ax.set_title("Utility vs Compression Ratio on Sentinel-2 CEMS (n=100)")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        path = FIGURES / f"utility_vs_compression_ratio.{suffix}"
        fig.savefig(path, dpi=300)
        paths.append(path)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    selected = [row for row in rows if row["method"].startswith("jpeg") or row["method"] in {"vqvae_full", "vqvae_utility_45"}]
    for row in selected:
        method = row["method"]
        ax.scatter(
            f(row, "bandwidth_saved_percent_mean"),
            f(row, "detector_retention_mean"),
            s=130 if method == "vqvae_utility_45" else 70,
            color=colors.get(method, "gray"),
            marker="*" if method == "vqvae_utility_45" else "o",
            edgecolor="black",
            linewidth=0.6,
        )
        ax.annotate(method_label(method), (f(row, "bandwidth_saved_percent_mean"), f(row, "detector_retention_mean")), xytext=(5, 3), textcoords="offset points", fontsize=7)
    ax.axvline(f(row_by_method(rows, "vqvae_utility_45"), "bandwidth_saved_percent_mean"), color="#E45756", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Bandwidth saved (%)")
    ax.set_ylabel("Detector retention")
    ax.set_title("Detector Retention vs Bandwidth Saved")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        path = FIGURES / f"detector_retention_vs_bandwidth.{suffix}"
        fig.savefig(path, dpi=300)
        paths.append(path)
    plt.close(fig)
    return paths


def satellite_implications(rows: list[dict[str, str]]) -> str:
    utility = row_by_method(rows, "vqvae_utility_45")
    jpeg20 = row_by_method(rows, "jpeg_quality_20")
    jpeg80 = row_by_method(rows, "jpeg_quality_80")
    raw_mib_per_image = 512 * 512 * 3 / (1024 * 1024)
    daily_images = 1000
    raw_daily = raw_mib_per_image * daily_images

    def volume(row: dict[str, str]) -> float:
        return raw_daily / f(row, "compression_ratio_mean")

    util_daily = volume(utility)
    jpeg20_daily = volume(jpeg20)
    jpeg80_daily = volume(jpeg80)
    util_vs_jpeg20 = (1 - util_daily / jpeg20_daily) * 100
    util_vs_jpeg80 = (1 - util_daily / jpeg80_daily) * 100
    text = f"""# Practical Satellite Communication Implications

## Assumptions
- Representative onboard image unit: 512 x 512 RGB Sentinel-2 wildfire patch.
- Uncompressed volume: {raw_mib_per_image:.2f} MiB per image.
- Daily acquisition scenario: {daily_images} wildfire-relevant images per day.
- This is a communication-volume analysis, not a complete spacecraft link-budget model.

## Daily Transmitted Volume
- Raw uncompressed transmission: {raw_daily:.1f} MiB/day.
- JPEG20: {jpeg20_daily:.2f} MiB/day at {f(jpeg20, 'compression_ratio_mean'):.2f}x compression.
- JPEG80: {jpeg80_daily:.2f} MiB/day at {f(jpeg80, 'compression_ratio_mean'):.2f}x compression.
- Utility-aware VQ-VAE: {util_daily:.2f} MiB/day at {f(utility, 'compression_ratio_mean'):.2f}x compression.

## Interpretation
Utility-aware compression reduces transmitted volume by {util_vs_jpeg20:.1f}% relative to JPEG20 and {util_vs_jpeg80:.1f}% relative to JPEG80 under the stated patch-volume assumption. This explains why JPEG can be superior in visual fidelity and detector retention while still being less attractive for severe downlink constraints.

## Practical Meaning
For a CubeSat or smallsat wildfire-monitoring payload, the utility-aware operating point is not positioned as a universal replacement for JPEG. It is positioned as a high-compression emergency mode for resource-constrained downlinks, intermittent connectivity, or onboard triage where preserving mission utility per transmitted bit is more important than preserving all visual detail.
"""
    (ROOT / "satellite_implications.md").write_text(text, encoding="utf-8")
    return text


def earth_observation_discussion() -> str:
    text = """# Implications for Earth Observation Systems

Wildfire monitoring is time-sensitive: downlinked imagery is valuable when it supports rapid situational awareness, prioritization, and response. The Sentinel-2 validation shows that semantic token prioritization can retain measurable wildfire utility while substantially reducing communication volume. This is relevant to Earth Observation systems where onboard storage, contact windows, and downlink capacity constrain operational decision-making.

For ESA and Copernicus-style disaster monitoring, the result suggests a research pathway in which onboard AI triages wildfire-relevant content before transmission. For TU Delft-style remote sensing research, the contribution is a controlled evaluation of semantic compression trade-offs under measurable utility, distortion, and communication metrics. For SnT and space systems research, the platform connects edge intelligence, semantic communication, and satellite resource constraints in a reproducible experimental pipeline.

The current evidence should be read as a validation of feasibility, not as an operational claim. JPEG remains strong for image fidelity and detector preservation, while semantic token selection is most relevant when the operating point is dominated by bandwidth, intermittent connectivity, and mission utility per bit.
"""
    (ROOT / "earth_observation_discussion.md").write_text(text, encoding="utf-8")
    return text


def limitations_future_work() -> str:
    text = """# Threats to Validity, Limitations, and Future Work

## Threats to Validity and Limitations
1. Sentinel-2 size limitations: the current benchmark uses 100 fixed-size RGB CEMS patches. This is statistically stronger than the initial sample result, but not yet a full georeferenced Sentinel-2 archive evaluation.
2. Detector dependence: SUS and detector retention depend on the wildfire utility detector. Detector bias can affect utility estimates, especially when smoke, burn scars, and active fire signatures vary by sensor and geography.
3. SUS formulation assumptions: SUS is fixed in this work and provides a useful mission-oriented proxy, but its weighting reflects a specific prioritization of detector confidence, object retention, relevance mass, and important-region preservation.
4. Dataset bias: DFire, FLAME, and Sentinel-2/CEMS differ in viewpoint, resolution, labels, and acquisition conditions. Cross-dataset comparisons should therefore be interpreted as generalization evidence, not as identical-distribution testing.
5. Simulated communication model: bandwidth savings and daily-volume examples use simplified assumptions and do not replace a full spacecraft link budget.
6. Generalization risk: wildfire-centric utility may not transfer directly to floods, ships, agriculture, or urban monitoring without mission-specific validation.

## Mitigation Strategies
The next study should expand to 500+ Sentinel-2 scenes, preserve geospatial metadata, align FIRMS and burned-area masks, and validate semantic utility against downstream fire mapping or emergency-response tasks.

## Future Work Roadmap
Near-term: 500+ Sentinel-2 scenes, FIRMS validation, burned-area segmentation.

Mid-term: Landsat, MODIS, and additional disaster-response tasks such as flooding and smoke plume monitoring.

Long-term: CubeSat deployment, onboard inference, hardware-in-the-loop testing, and validation over real satellite communication systems.
"""
    (ROOT / "limitations_and_future_work.md").write_text(text, encoding="utf-8")
    return text


def markdown_table(rows: list[dict[str, str]], keys: list[str]) -> str:
    lines = ["| " + " | ".join(keys) + " |", "| " + " | ".join(["---"] * len(keys)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(key, "")) for key in keys) + " |")
    return "\n".join(lines)


def make_pdf(path: Path, title: str, body: str, figures: list[Path]) -> None:
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    with PdfPages(path) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.07, 0.94, title, fontsize=16, weight="bold")
        y = 0.90
        for paragraph in body.splitlines():
            fontsize = 10 if paragraph.startswith("#") else 8.4
            weight = "bold" if paragraph.startswith("#") else "normal"
            text = paragraph.replace("#", "").strip() if paragraph.startswith("#") else paragraph
            for line in textwrap.wrap(text, width=98) or [""]:
                fig.text(0.07, y, line, fontsize=fontsize, weight=weight)
                y -= 0.020
                if y < 0.07:
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)
                    fig = plt.figure(figsize=(8.27, 11.69))
                    y = 0.94
            if paragraph == "":
                y -= 0.010
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        for figure in figures:
            if figure.suffix.lower() != ".png":
                continue
            fig, ax = plt.subplots(figsize=(8.27, 5.8))
            ax.imshow(mpimg.imread(figure))
            ax.axis("off")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def updated_preprint(rows: list[dict[str, str]], figures: list[Path], sat: str, eo: str, lim: str) -> None:
    cross = read_csv(Path("results/earth_observation_validation_100_patch/cross_dataset_comparison.csv"))
    utility = row_by_method(rows, "vqvae_utility_45")
    jpeg20 = row_by_method(rows, "jpeg_quality_20")
    body = f"""# Semantic Utility-Aware Compression for Wildfire-Centric Earth Observation Systems

## Abstract
This paper evaluates semantic utility-aware token prioritization for wildfire-centric Earth Observation compression under severe communication constraints. The evaluation uses DFire (n=50), FLAME repository imagery (n=18), and a 100-scene Sentinel-2/CEMS fixed-patch benchmark. The proposed utility-aware operating point is compared with JPEG quality levels, full VQ-VAE reconstruction, random token selection, and entropy-based token selection using SUS, detector retention, PSNR, SSIM, LPIPS, compression ratio, bandwidth savings, bootstrap confidence intervals, paired t-tests, Wilcoxon tests, and Cohen's d. JPEG achieves higher image-fidelity and detector-retention scores, while utility-aware token selection achieves substantially higher compression and bandwidth savings. On Sentinel-2, utility-aware selection reaches SUS={f(utility, 'semantic_utility_score_mean'):.2f}, detector retention={f(utility, 'detector_retention_mean'):.3f}, compression ratio={f(utility, 'compression_ratio_mean'):.2f}x, and bandwidth savings={f(utility, 'bandwidth_saved_percent_mean'):.2f}%. The results position semantic compression as a bandwidth-constrained operating mode rather than a replacement for conventional image codecs.

## Why Not JPEG?
JPEG appears to outperform utility-aware compression in SUS and detector retention on the Sentinel-2 benchmark. This is expected and should not be hidden: JPEG is a mature distortion-oriented image codec and is highly effective when the goal is preserving visual fidelity. In the current results, JPEG20 reaches SUS={f(jpeg20, 'semantic_utility_score_mean'):.2f} and detector retention={f(jpeg20, 'detector_retention_mean'):.3f}, compared with SUS={f(utility, 'semantic_utility_score_mean'):.2f} and detector retention={f(utility, 'detector_retention_mean'):.3f} for utility-aware token selection.

The scientific question is different. The objective is not to beat JPEG on fidelity at moderate compression; it is to preserve mission utility under severe bandwidth constraints. Utility-aware compression reaches approximately {f(utility, 'compression_ratio_mean'):.2f}x compression and {f(utility, 'bandwidth_saved_percent_mean'):.2f}% bandwidth savings, compared with {f(jpeg20, 'compression_ratio_mean'):.2f}x compression and {f(jpeg20, 'bandwidth_saved_percent_mean'):.2f}% bandwidth savings for JPEG20. For satellite systems, especially CubeSats and intermittent downlinks, communication efficiency can dominate visual fidelity. The present result should therefore be interpreted as evidence for a high-compression emergency or triage mode, not as evidence that semantic token compression universally replaces JPEG.

## Cross-Dataset Results
{markdown_table(cross, ['dataset', 'n_images', 'sus_mean', 'detector_retention_mean', 'bandwidth_saved_percent_mean', 'compression_ratio_mean'])}

## Sentinel-2 Results
{markdown_table(rows, ['method', 'n_images', 'semantic_utility_score_mean', 'detector_retention_mean', 'bandwidth_saved_percent_mean', 'compression_ratio_mean', 'psnr_mean', 'ssim_mean', 'lpips_mean'])}

{sat}

{eo}

{lim}

## Conclusion
The 100-scene Sentinel-2 validation strengthens the Earth Observation relevance of CompressAI by showing that utility-aware token prioritization can preserve measurable wildfire utility while achieving extreme communication reduction. JPEG remains stronger for image fidelity and detector retention, and this is an important baseline result. Utility-aware compression is best positioned as an emerging semantic communication mode for low-bandwidth, onboard-AI, or rapid-disaster-response settings where the central constraint is mission utility per transmitted bit. Future work should extend the evaluation to 500+ georeferenced Sentinel-2 scenes with FIRMS and burned-area validation.
"""
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "updated_preprint.md").write_text(body, encoding="utf-8")
    make_pdf(ROOT / "updated_preprint.pdf", "Updated Preprint", body, figures)


def supervisor_ready(rows: list[dict[str, str]], figures: list[Path]) -> None:
    utility = row_by_method(rows, "vqvae_utility_45")
    body = f"""# Executive Summary
CompressAI now has a publication-oriented Earth Observation validation package using DFire, FLAME, and a 100-scene Sentinel-2/CEMS benchmark. The strongest result is not that the method beats JPEG on visual fidelity; it does not. The strongest result is that utility-aware token selection achieves very high communication efficiency while retaining measurable wildfire utility.

# Most Important Findings
- Sentinel-2 utility-aware SUS: {f(utility, 'semantic_utility_score_mean'):.2f}.
- Detector retention: {f(utility, 'detector_retention_mean'):.3f}.
- Compression ratio: {f(utility, 'compression_ratio_mean'):.2f}x.
- Bandwidth saved: {f(utility, 'bandwidth_saved_percent_mean'):.2f}%.
- Utility-aware selection improves SUS over random and entropy token selection at nearly identical bandwidth savings.

# Research Contributions
1. A reproducible wildfire-centric semantic compression evaluation pipeline.
2. A formal mission-utility evaluation across image and Earth Observation datasets.
3. A 100-scene Sentinel-2/CEMS benchmark with statistical analysis.
4. A clear communication-efficiency argument for satellite and edge-AI systems.

# PhD Extension Opportunities
Near-term: scale to 500+ Sentinel-2 scenes, integrate FIRMS labels, and validate against burned-area segmentation. Mid-term: extend to Landsat, MODIS, and other disasters. Long-term: deploy onboard inference and test over real satellite communication links.

# Potential Publication Venues
IEEE IGARSS, IEEE TGRS letters/short papers after expanded geospatial validation, IEEE JSTARS, Remote Sensing, ESA Phi-week/EO workshops, TU Delft/SnT space AI workshops, and arXiv as an early preprint.
"""
    make_pdf(ROOT / "supervisor_ready_preprint.pdf", "Supervisor-Ready Preprint Package", body, figures)


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    rows = sentinel_rows()
    figures = generate_figures(rows)
    sat = satellite_implications(rows)
    eo = earth_observation_discussion()
    lim = limitations_future_work()
    updated_preprint(rows, figures, sat, eo, lim)
    supervisor_ready(rows, figures)


if __name__ == "__main__":
    main()
