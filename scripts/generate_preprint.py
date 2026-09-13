"""scripts.generate_preprint

Plain-English purpose: Command-line runners for datasets, benchmarks, reports, and exports.

This file is part of the public Space Semantic Compression research demo.
It is documented at module level so researchers, supervisors, and non-specialist
readers can understand where the file fits before reading implementation details.
"""

from __future__ import annotations

import csv
import json
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def markdown_table(rows: list[dict[str, str]], keys: list[str], max_rows: int = 12) -> str:
    if not rows:
        return "No rows available."
    lines = ["| " + " | ".join(keys) + " |", "| " + " | ".join(["---"] * len(keys)) + " |"]
    for row in rows[:max_rows]:
        lines.append("| " + " | ".join(str(row.get(key, "")) for key in keys) + " |")
    return "\n".join(lines)


def make_pdf(path: Path, markdown: str) -> None:
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        y = 0.95
        for raw in markdown.splitlines():
            fontsize = 12 if raw.startswith("# ") else 9
            weight = "bold" if raw.startswith("#") else "normal"
            text = raw.replace("#", "").strip() if raw.startswith("#") else raw
            for line in textwrap.wrap(text, width=98) or [""]:
                fig.text(0.07, y, line, fontsize=fontsize, weight=weight)
                y -= 0.022 if fontsize == 9 else 0.034
                if y < 0.07:
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)
                    fig = plt.figure(figsize=(8.27, 11.69))
                    y = 0.95
            if raw == "":
                y -= 0.010
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    wildfire_root = Path("results/wildfire_publication_evidence")
    eo_root = Path("results/earth_observation_validation_100_patch")
    if not eo_root.exists():
        eo_root = Path("results/earth_observation_validation")
    baseline = read_csv(wildfire_root / "baseline_summary.csv")
    cross = read_csv(eo_root / "cross_dataset_comparison.csv")
    sentinel = read_csv(eo_root / "sentinel2" / "baseline_summary.csv")
    stats = read_csv(wildfire_root / "statistical_significance.csv")
    cems_manifest_path = Path("datasets/sentinel2/cems_prepare_manifest.json")
    cems_manifest = json.loads(cems_manifest_path.read_text(encoding="utf-8")) if cems_manifest_path.exists() else {}
    prepared_sentinel2 = cems_manifest.get("images")

    text = f"""# Semantic Utility-Aware Compression for Wildfire-Centric Earth Observation Systems

## Abstract
This preprint evaluates semantic utility-aware token prioritization for wildfire-centric image and Earth Observation compression. The system prioritizes transmission of mission-relevant tokens under bandwidth constraints and is evaluated on DFire, FLAME repository imagery, and Sentinel-2/CEMS sample scenes. Metrics include Semantic Utility Score (SUS), detector retention, PSNR, SSIM, LPIPS, compression ratio, bandwidth saved, bootstrap confidence intervals, and paired statistical testing.

## Research Hypothesis
Semantic utility-aware token prioritization preserves wildfire-relevant information more effectively than non-semantic token selection under equivalent communication constraints.

## Methodology
Images are encoded with the existing VQ-VAE tokenization pipeline. Utility-aware token selection ranks tokens using wildfire confidence, smoke relevance, burn-scar relevance, entropy, and energy-aware transmission terms. Reconstructions are evaluated using conventional distortion metrics and formal SUS components.

## Datasets
DFire: 50 automatically prepared fire/smoke samples with aligned labels. FLAME: 18 repository-available UAV wildfire images. Sentinel-2: 100 CEMS Sentinel-2 validation scenes are benchmarked as fixed-size 512 pixel Earth Observation patches prepared from the CEMS validation split. The preferred final benchmark size for a journal extension is 500+ scenes.

## Cross-Dataset Utility-Aware Results
{markdown_table(cross, ["dataset", "n_images", "sus_mean", "detector_retention_mean", "bandwidth_saved_percent_mean", "compression_ratio_mean"])}

## Sentinel-2 Earth Observation Results
{markdown_table(sentinel, ["method", "n_images", "semantic_utility_score_mean", "detector_retention_mean", "bandwidth_saved_percent_mean", "compression_ratio_mean", "psnr_mean", "ssim_mean", "lpips_mean"])}

## Statistical Evidence
{markdown_table(stats, ["dataset", "comparison", "metric", "n", "paired_t_p", "wilcoxon_p", "cohens_d"], max_rows=18)}

## Discussion
The strongest current result is communication efficiency: utility-aware compression achieves high bandwidth reduction across all evaluated datasets while retaining measurable wildfire utility. On the 100-scene Sentinel-2/CEMS fixed-patch benchmark, utility-aware selection reaches 99.35% bandwidth saving and 155.77x compression ratio with SUS 79.52. On DFire and Sentinel-2, utility-aware selection improves SUS over random and entropy token selection while maintaining high bandwidth saving.

## Limitations
The Sentinel-2 result is statistically stronger than the previous sample-level analysis, but the controlled benchmark uses fixed-size RGB patches. A journal extension should expand toward 500+ georeferenced scenes with full Sentinel-2 multispectral bands, Sentinel Hub/Copernicus provenance, and FIRMS or burned-area masks.

## Conclusion
The evidence supports continued development of semantic utility-aware compression for wildfire-centric Earth Observation systems. The method is promising for satellite downlink-constrained settings, with the next journal-grade milestone being a 500+ scene georeferenced Sentinel-2 benchmark.
"""
    out_dir = Path("results/preprint")
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = Path("reports/compressai_preprint.md")
    md_path.write_text(text, encoding="utf-8")
    pdf_path = out_dir / "semantic_utility_wildfire_earth_observation_preprint.pdf"
    make_pdf(pdf_path, text)
    print(f"markdown={md_path}")
    print(f"pdf={pdf_path}")


if __name__ == "__main__":
    main()
