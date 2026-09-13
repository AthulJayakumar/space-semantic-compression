"""evaluation.sentinel2_validation

Plain-English purpose: Benchmark pipelines, statistics, and publication-oriented experiments.

This file is part of the public Space Semantic Compression research demo.
It is documented at module level so researchers, supervisors, and non-specialist
readers can understand where the file fits before reading implementation details.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from communication.satellite_analysis import SatelliteCommunicationAnalyzer
from datasets.sentinel2 import Sentinel2Dataset
from datasets.wildfire_datasets import DatasetItem
from evaluation.wildfire_validation import WildfireValidationPipeline


class Sentinel2ValidationPipeline:
    """Earth Observation satellite validation wrapper for Sentinel-2 imagery."""

    def __init__(self, compression_service, output_root: Path = Path("results/earth_observation_validation")) -> None:
        self.compression_service = compression_service
        self.output_root = output_root
        self.output_root.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        sentinel2_root: Path = Path("datasets/sentinel2"),
        wildfire_results_root: Path = Path("results/wildfire_publication_evidence"),
        limit: int | None = None,
    ) -> dict[str, str]:
        sentinel_dataset = Sentinel2Dataset(sentinel2_root)
        items = list(sentinel_dataset)
        if limit is not None:
            items = items[:limit]
        if not items:
            raise ValueError(f"No Sentinel-2 imagery found in {sentinel2_root / 'images'}")

        adapted = [
            DatasetItem(
                image_path=item.rgb_path,
                label_path=item.label_mask_path,
                metadata={"source": "Sentinel-2", "metadata_path": str(item.metadata_path or "")},
                split="benchmark",
            )
            for item in items
        ]
        validation = WildfireValidationPipeline(self.compression_service, self.output_root / "sentinel2")
        outputs = validation.run({"sentinel2": adapted})

        cross_dataset = self.cross_dataset_comparison(wildfire_results_root, self.output_root / "sentinel2")
        communication = self.communication_analysis(self.output_root / "sentinel2")
        figures = self.generate_earth_observation_figures(self.output_root / "sentinel2", cross_dataset["rows"], items)
        report = self.write_report(items, cross_dataset["path"], communication["communication_csv"], figures)
        pdfs = self.generate_pdf_outputs(report, figures, cross_dataset["path"], communication["communication_csv"])
        return {
            **outputs,
            "sentinel2_results_md": str(report),
            "cross_dataset_comparison_csv": cross_dataset["path"],
            **communication,
            **pdfs,
        }

    def cross_dataset_comparison(self, wildfire_results_root: Path, sentinel_results_root: Path) -> dict[str, object]:
        rows: list[dict[str, object]] = []
        for path in [
            wildfire_results_root / "baseline_summary.csv",
            sentinel_results_root / "baseline_summary.csv",
        ]:
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    if row.get("method") != "vqvae_utility_45":
                        continue
                    rows.append(
                        {
                            "dataset": row.get("dataset"),
                            "method": row.get("method"),
                            "n_images": row.get("n_images"),
                            "sus_mean": row.get("semantic_utility_score_mean"),
                            "sus_ci_low": row.get("semantic_utility_score_ci_low"),
                            "sus_ci_high": row.get("semantic_utility_score_ci_high"),
                            "detector_retention_mean": row.get("detector_retention_mean"),
                            "bandwidth_saved_percent_mean": row.get("bandwidth_saved_percent_mean"),
                            "compression_ratio_mean": row.get("compression_ratio_mean"),
                        }
                    )
        path = self.output_root / "cross_dataset_comparison.csv"
        self._write_csv(path, rows)
        return {"path": str(path), "rows": rows}

    def communication_analysis(self, sentinel_results_root: Path) -> dict[str, str]:
        baseline = sentinel_results_root / "baseline_summary.csv"
        rows: list[dict[str, object]] = []
        if baseline.exists():
            with baseline.open("r", encoding="utf-8", newline="") as handle:
                rows = [row for row in csv.DictReader(handle) if row.get("method") == "vqvae_utility_45"]
        return SatelliteCommunicationAnalyzer(downlink_mbps=50.0, daily_images=1000).export(rows, self.output_root)

    def generate_earth_observation_figures(
        self,
        sentinel_results_root: Path,
        comparison_rows: list[dict[str, object]],
        items,
    ) -> list[str]:
        figure_dir = self.output_root / "figures"
        figure_dir.mkdir(parents=True, exist_ok=True)
        paths: list[str] = []
        try:
            import matplotlib.pyplot as plt
            import matplotlib.image as mpimg
        except Exception:
            return paths

        source_image = items[0].rgb_path if items else None
        if source_image and source_image.exists():
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(mpimg.imread(source_image))
            ax.axis("off")
            path = figure_dir / "figure_1_sentinel2_wildfire_sample.png"
            fig.savefig(path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            paths.append(str(path))

        for pattern, name in [
            ("semantic_heatmap_*.png", "figure_2_utility_heatmap.png"),
            ("token_mask_*.png", "figure_3_token_importance_map.png"),
        ]:
            source = self._latest_file(Path("outputs"), pattern)
            if source:
                image = mpimg.imread(source)
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.imshow(image)
                ax.axis("off")
                path = figure_dir / name
                fig.savefig(path, dpi=300, bbox_inches="tight")
                plt.close(fig)
                paths.append(str(path))

        retention_path = sentinel_results_root / "retention_summary.csv"
        retention_rows: list[dict[str, str]] = []
        if retention_path.exists():
            with retention_path.open("r", encoding="utf-8", newline="") as handle:
                retention_rows = list(csv.DictReader(handle))

        if retention_rows:
            x = [float(row["keep_ratio"]) for row in retention_rows]
            for metric, ylabel, name in [
                ("bandwidth_saved_percent_mean", "Bandwidth saved (%)", "figure_4_bandwidth_vs_sus.png"),
                ("detector_retention_mean", "Detector retention", "figure_5_detector_retention_vs_retention.png"),
            ]:
                y = [float(row.get(metric) or 0.0) for row in retention_rows]
                fig, ax = plt.subplots(figsize=(6.8, 4.2))
                ax.plot(x, y, marker="o", linewidth=2)
                ax.set_xlabel("Token retention ratio")
                ax.set_ylabel(ylabel)
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                path = figure_dir / name
                fig.savefig(path, dpi=300)
                fig.savefig(path.with_suffix(".pdf"))
                plt.close(fig)
                paths.append(str(path))

        if comparison_rows:
            labels = [str(row["dataset"]) for row in comparison_rows]
            sus = [float(row.get("sus_mean") or 0.0) for row in comparison_rows]
            detector = [float(row.get("detector_retention_mean") or 0.0) * 100.0 for row in comparison_rows]
            fig, ax = plt.subplots(figsize=(7.2, 4.2))
            positions = range(len(labels))
            ax.bar([p - 0.18 for p in positions], sus, width=0.36, label="SUS")
            ax.bar([p + 0.18 for p in positions], detector, width=0.36, label="Detector retention (%)")
            ax.set_xticks(list(positions))
            ax.set_xticklabels(labels)
            ax.set_ylabel("Score")
            ax.grid(axis="y", alpha=0.3)
            ax.legend()
            fig.tight_layout()
            path = figure_dir / "figure_6_cross_dataset_comparison.png"
            fig.savefig(path, dpi=300)
            fig.savefig(path.with_suffix(".pdf"))
            plt.close(fig)
            paths.append(str(path))
        return paths

    def _latest_file(self, directory: Path, pattern: str) -> Path | None:
        files = sorted(directory.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
        return files[0] if files else None

    def write_report(
        self,
        items,
        comparison_csv: str,
        communication_csv: str,
        figures: list[str],
    ) -> Path:
        reports_dir = Path("reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        path = reports_dir / "sentinel2_results.md"
        text = [
            "# Sentinel-2 Earth Observation Satellite Validation",
            "",
            "## Methodology",
            "Sentinel-2 RGB composites or GeoTIFF exports are evaluated with the existing semantic token-retention benchmark. The validation uses the wildfire utility detector and the same metrics as DFire/FLAME: SUS, detector retention, PSNR, SSIM, LPIPS, compression ratio, and bandwidth saved.",
            "",
            "## Dataset Description",
            f"Sentinel-2 scenes evaluated: {len(items)}. Inputs are discovered from `datasets/sentinel2/images`, with optional metadata and FIRMS-derived masks under `datasets/sentinel2/metadata` and `datasets/sentinel2/labels`.",
            "",
            "## Utility Evaluation",
            "The detector combines RGB fire/smoke cues with optional Sentinel-2 NIR/SWIR burn-scar and thermal-proxy indices when bands are available.",
            "",
            "## Communication Savings",
            f"Satellite communication estimates are exported to `{communication_csv}` using image-level compression outputs and a configurable downlink model.",
            "",
            "## Cross-Dataset Comparison",
            f"Cross-dataset comparison is exported to `{comparison_csv}` and compares DFire, FLAME, and Sentinel-2 under the same utility-aware VQ-VAE selection condition.",
            "",
            "## Figures",
            *[f"- `{figure}`" for figure in figures],
            "",
            "## Limitations",
            "- Sentinel-2 validation depends on available local Sentinel Hub/Copernicus/CEMS exports.",
            "- FIRMS matching requires geospatial metadata or bounding boxes for each Sentinel-2 image.",
            "- RGB-only Sentinel-2 scenes do not expose the full value of SWIR/NIR burn-scar indices.",
            "",
            "## Future Work",
            "- Expand to a larger CEMS/Sentinel-2 wildfire benchmark.",
            "- Add authenticated Copernicus/Sentinel Hub acquisition for reproducible full-scene experiments.",
            "- Validate utility retention against georeferenced burned-area masks.",
        ]
        path.write_text("\n".join(text), encoding="utf-8")
        return path

    def generate_pdf_outputs(
        self,
        report_path: Path,
        figures: list[str],
        comparison_csv: str,
        communication_csv: str,
    ) -> dict[str, str]:
        pdfs = {
            "sentinel2_validation_report_pdf": self.output_root / "sentinel2_validation_report.pdf",
            "earth_observation_results_pdf": self.output_root / "earth_observation_results.pdf",
            "updated_supervisor_package_pdf": self.output_root / "updated_supervisor_package.pdf",
            "updated_preprint_pdf": self.output_root / "updated_preprint.pdf",
        }
        body = report_path.read_text(encoding="utf-8")
        for title, path in [
            ("Sentinel-2 Validation Report", pdfs["sentinel2_validation_report_pdf"]),
            ("Earth Observation Results", pdfs["earth_observation_results_pdf"]),
            ("Updated Supervisor Package: Satellite Earth Observation Validation", pdfs["updated_supervisor_package_pdf"]),
            ("CompressAI Preprint Update: Earth Observation Satellite Validation", pdfs["updated_preprint_pdf"]),
        ]:
            self._make_pdf(path, title, body, figures, comparison_csv, communication_csv)
        return {key: str(value) for key, value in pdfs.items()}

    def _make_pdf(self, path: Path, title: str, body: str, figures: list[str], comparison_csv: str, communication_csv: str) -> None:
        from matplotlib.backends.backend_pdf import PdfPages
        import matplotlib.image as mpimg
        import matplotlib.pyplot as plt
        import textwrap

        path.parent.mkdir(parents=True, exist_ok=True)
        with PdfPages(path) as pdf:
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.text(0.08, 0.94, title, fontsize=16, weight="bold")
            y = 0.89
            extra = body + f"\n\nArtifacts:\n- {comparison_csv}\n- {communication_csv}"
            for paragraph in extra.splitlines():
                for line in textwrap.wrap(paragraph, width=94) or [""]:
                    fig.text(0.08, y, line, fontsize=8.5)
                    y -= 0.020
                    if y < 0.08:
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

    def _write_csv(self, path: Path, rows: list[dict[str, object]]) -> None:
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        keys = sorted({key for row in rows for key in row.keys()})
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
