"""evaluation.wildfire_validation

Plain-English purpose: Benchmark pipelines, statistics, and publication-oriented experiments.

This file is part of the public Space Semantic Compression research demo.
It is documented at module level so researchers, supervisors, and non-specialist
readers can understand where the file fits before reading implementation details.
"""

from __future__ import annotations

import csv
import json
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

from backend.services.compression_service import CompressionService
from datasets.wildfire_datasets import DatasetItem
from evaluation.publication_experiments import PublicationExperimentRunner


class WildfireValidationPipeline:
    """Dataset-level validation pipeline for wildfire semantic compression evidence."""

    metrics = [
        "psnr",
        "ssim",
        "lpips",
        "semantic_utility_score",
        "detector_retention",
        "compression_ratio",
        "bandwidth_saved_percent",
    ]

    def __init__(self, compression_service: CompressionService, output_root: Path = Path("results/wildfire_validation")) -> None:
        self.service = compression_service
        self.output_root = output_root
        self.output_root.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        datasets: dict[str, Iterable[DatasetItem]],
        mission: str = "wildfire_detection",
        limit: int | None = None,
    ) -> dict[str, str]:
        manifest: list[dict[str, object]] = []
        all_retention: list[dict[str, object]] = []
        all_baseline: list[dict[str, object]] = []
        all_stats: list[dict[str, object]] = []
        all_figures: list[str] = []

        for dataset_name, dataset in datasets.items():
            items = list(dataset)
            if limit is not None:
                items = items[:limit]
            image_paths = [item.image_path for item in items]
            manifest.append(
                {
                    "dataset": dataset_name,
                    "images": len(image_paths),
                    "labels_aligned": sum(1 for item in items if item.label_path is not None),
                    "first_image": str(image_paths[0]) if image_paths else None,
                }
            )
            if not image_paths:
                continue

            runner = PublicationExperimentRunner(self.service, self.output_root / dataset_name)
            retention = runner.retention_study(image_paths, mission)
            baseline = runner.baseline_comparison(image_paths, mission)
            ablation = runner.ablation_study(image_paths, mission)
            stats = runner.statistical_analysis(baseline)
            figures = runner.generate_figures(retention, baseline)
            runner.generate_report(retention, baseline, ablation, stats, figures)

            for row in retention:
                row["dataset"] = dataset_name
            for row in baseline:
                row["dataset"] = dataset_name
            for row in stats:
                row["dataset"] = dataset_name
            all_retention.extend(retention)
            all_baseline.extend(baseline)
            all_stats.extend(stats)
            all_figures.extend(figures)

        retention_summary = self.aggregate(all_retention, ["dataset", "keep_ratio"], self.metrics)
        baseline_summary = self.aggregate(all_baseline, ["dataset", "method"], self.metrics)
        self._write_csv(self.output_root / "retention_summary.csv", retention_summary)
        self._write_csv(self.output_root / "baseline_summary.csv", baseline_summary)
        self._write_csv(self.output_root / "statistical_significance.csv", all_stats)
        (self.output_root / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        validation_figures = self.generate_validation_figures(retention_summary, baseline_summary)
        pdfs = self.generate_pdf_package(manifest, retention_summary, baseline_summary, all_stats, validation_figures + all_figures)
        return {
            "manifest": str(self.output_root / "dataset_manifest.json"),
            "retention_summary": str(self.output_root / "retention_summary.csv"),
            "baseline_summary": str(self.output_root / "baseline_summary.csv"),
            "statistical_significance": str(self.output_root / "statistical_significance.csv"),
            **pdfs,
        }

    def aggregate(
        self,
        rows: list[dict[str, object]],
        group_keys: list[str],
        metrics: list[str],
    ) -> list[dict[str, object]]:
        grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
        for row in rows:
            grouped[tuple(row.get(key) for key in group_keys)].append(row)

        summary: list[dict[str, object]] = []
        for group, group_rows in grouped.items():
            out = {key: value for key, value in zip(group_keys, group)}
            out["n_images"] = len({row.get("image") for row in group_rows})
            for metric in metrics:
                values = [float(row[metric]) for row in group_rows if row.get(metric) is not None]
                if not values:
                    continue
                mean, low, high = self.bootstrap_mean_ci(values)
                out[f"{metric}_mean"] = mean
                out[f"{metric}_ci_low"] = low
                out[f"{metric}_ci_high"] = high
            summary.append(out)
        return sorted(summary, key=lambda row: tuple(str(row.get(key, "")) for key in group_keys))

    def bootstrap_mean_ci(self, values: list[float], n_bootstrap: int = 1000) -> tuple[float, float, float]:
        arr = np.asarray(values, dtype="float64")
        if arr.size == 0:
            return float("nan"), float("nan"), float("nan")
        rng = np.random.default_rng(2026)
        means = [float(rng.choice(arr, size=arr.size, replace=True).mean()) for _ in range(n_bootstrap)]
        return (
            round(float(arr.mean()), 6),
            round(float(np.quantile(means, 0.025)), 6),
            round(float(np.quantile(means, 0.975)), 6),
        )

    def generate_validation_figures(
        self,
        retention_summary: list[dict[str, object]],
        baseline_summary: list[dict[str, object]],
    ) -> list[str]:
        figure_dir = self.output_root / "figures"
        figure_dir.mkdir(parents=True, exist_ok=True)
        paths: list[str] = []
        try:
            import matplotlib.pyplot as plt
        except Exception:
            return paths

        for metric, ylabel, filename in [
            ("semantic_utility_score", "SUS", "retention_vs_sus.png"),
            ("bandwidth_saved_percent", "Bandwidth saved (%)", "retention_vs_bandwidth.png"),
            ("psnr", "PSNR (dB)", "retention_vs_psnr.png"),
            ("ssim", "SSIM", "retention_vs_ssim.png"),
        ]:
            fig, ax = plt.subplots(figsize=(6.8, 4.2))
            by_dataset: dict[str, list[dict[str, object]]] = defaultdict(list)
            for row in retention_summary:
                by_dataset[str(row.get("dataset"))].append(row)
            for dataset_name, rows in by_dataset.items():
                rows = sorted(rows, key=lambda row: float(row.get("keep_ratio") or 0))
                x = [float(row["keep_ratio"]) for row in rows]
                y = [float(row.get(f"{metric}_mean", 0.0)) for row in rows]
                low = [float(row.get(f"{metric}_ci_low", mean)) for row, mean in zip(rows, y)]
                high = [float(row.get(f"{metric}_ci_high", mean)) for row, mean in zip(rows, y)]
                yerr = [[mean - lo for mean, lo in zip(y, low)], [hi - mean for mean, hi in zip(y, high)]]
                ax.errorbar(x, y, yerr=yerr, marker="o", linewidth=2, capsize=3, label=dataset_name)
            ax.set_xlabel("Token retention ratio")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            path = figure_dir / filename
            fig.savefig(path, dpi=300)
            fig.savefig(path.with_suffix(".pdf"))
            plt.close(fig)
            paths.append(str(path))

        sus_rows = [row for row in baseline_summary if row.get("semantic_utility_score_mean") is not None]
        if sus_rows:
            labels = [str(row.get("method")) for row in sus_rows]
            values = [float(row["semantic_utility_score_mean"]) for row in sus_rows]
            fig, ax = plt.subplots(figsize=(8.0, 4.4))
            positions = np.arange(len(labels))
            ax.bar(positions, values)
            ax.set_ylabel("SUS")
            ax.set_xticks(positions)
            ax.set_xticklabels(labels, rotation=35, ha="right")
            ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()
            path = figure_dir / "baseline_sus_comparison.png"
            fig.savefig(path, dpi=300)
            fig.savefig(path.with_suffix(".pdf"))
            plt.close(fig)
            paths.append(str(path))
        return paths

    def generate_pdf_package(
        self,
        manifest: list[dict[str, object]],
        retention_summary: list[dict[str, object]],
        baseline_summary: list[dict[str, object]],
        stats: list[dict[str, object]],
        figures: list[str],
    ) -> dict[str, str]:
        pdfs = {
            "wildfire_results_pdf": self.output_root / "wildfire_results.pdf",
            "publication_ready_report_pdf": self.output_root / "publication_ready_report.pdf",
            "supervisor_outreach_package_pdf": self.output_root / "supervisor_outreach_package.pdf",
        }
        self._make_pdf(
            pdfs["wildfire_results_pdf"],
            "CompressAI Wildfire Results",
            [
                "Dataset-level validation outputs generated by the automated wildfire benchmark pipeline.",
                self._manifest_text(manifest),
                "Retention Summary",
                self._rows_text(retention_summary[:12]),
                "Baseline Summary",
                self._rows_text(baseline_summary[:12]),
                "Key Findings for PhD Applications",
                self._key_findings(retention_summary, baseline_summary, stats),
            ],
            figures[:4],
        )
        self._make_pdf(
            pdfs["publication_ready_report_pdf"],
            "CompressAI Publication-Ready Report",
            [
                "Metrics: PSNR, SSIM, LPIPS, SUS, detector retention, compression ratio, and bandwidth saved.",
                "Key Findings for PhD Applications",
                self._key_findings(retention_summary, baseline_summary, stats),
                "Statistical Significance",
                self._rows_text(stats[:16]),
            ],
            figures,
        )
        self._make_pdf(
            pdfs["supervisor_outreach_package_pdf"],
            "CompressAI Supervisor Outreach Package",
            [
                "Hypothesis: semantic utility-aware token prioritization preserves wildfire-relevant information under bandwidth constraints.",
                self._manifest_text(manifest),
                "Primary evidence consists of retention sweeps, baseline comparison, bootstrap confidence intervals, and paired statistical tests.",
                "Interpretation must respect dataset size and label availability recorded in the manifest.",
                "Key Findings for PhD Applications",
                self._key_findings(retention_summary, baseline_summary, stats),
            ],
            figures[:6],
        )
        return {key: str(value) for key, value in pdfs.items()}

    def _make_pdf(self, path: Path, title: str, sections: list[str], figures: list[str]) -> None:
        from matplotlib.backends.backend_pdf import PdfPages
        import matplotlib.pyplot as plt

        path.parent.mkdir(parents=True, exist_ok=True)
        with PdfPages(path) as pdf:
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.text(0.08, 0.94, title, fontsize=18, weight="bold")
            y = 0.88
            for section in sections:
                for line in textwrap.wrap(section, width=95) or [""]:
                    fig.text(0.08, y, line, fontsize=9)
                    y -= 0.022
                    if y < 0.08:
                        pdf.savefig(fig, bbox_inches="tight")
                        plt.close(fig)
                        fig = plt.figure(figsize=(8.27, 11.69))
                        y = 0.94
                y -= 0.012
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            for figure in figures:
                source = Path(figure)
                if not source.exists() or source.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                    continue
                try:
                    import matplotlib.image as mpimg

                    image = mpimg.imread(source)
                    fig, ax = plt.subplots(figsize=(8.27, 5.6))
                    ax.imshow(image)
                    ax.axis("off")
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)
                except Exception:
                    continue

    def _manifest_text(self, manifest: list[dict[str, object]]) -> str:
        if not manifest:
            return "No datasets were discovered."
        return "Datasets: " + "; ".join(
            f"{row['dataset']} n={row['images']} labels={row['labels_aligned']}" for row in manifest
        )

    def _rows_text(self, rows: list[dict[str, object]]) -> str:
        if not rows:
            return "No rows generated."
        keys = list(rows[0].keys())[:8]
        lines = [", ".join(keys)]
        for row in rows:
            lines.append(", ".join(str(row.get(key, "")) for key in keys))
        return "\n".join(lines)

    def _key_findings(
        self,
        retention_summary: list[dict[str, object]],
        baseline_summary: list[dict[str, object]],
        stats: list[dict[str, object]],
    ) -> str:
        findings: list[str] = []
        utility_rows = [row for row in baseline_summary if row.get("method") == "vqvae_utility_45"]
        for row in utility_rows:
            dataset = row.get("dataset")
            sus = row.get("semantic_utility_score_mean")
            detector = row.get("detector_retention_mean")
            bandwidth = row.get("bandwidth_saved_percent_mean")
            ratio = row.get("compression_ratio_mean")
            findings.append(
                f"{dataset}: utility-aware token selection achieved SUS={sus}, detector retention={detector}, "
                f"bandwidth saved={bandwidth}%, compression ratio={ratio}."
            )
        best_low_rate = [
            row
            for row in retention_summary
            if float(row.get("keep_ratio") or 1.0) <= 0.5 and row.get("semantic_utility_score_mean") is not None
        ]
        if best_low_rate:
            best = max(best_low_rate, key=lambda row: float(row["semantic_utility_score_mean"]))
            findings.append(
                f"Best low-bandwidth retention point: {best.get('dataset')} at keep_ratio={best.get('keep_ratio')} "
                f"with SUS={best.get('semantic_utility_score_mean')} and bandwidth saved={best.get('bandwidth_saved_percent_mean')}%."
            )
        significant = [
            row
            for row in stats
            if row.get("metric") in {"semantic_utility_score", "bandwidth_saved_percent", "detector_retention"}
            and row.get("paired_t_p") is not None
            and float(row.get("paired_t_p") or 1.0) < 0.05
        ]
        findings.append(f"Statistical tests with p<0.05 in the headline metrics: {len(significant)}.")
        if not findings:
            return "No benchmark rows were generated; run the pipeline on DFire and FLAME to populate this section."
        return "\n".join(findings)

    def _write_csv(self, path: Path, rows: list[dict[str, object]]) -> None:
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        keys = sorted({key for row in rows for key in row.keys()})
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
