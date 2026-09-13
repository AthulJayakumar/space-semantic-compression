"""evaluation.publication_experiments

Plain-English purpose: Benchmark pipelines, statistics, and publication-oriented experiments.

This file is part of the public Space Semantic Compression research demo.
It is documented at module level so researchers, supervisors, and non-specialist
readers can understand where the file fits before reading implementation details.
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image

from backend.schemas.response_schema import TransmissionConfig
from backend.services.compression_service import CompressionService
from backend.utils.image_utils import load_image_bytes
from backend.utils.tensor_utils import image_to_tensor, tensor_to_image
from evaluation.statistics import StatisticalValidator
from metrics.semantic_utility import SemanticUtilityMetric
from token_selection.utility_pruner import TokenSelectionWeights, UtilityAwareTokenPruner


class PublicationExperimentRunner:
    def __init__(self, compression_service: CompressionService, output_root: Path = Path("results")) -> None:
        self.service = compression_service
        self.output_root = output_root
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.stats = StatisticalValidator()

    def run_all(self, image_paths: list[Path], mission: str = "wildfire_detection") -> dict[str, str]:
        image_paths = [Path(path) for path in image_paths]
        retention_rows = self.retention_study(image_paths, mission)
        baseline_rows = self.baseline_comparison(image_paths, mission)
        ablation_rows = self.ablation_study(image_paths, mission)
        stats_rows = self.statistical_analysis(baseline_rows)
        figure_paths = self.generate_figures(retention_rows, baseline_rows)
        report_path = self.generate_report(retention_rows, baseline_rows, ablation_rows, stats_rows, figure_paths)
        return {
            "retention_csv": str(self.output_root / "retention_study.csv"),
            "baseline_csv": str(self.output_root / "baseline_comparison.csv"),
            "ablation_csv": str(self.output_root / "ablation_study.csv"),
            "statistics_csv": str(self.output_root / "statistical_analysis.csv"),
            "report": str(report_path),
        }

    def retention_study(self, image_paths: list[Path], mission: str) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for image_path in image_paths:
            payload = image_path.read_bytes()
            for keep_ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                result = self.service.compress_image(
                    payload,
                    image_path.name,
                    TransmissionConfig(semantic_keep_ratio=keep_ratio),
                    mission=mission,
                )
                rows.append(self._row_from_result(image_path, f"utility_aware_{int(keep_ratio * 100)}", keep_ratio, result))
        self._write_csv(self.output_root / "retention_study.csv", rows)
        self._write_json(self.output_root / "retention_study.json", rows)
        return rows

    def baseline_comparison(self, image_paths: list[Path], mission: str) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for image_path in image_paths:
            payload = image_path.read_bytes()
            for quality in [20, 40, 60, 80]:
                rows.append(self._jpeg_baseline(image_path, quality))
            rows.append(self._vqvae_variant(image_path, payload, "vqvae_full", 1.0, "utility", mission))
            rows.append(self._vqvae_variant(image_path, payload, "vqvae_random_45", 0.45, "random", mission))
            rows.append(self._vqvae_variant(image_path, payload, "vqvae_entropy_45", 0.45, "entropy", mission))
            rows.append(self._vqvae_variant(image_path, payload, "vqvae_utility_45", 0.45, "utility", mission))
        self._write_csv(self.output_root / "baseline_comparison.csv", rows)
        self._write_json(self.output_root / "baseline_comparison.json", rows)
        return rows

    def ablation_study(self, image_paths: list[Path], mission: str) -> list[dict[str, object]]:
        variants = {
            "full_system": TokenSelectionWeights(alpha_utility=0.55, beta_entropy=0.20, gamma_cost=0.05, delta_detail=0.20),
            "without_utility_map": TokenSelectionWeights(alpha_utility=0.0, beta_entropy=0.65, gamma_cost=0.10, delta_detail=0.25),
            "without_detector_confidence": TokenSelectionWeights(alpha_utility=0.40, beta_entropy=0.30, gamma_cost=0.10, delta_detail=0.20),
            "without_entropy": TokenSelectionWeights(alpha_utility=0.70, beta_entropy=0.0, gamma_cost=0.10, delta_detail=0.20),
            "without_detail_term": TokenSelectionWeights(alpha_utility=0.65, beta_entropy=0.25, gamma_cost=0.10, delta_detail=0.0),
            "without_energy_term": TokenSelectionWeights(alpha_utility=0.55, beta_entropy=0.25, gamma_cost=0.0, delta_detail=0.20),
            "without_adaptive_transmission": TokenSelectionWeights(alpha_utility=0.55, beta_entropy=0.20, gamma_cost=0.05, delta_detail=0.20),
        }
        original_pruner = self.service.utility_pruner
        rows: list[dict[str, object]] = []
        try:
            for image_path in image_paths:
                payload = image_path.read_bytes()
                for name, weights in variants.items():
                    self.service.utility_pruner = UtilityAwareTokenPruner(weights)
                    keep_ratio = 1.0 if name == "without_adaptive_transmission" else 0.45
                    result = self.service.compress_image(
                        payload,
                        image_path.name,
                        TransmissionConfig(semantic_keep_ratio=keep_ratio, token_selection_mode="custom"),
                        mission=mission,
                    )
                    rows.append(self._row_from_result(image_path, name, keep_ratio, result))
        finally:
            self.service.utility_pruner = original_pruner
        self._write_csv(self.output_root / "ablation_study.csv", rows)
        self._write_json(self.output_root / "ablation_study.json", rows)
        return rows

    def statistical_analysis(self, baseline_rows: list[dict[str, object]]) -> list[dict[str, object]]:
        comparisons = [
            ("jpeg_quality_20", "vqvae_utility_45"),
            ("jpeg_quality_40", "vqvae_utility_45"),
            ("vqvae_full", "vqvae_utility_45"),
            ("vqvae_random_45", "vqvae_utility_45"),
            ("vqvae_entropy_45", "vqvae_utility_45"),
        ]
        rows: list[dict[str, object]] = []
        for baseline, candidate in comparisons:
            for metric in [
                "semantic_utility_score",
                "detector_retention",
                "compression_ratio",
                "bandwidth_saved_percent",
                "psnr",
                "ssim",
                "lpips",
            ]:
                b_values = self._metric_by_image(baseline_rows, baseline, metric)
                c_values = self._metric_by_image(baseline_rows, candidate, metric)
                shared = sorted(set(b_values) & set(c_values))
                if len(shared) < 1:
                    continue
                b = [b_values[key] for key in shared]
                c = [c_values[key] for key in shared]
                ttest = self.stats.paired_t_test(b, c)
                wilcoxon = self.stats.wilcoxon(b, c)
                ci_low, ci_high = self.bootstrap_ci([c_val - b_val for b_val, c_val in zip(b, c)])
                rows.append(
                    {
                        "comparison": f"{baseline}_vs_{candidate}",
                        "metric": metric,
                        "n": len(shared),
                        "paired_t_p": ttest.p_value,
                        "wilcoxon_p": wilcoxon.p_value,
                        "cohens_d": ttest.effect_size,
                        "bootstrap_ci_low": ci_low,
                        "bootstrap_ci_high": ci_high,
                        "note": "inferential p-values require n>=2" if len(shared) < 2 else "",
                    }
                )
        self._write_csv(self.output_root / "statistical_analysis.csv", rows)
        self._write_json(self.output_root / "statistical_analysis.json", rows)
        return rows

    def generate_figures(self, retention_rows: list[dict[str, object]], baseline_rows: list[dict[str, object]]) -> list[str]:
        figures_dir = Path("figures")
        figures_dir.mkdir(parents=True, exist_ok=True)
        paths: list[str] = []
        try:
            import matplotlib.pyplot as plt
        except Exception:
            return paths
        retention = sorted(retention_rows, key=lambda row: float(row["keep_ratio"]))
        if retention:
            x = [float(row["keep_ratio"]) for row in retention]
            for name, y_key, ylabel in [
                ("figure_3_rate_vs_utility.png", "semantic_utility_score", "Semantic Utility Score"),
                ("figure_4_bandwidth_vs_sus.png", "bandwidth_saved_percent", "Bandwidth Saved (%)"),
                ("figure_5_energy_vs_utility.png", "energy_j", "Energy (J)"),
                ("figure_6_retention_sweep_curves.png", "ssim", "SSIM"),
            ]:
                y = [float(row[y_key]) for row in retention if row.get(y_key) is not None]
                plt.figure(figsize=(6.5, 4.0), dpi=220)
                plt.plot(x[: len(y)], y, marker="o", linewidth=2)
                plt.xlabel("Token retention ratio")
                plt.ylabel(ylabel)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                out = figures_dir / name
                plt.savefig(out)
                plt.close()
                paths.append(str(out))
        latest_heatmap = self._latest_file(Path("outputs"), "semantic_heatmap_*.png")
        latest_mask = self._latest_file(Path("outputs"), "token_mask_*.png")
        latest_recon = self._latest_file(Path("outputs"), "reconstruction_*.png")
        for idx, source in [(1, latest_heatmap), (2, latest_mask), (7, latest_recon)]:
            if source:
                target = figures_dir / f"figure_{idx}_{source.name}"
                target.write_bytes(source.read_bytes())
                paths.append(str(target))
        return paths

    def generate_report(
        self,
        retention_rows: list[dict[str, object]],
        baseline_rows: list[dict[str, object]],
        ablation_rows: list[dict[str, object]],
        stats_rows: list[dict[str, object]],
        figure_paths: list[str],
    ) -> Path:
        reports_dir = Path("reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        path = reports_dir / "research_report.md"
        text = [
            "# CompressAI Experimental Research Report",
            "",
            "## Methodology",
            "This report evaluates semantic utility-aware token prioritization for wildfire-centric Earth observation compression.",
            "",
            "## Datasets",
            "Experiments consume local Sentinel-2, MODIS, FIRMS, or benchmark image folders through the dataset pipeline.",
            "",
            "## Metrics",
            "Metrics include PSNR, SSIM, LPIPS, SUS, compression ratio, bandwidth saved, energy estimate, and detector-retention components.",
            "",
            "## Retention Study",
            self._markdown_table(retention_rows[:12]),
            "",
            "## Baseline Comparison",
            self._markdown_table(baseline_rows[:16]),
            "",
            "## Ablation Study",
            self._markdown_table(ablation_rows[:12]),
            "",
            "## Statistical Analysis",
            self._markdown_table(stats_rows[:16]),
            "",
            "## Figures",
            *[f"- `{figure}`" for figure in figure_paths],
            "",
            "## Discussion",
            "Results should be interpreted as experimental evidence generated by the current benchmark inputs. Dataset-level conclusions require larger benchmark sets and statistical validation.",
            "",
            "## Limitations",
            "- Wildfire detector quality depends on available YOLO weights or deterministic fallback utility maps.",
            "- Satellite-link and energy estimates are simplified research models.",
            "- Edge deployment results are hardware-dependent.",
            "",
            "## Future Work",
            "- Expand dataset-level wildfire evaluation.",
            "- Add trained wildfire YOLO weights.",
            "- Validate SUS against downstream wildfire detection accuracy.",
            "- Run Jetson-class deployment benchmarks.",
        ]
        path.write_text("\n".join(text), encoding="utf-8")
        return path

    def _vqvae_variant(self, image_path: Path, payload: bytes, name: str, keep_ratio: float, selection: str, mission: str) -> dict[str, object]:
        if selection == "utility":
            result = self.service.compress_image(payload, image_path.name, TransmissionConfig(semantic_keep_ratio=keep_ratio), mission=mission)
            return self._row_from_result(image_path, name, keep_ratio, result)
        return self._manual_selection_variant(image_path, payload, name, keep_ratio, selection, mission)

    def _manual_selection_variant(self, image_path: Path, payload: bytes, name: str, keep_ratio: float, selection: str, mission: str) -> dict[str, object]:
        original = load_image_bytes(payload)
        tensor = image_to_tensor(original, self.service.encoder_service.device, self.service.encoder_service.stride)
        tokens = self.service.encoder_service.encode(tensor)
        shape = tuple(tokens.shape[-2:])
        detector_before = self.service._detect_mission_utility(original, shape, mission)
        if selection == "random":
            rng = random.Random(1234)
            order = list(range(tokens.numel()))
            rng.shuffle(order)
            keep_count = max(1, int(round(tokens.numel() * keep_ratio)))
            mask = np.zeros(tokens.numel(), dtype=bool)
            mask[order[:keep_count]] = True
            keep_mask = mask.reshape(shape)
        else:
            entropy_only = TokenSelectionWeights(alpha_utility=0.0, beta_entropy=1.0, gamma_cost=0.0, delta_detail=0.0)
            keep_mask, _ = UtilityAwareTokenPruner(entropy_only).select(tokens, np.zeros(shape, dtype="float32"), keep_ratio)
        pruned = self.service.token_service.prune_tokens(tokens, keep_mask)
        reconstruction = tensor_to_image(self.service.decoder_service.decode(pruned))
        detector_after = self.service._detect_mission_utility(reconstruction, shape, mission)
        sus, components = SemanticUtilityMetric().score_before_after(detector_before, detector_after)
        full_payload = self.service.token_service.estimate_payload_kb(tokens)
        compressed_payload = self.service.token_service.estimate_payload_kb(pruned, keep_mask)
        original_kb = len(payload) / 1024.0
        lpips = self.service.metrics_service.lpips(original, reconstruction)
        return {
            "image": str(image_path),
            "method": name,
            "keep_ratio": keep_ratio,
            "psnr": round(self.service.metrics_service.psnr(original, reconstruction), 4),
            "ssim": round(self.service.metrics_service.ssim(original, reconstruction), 4),
            "lpips": round(lpips, 6) if lpips is not None else None,
            "semantic_utility_score": round(sus, 4),
            "compression_ratio": round(original_kb / max(compressed_payload, 1e-9), 4),
            "bandwidth_saved_percent": round(max(0.0, (1.0 - compressed_payload / original_kb) * 100.0), 2),
            "energy_j": None,
            "detector_retention": round(components.detector_retention, 6),
            "object_retention": round(components.object_retention, 6),
            "relevance_retention": round(components.relevance_retention, 6),
            "region_preservation": round(components.region_preservation, 6),
            "full_payload_kb": round(full_payload, 4),
            "compressed_payload_kb": round(compressed_payload, 4),
        }

    def _jpeg_baseline(self, image_path: Path, quality: int) -> dict[str, object]:
        original = Image.open(image_path).convert("RGB")
        import io

        buffer = io.BytesIO()
        original.save(buffer, format="JPEG", quality=quality, optimize=True)
        decoded = load_image_bytes(buffer.getvalue())
        shape = (32, 32)
        before = self.service._detect_mission_utility(original, shape, "wildfire_detection")
        after = self.service._detect_mission_utility(decoded, shape, "wildfire_detection")
        sus, components = SemanticUtilityMetric().score_before_after(before, after)
        original_kb = image_path.stat().st_size / 1024.0
        compressed_kb = len(buffer.getvalue()) / 1024.0
        lpips = self.service.metrics_service.lpips(original, decoded)
        return {
            "image": str(image_path),
            "method": f"jpeg_quality_{quality}",
            "keep_ratio": None,
            "psnr": round(self.service.metrics_service.psnr(original, decoded), 4),
            "ssim": round(self.service.metrics_service.ssim(original, decoded), 4),
            "lpips": round(lpips, 6) if lpips is not None else None,
            "semantic_utility_score": round(sus, 4),
            "compression_ratio": round(original_kb / max(compressed_kb, 1e-9), 4),
            "bandwidth_saved_percent": round(max(0.0, (1.0 - compressed_kb / original_kb) * 100.0), 2),
            "energy_j": None,
            "detector_retention": round(components.detector_retention, 6),
            "object_retention": round(components.object_retention, 6),
            "relevance_retention": round(components.relevance_retention, 6),
            "region_preservation": round(components.region_preservation, 6),
        }

    def _row_from_result(self, image_path: Path, method: str, keep_ratio: float, result) -> dict[str, object]:
        return {
            "image": str(image_path),
            "method": method,
            "keep_ratio": keep_ratio,
            "psnr": result.psnr,
            "ssim": result.ssim,
            "lpips": result.lpips,
            "semantic_utility_score": result.semantic_utility_score,
            "compression_ratio": result.compression_ratio,
            "bandwidth_saved_percent": result.bandwidth_saved_percent,
            "energy_j": result.total_energy_j,
            "detector_retention": result.detector_retention,
            "object_retention": result.object_retention,
            "relevance_retention": result.relevance_retention,
            "region_preservation": result.region_preservation,
        }

    def _metric_by_image(self, rows: list[dict[str, object]], method: str, metric: str) -> dict[str, float]:
        return {str(row["image"]): float(row[metric]) for row in rows if row.get("method") == method and row.get(metric) is not None}

    def bootstrap_ci(self, values: list[float], n_bootstrap: int = 1000, confidence: float = 0.95) -> tuple[float, float]:
        if not values:
            return float("nan"), float("nan")
        arr = np.asarray(values, dtype="float64")
        rng = np.random.default_rng(1234)
        means = [float(rng.choice(arr, size=arr.size, replace=True).mean()) for _ in range(n_bootstrap)]
        alpha = (1.0 - confidence) / 2.0
        return round(float(np.quantile(means, alpha)), 6), round(float(np.quantile(means, 1.0 - alpha)), 6)

    def _write_csv(self, path: Path, rows: list[dict[str, object]]) -> None:
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        keys = sorted({key for row in rows for key in row.keys()})
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)

    def _write_json(self, path: Path, rows: list[dict[str, object]]) -> None:
        path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    def _latest_file(self, directory: Path, pattern: str) -> Path | None:
        files = sorted(directory.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
        return files[0] if files else None

    def _markdown_table(self, rows: list[dict[str, object]]) -> str:
        if not rows:
            return "No rows generated."
        preferred = [
            "method",
            "comparison",
            "metric",
            "n",
            "keep_ratio",
            "compression_ratio",
            "bandwidth_saved_percent",
            "psnr",
            "ssim",
            "lpips",
            "semantic_utility_score",
            "energy_j",
            "paired_t_p",
            "wilcoxon_p",
            "cohens_d",
            "bootstrap_ci_low",
            "bootstrap_ci_high",
            "note",
        ]
        keys = [key for key in preferred if key in rows[0]]
        if not keys:
            keys = list(rows[0].keys())[:8]
        header = "| " + " | ".join(keys) + " |"
        sep = "| " + " | ".join(["---"] * len(keys)) + " |"
        lines = [header, sep]
        for row in rows:
            lines.append("| " + " | ".join(str(row.get(key, "")) for key in keys) + " |")
        return "\n".join(lines)
