"""evaluation.dataset_evaluator

Plain-English purpose: Benchmark pipelines, statistics, and publication-oriented experiments.

This file is part of the public Space Semantic Compression research demo.
It is documented at module level so researchers, supervisors, and non-specialist
readers can understand where the file fits before reading implementation details.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from backend.schemas.response_schema import TransmissionConfig
from backend.services.compression_service import CompressionService


class DatasetEvaluator:
    def __init__(self, compression_service: CompressionService, output_dir: Path) -> None:
        self.compression_service = compression_service
        self.output_dir = output_dir

    def evaluate_directory(
        self,
        dataset_dir: Path,
        keep_ratio: float = 0.45,
        mission: str = "wildfire_detection",
        limit: int | None = None,
    ) -> dict[str, object]:
        paths = sorted(
            [
                *dataset_dir.rglob("*.jpg"),
                *dataset_dir.rglob("*.jpeg"),
                *dataset_dir.rglob("*.png"),
                *dataset_dir.rglob("*.webp"),
            ]
        )
        if limit is not None:
            paths = paths[:limit]
        rows: list[dict[str, object]] = []
        for path in paths:
            result = self.compression_service.compress_image(
                path.read_bytes(),
                filename=path.name,
                transmission_config=TransmissionConfig(semantic_keep_ratio=keep_ratio),
                mission=mission,
            )
            rows.append(
                {
                    "image": str(path),
                    "compression_ratio": result.compression_ratio,
                    "bandwidth_saved_percent": result.bandwidth_saved_percent,
                    "psnr": result.psnr,
                    "ssim": result.ssim,
                    "lpips": result.lpips,
                    "sus": result.semantic_utility_score,
                    "objective": result.objective_value,
                    "energy_j": result.total_energy_j,
                    "latency_ms": result.inference_latency_ms,
                }
            )
        summary = self._summary(rows)
        result_dir = self.output_dir / "dataset_evaluations"
        result_dir.mkdir(parents=True, exist_ok=True)
        csv_path = result_dir / f"{dataset_dir.name}_keep_{int(keep_ratio * 100)}.csv"
        json_path = result_dir / f"{dataset_dir.name}_keep_{int(keep_ratio * 100)}.json"
        self._write_csv(csv_path, rows)
        json_path.write_text(json.dumps({"rows": rows, "summary": summary}, indent=2), encoding="utf-8")
        return {"csv_path": str(csv_path), "json_path": str(json_path), "summary": summary, "n": len(rows)}

    def _summary(self, rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
        metrics = ["compression_ratio", "bandwidth_saved_percent", "psnr", "ssim", "lpips", "sus", "objective", "energy_j"]
        summary: dict[str, dict[str, float]] = {}
        for metric in metrics:
            values = np.asarray([row[metric] for row in rows if row.get(metric) is not None], dtype="float64")
            if values.size == 0:
                continue
            ci95 = 1.96 * float(values.std(ddof=1)) / np.sqrt(values.size) if values.size > 1 else 0.0
            summary[metric] = {
                "mean": round(float(values.mean()), 6),
                "median": round(float(np.median(values)), 6),
                "std": round(float(values.std(ddof=1)) if values.size > 1 else 0.0, 6),
                "ci95": round(ci95, 6),
            }
        return summary

    def _write_csv(self, path: Path, rows: list[dict[str, object]]) -> None:
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
