"""backend.services.benchmark_service

Plain-English purpose: FastAPI web service and application orchestration.

This file is part of the public Space Semantic Compression research demo.
It is documented at module level so researchers, supervisors, and non-specialist
readers can understand where the file fits before reading implementation details.
"""

from __future__ import annotations

import csv
import json
import time
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import torch

from baselines import AutoencoderCodec, LightweightCNNCodec, VariationalAutoencoderCodec
from backend.schemas.response_schema import BenchmarkEntry, BenchmarkResponse, TransmissionConfig
from backend.services.compression_service import CompressionService
from backend.services.experiment_tracker import ExperimentTracker
from metrics.semantic_utility import SemanticUtilityMetric
from backend.utils.image_utils import load_image_bytes
from backend.utils.tensor_utils import image_to_tensor, tensor_to_image


class BenchmarkService:
    def __init__(self, compression_service: CompressionService, output_dir: Path) -> None:
        self.compression_service = compression_service
        self.output_dir = output_dir
        self.tracker = ExperimentTracker(output_dir)
        self.sus_metric = SemanticUtilityMetric()

    def run_image_benchmark(self, image_bytes: bytes, filename: str = "benchmark.png") -> BenchmarkResponse:
        benchmark_id = uuid4().hex
        benchmark_dir = self.output_dir / "benchmarks"
        benchmark_dir.mkdir(parents=True, exist_ok=True)

        entries = [
            *[self._jpeg_baseline(image_bytes, quality=quality) for quality in (20, 30, 40, 50, 60, 70, 80)],
            self._semantic_vqvae(image_bytes, filename, keep_ratio=1.0, name="vqvae_full_tokens"),
            self._semantic_vqvae(image_bytes, filename, keep_ratio=0.45, name="semantic_vqvae_45pct"),
            self._semantic_vqvae(image_bytes, filename, keep_ratio=0.25, name="semantic_vqvae_25pct"),
            *self._trained_neural_baselines(image_bytes),
        ]

        experiment_dir = benchmark_dir / benchmark_id
        experiment_dir.mkdir(parents=True, exist_ok=True)
        csv_path = experiment_dir / "results.csv"
        json_path = experiment_dir / "results.json"
        self._write_csv(csv_path, entries)
        tensorboard_dir = self.tracker.write_tensorboard_scalars(
            experiment_dir,
            {
                f"{entry.model_name}/compression_ratio": entry.compression_ratio
                for entry in entries
                if entry.compression_ratio is not None
            },
        )
        metadata = {
            "benchmark_id": benchmark_id,
            "filename": filename,
            "hardware": self.tracker.hardware_metadata(),
            "tensorboard_dir": tensorboard_dir,
        }
        metadata_path = self.tracker.write_metadata(experiment_dir, "benchmark", metadata)
        json_path.write_text(
            json.dumps(
                {
                    "benchmark_id": benchmark_id,
                    "metadata_path": str(metadata_path),
                    "entries": [entry.model_dump() for entry in entries],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return BenchmarkResponse(
            benchmark_id=benchmark_id,
            entries=entries,
            csv_path=str(csv_path),
            json_path=str(json_path),
        )

    def _semantic_vqvae(self, image_bytes: bytes, filename: str, keep_ratio: float, name: str) -> BenchmarkEntry:
        start = time.perf_counter()
        result = self.compression_service.compress_image(
            image_bytes,
            filename=filename,
            transmission_config=TransmissionConfig(semantic_keep_ratio=keep_ratio),
        )
        latency = (time.perf_counter() - start) * 1000.0
        return BenchmarkEntry(
            model_name=name,
            status="ok",
            compression_ratio=result.compression_ratio,
            bandwidth_saved_percent=result.bandwidth_saved_percent,
            psnr=result.psnr,
            ssim=result.ssim,
            lpips=result.lpips,
            semantic_fidelity_percent=result.transmission.semantic_fidelity_percent if result.transmission else None,
            semantic_utility_score=result.semantic_utility_score,
            objective_value=result.objective_value,
            total_energy_j=result.total_energy_j,
            token_entropy_bits=result.transmission.token_entropy_bits if result.transmission else None,
            inference_latency_ms=round(latency, 4),
        )

    def _jpeg_baseline(self, image_bytes: bytes, quality: int) -> BenchmarkEntry:
        image = load_image_bytes(image_bytes)
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=quality, optimize=True)
        original_kb = len(image_bytes) / 1024.0
        compressed_kb = len(buffer.getvalue()) / 1024.0
        ratio = original_kb / compressed_kb if compressed_kb else 0.0
        decoded = load_image_bytes(buffer.getvalue())
        metrics = self.compression_service.metrics_service
        semantic = self.compression_service.semantic_service.analyze(image, (32, 32))
        jpeg_semantic = self.compression_service.semantic_service.analyze(decoded, (32, 32))
        keep_mask = jpeg_semantic.importance_map >= float(jpeg_semantic.importance_map.mean())
        sus, _ = self.sus_metric.score(
            detector_confidence=semantic.importance_map,
            relevance_map=semantic.importance_map,
            utility_map=semantic.importance_map,
            keep_mask=keep_mask,
        )
        lpips_value = metrics.lpips(image, decoded)
        return BenchmarkEntry(
            model_name=f"jpeg_quality_{quality}",
            status="ok",
            compression_ratio=round(ratio, 4),
            bandwidth_saved_percent=round(max(0.0, (1.0 - compressed_kb / original_kb) * 100.0), 2),
            psnr=round(metrics.psnr(image, decoded), 4),
            ssim=round(metrics.ssim(image, decoded), 4),
            lpips=round(lpips_value, 6) if lpips_value is not None else None,
            semantic_utility_score=round(sus, 4),
            notes="Classical decoded JPEG baseline.",
        )

    def _trained_neural_baselines(self, image_bytes: bytes) -> list[BenchmarkEntry]:
        specs = [
            ("standard_autoencoder", "models/checkpoints/autoencoder_baseline.pt", AutoencoderCodec),
            ("variational_autoencoder", "models/checkpoints/vae_baseline.pt", VariationalAutoencoderCodec),
            ("lightweight_cnn", "models/checkpoints/lightweight_cnn_baseline.pt", LightweightCNNCodec),
        ]
        entries: list[BenchmarkEntry] = []
        for name, checkpoint_path, model_factory in specs:
            checkpoint = Path(checkpoint_path)
            if not checkpoint.exists():
                continue
            entries.append(self._evaluate_trained_baseline(name, checkpoint, model_factory, image_bytes))
        return entries

    def _evaluate_trained_baseline(self, name: str, checkpoint_path: Path, model_factory, image_bytes: bytes) -> BenchmarkEntry:
        start = time.perf_counter()
        device = self.compression_service.encoder_service.device
        image = load_image_bytes(image_bytes)
        tensor = image_to_tensor(image, device, stride=8)
        model = model_factory().to(device).eval()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model"])
        with torch.no_grad():
            output = model(tensor)
            xhat, latent = output[0], output[1]
        reconstruction = tensor_to_image(xhat)
        latent_kb = latent.detach().cpu().numpy().astype("float16").nbytes / 1024.0
        original_kb = len(image_bytes) / 1024.0
        ratio = original_kb / max(latent_kb, 1e-9)
        metrics = self.compression_service.metrics_service
        lpips_value = metrics.lpips(image, reconstruction)
        latency = (time.perf_counter() - start) * 1000.0
        return BenchmarkEntry(
            model_name=name,
            status="ok",
            compression_ratio=round(ratio, 4),
            bandwidth_saved_percent=round(max(0.0, (1.0 - latent_kb / original_kb) * 100.0), 2),
            psnr=round(metrics.psnr(image, reconstruction), 4),
            ssim=round(metrics.ssim(image, reconstruction), 4),
            lpips=round(lpips_value, 6) if lpips_value is not None else None,
            inference_latency_ms=round(latency, 4),
            notes=f"Evaluated from trained checkpoint {checkpoint_path}.",
        )

    def _write_csv(self, path: Path, entries: list[BenchmarkEntry]) -> None:
        fieldnames = list(BenchmarkEntry.model_fields.keys())
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for entry in entries:
                writer.writerow(entry.model_dump())
