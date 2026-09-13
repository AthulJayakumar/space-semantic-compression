"""deployment.edge_benchmark

Plain-English purpose: Edge-device profiling and export helpers.

This file is part of the public Space Semantic Compression research demo.
It is documented at module level so researchers, supervisors, and non-specialist
readers can understand where the file fits before reading implementation details.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import statistics
import time
import tracemalloc
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from backend.services.decoder_service import DecoderService
from backend.services.encoder_service import EncoderService
from backend.utils.image_utils import load_image_bytes
from backend.utils.tensor_utils import image_to_tensor, tensor_to_image


@dataclass(frozen=True)
class EdgeBenchmarkRow:
    image: str
    backend: str
    target: str
    device: str
    status: str
    latency_ms: float | None
    preprocessing_ms: float | None
    tokenization_ms: float | None
    reconstruction_ms: float | None
    postprocessing_ms: float | None
    peak_memory_mb: float | None
    fps: float | None
    notes: str


class EdgeDeploymentBenchmark:
    """Benchmark the current VQ-VAE inference path for deployment reporting."""

    def __init__(
        self,
        checkpoint_path: Path = Path("checkpoints/vqvae_s16k8.pt"),
        output_dir: Path = Path("results/edge"),
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, image_path: Path, repeats: int = 3, include_simulated_edge: bool = True) -> list[EdgeBenchmarkRow]:
        rows: list[EdgeBenchmarkRow] = []
        rows.append(self._benchmark_pytorch(image_path, "cpu", repeats))
        if torch.cuda.is_available():
            rows.append(self._benchmark_pytorch(image_path, "cuda", repeats))
        else:
            rows.append(
                EdgeBenchmarkRow(
                    image=str(image_path),
                    backend="PyTorch",
                    target="NVIDIA GPU",
                    device="cuda",
                    status="unavailable",
                    latency_ms=None,
                    preprocessing_ms=None,
                    tokenization_ms=None,
                    reconstruction_ms=None,
                    postprocessing_ms=None,
                    peak_memory_mb=None,
                    fps=None,
                    notes="CUDA was not available on this machine.",
                )
            )
        rows.extend(self._optional_runtime_rows(image_path))
        if include_simulated_edge:
            rows.extend(self._simulated_jetson_rows(rows, image_path))
        self._write_outputs(rows)
        return rows

    def _benchmark_pytorch(self, image_path: Path, device: str, repeats: int) -> EdgeBenchmarkRow:
        try:
            encoder = EncoderService(self.checkpoint_path, device)
            decoder = DecoderService(encoder)
            payload = image_path.read_bytes()
            original = load_image_bytes(payload)
            _ = encoder.model

            measurements: list[dict[str, float]] = []
            for _idx in range(max(1, repeats)):
                tracemalloc.start()
                if device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()

                t0 = time.perf_counter()
                tensor = image_to_tensor(original, encoder.device, encoder.stride)
                t1 = time.perf_counter()
                with torch.no_grad():
                    tokens = encoder.encode(tensor)
                if device == "cuda":
                    torch.cuda.synchronize()
                t2 = time.perf_counter()
                with torch.no_grad():
                    reconstruction = decoder.decode(tokens)
                if device == "cuda":
                    torch.cuda.synchronize()
                t3 = time.perf_counter()
                _ = tensor_to_image(reconstruction)
                t4 = time.perf_counter()

                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                cuda_peak = torch.cuda.max_memory_allocated() if device == "cuda" else 0
                measurements.append(
                    {
                        "preprocessing_ms": (t1 - t0) * 1000,
                        "tokenization_ms": (t2 - t1) * 1000,
                        "reconstruction_ms": (t3 - t2) * 1000,
                        "postprocessing_ms": (t4 - t3) * 1000,
                        "latency_ms": (t4 - t0) * 1000,
                        "peak_memory_mb": max(peak, cuda_peak) / (1024 * 1024),
                    }
                )

            avg = {key: statistics.mean(row[key] for row in measurements) for key in measurements[0]}
            fps = 1000.0 / avg["latency_ms"] if avg["latency_ms"] > 0 else None
            return EdgeBenchmarkRow(
                image=str(image_path),
                backend="PyTorch",
                target="workstation_cpu" if device == "cpu" else "workstation_gpu",
                device=device,
                status="measured",
                latency_ms=round(avg["latency_ms"], 4),
                preprocessing_ms=round(avg["preprocessing_ms"], 4),
                tokenization_ms=round(avg["tokenization_ms"], 4),
                reconstruction_ms=round(avg["reconstruction_ms"], 4),
                postprocessing_ms=round(avg["postprocessing_ms"], 4),
                peak_memory_mb=round(avg["peak_memory_mb"], 4),
                fps=round(fps, 4) if fps is not None else None,
                notes="Measured encode/decode inference path.",
            )
        except Exception as exc:
            return EdgeBenchmarkRow(
                image=str(image_path),
                backend="PyTorch",
                target="workstation_cpu" if device == "cpu" else "workstation_gpu",
                device=device,
                status="failed",
                latency_ms=None,
                preprocessing_ms=None,
                tokenization_ms=None,
                reconstruction_ms=None,
                postprocessing_ms=None,
                peak_memory_mb=None,
                fps=None,
                notes=str(exc),
            )

    def _optional_runtime_rows(self, image_path: Path) -> list[EdgeBenchmarkRow]:
        rows: list[EdgeBenchmarkRow] = []
        onnx_available = importlib.util.find_spec("onnxruntime") is not None
        onnx_path = Path("models/checkpoints/vqvae.onnx")
        rows.append(
            EdgeBenchmarkRow(
                image=str(image_path),
                backend="ONNX Runtime",
                target="edge_runtime",
                device="cpu/gpu",
                status="ready" if onnx_available and onnx_path.exists() else "unavailable",
                latency_ms=None,
                preprocessing_ms=None,
                tokenization_ms=None,
                reconstruction_ms=None,
                postprocessing_ms=None,
                peak_memory_mb=None,
                fps=None,
                notes="Export an ONNX model with scripts/export_vqvae_onnx.py before measuring." if not onnx_path.exists() else "ONNX runtime detected; measurement hook is ready.",
            )
        )
        tensorrt_available = importlib.util.find_spec("tensorrt") is not None
        rows.append(
            EdgeBenchmarkRow(
                image=str(image_path),
                backend="TensorRT",
                target="jetson_or_desktop_gpu",
                device="cuda",
                status="ready" if tensorrt_available else "unavailable",
                latency_ms=None,
                preprocessing_ms=None,
                tokenization_ms=None,
                reconstruction_ms=None,
                postprocessing_ms=None,
                peak_memory_mb=None,
                fps=None,
                notes="TensorRT Python bindings were not found." if not tensorrt_available else "TensorRT bindings detected; engine export can be added for hardware runs.",
            )
        )
        return rows

    def _simulated_jetson_rows(self, rows: list[EdgeBenchmarkRow], image_path: Path) -> list[EdgeBenchmarkRow]:
        measured = next((row for row in rows if row.backend == "PyTorch" and row.status == "measured" and row.latency_ms), None)
        if measured is None:
            return []
        profiles = [
            ("Jetson Nano simulated", 4.5, 0.65, "Simulated low-power CubeSat-class edge profile."),
            ("Jetson Orin Nano simulated", 1.8, 0.85, "Simulated modern edge-GPU profile."),
        ]
        simulated: list[EdgeBenchmarkRow] = []
        for target, latency_multiplier, memory_multiplier, notes in profiles:
            latency = measured.latency_ms * latency_multiplier
            simulated.append(
                EdgeBenchmarkRow(
                    image=str(image_path),
                    backend="PyTorch",
                    target=target,
                    device="simulated_edge",
                    status="simulated",
                    latency_ms=round(latency, 4),
                    preprocessing_ms=self._scaled(measured.preprocessing_ms, latency_multiplier),
                    tokenization_ms=self._scaled(measured.tokenization_ms, latency_multiplier),
                    reconstruction_ms=self._scaled(measured.reconstruction_ms, latency_multiplier),
                    postprocessing_ms=self._scaled(measured.postprocessing_ms, latency_multiplier),
                    peak_memory_mb=self._scaled(measured.peak_memory_mb, memory_multiplier),
                    fps=round(1000.0 / latency, 4) if latency > 0 else None,
                    notes=notes,
                )
            )
        return simulated

    def _scaled(self, value: float | None, multiplier: float) -> float | None:
        return round(value * multiplier, 4) if value is not None else None

    def _write_outputs(self, rows: list[EdgeBenchmarkRow]) -> None:
        payload = [asdict(row) for row in rows]
        (self.output_dir / "edge_benchmark.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if not payload:
            (self.output_dir / "edge_benchmark.csv").write_text("", encoding="utf-8")
            return
        with (self.output_dir / "edge_benchmark.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(payload[0].keys()))
            writer.writeheader()
            writer.writerows(payload)
