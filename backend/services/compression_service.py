"""backend.services.compression_service

Plain-English purpose: FastAPI web service and application orchestration.

This file is part of the public Space Semantic Compression research demo.
It is documented at module level so researchers, supervisors, and non-specialist
readers can understand where the file fits before reading implementation details.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from metrics.semantic_utility import SemanticUtilityMetric
from research.objective import ResearchObjective
from semantic_ai import FloodDetector, ShipDetector, WildfireDetector
from backend.schemas.response_schema import (
    CompressionResponse,
    InferenceProfile,
    SemanticAnalysisResponse,
    TransmissionConfig,
)
from backend.services.decoder_service import DecoderService
from backend.services.encoder_service import EncoderService
from backend.services.metrics_service import MetricsService
from backend.services.profiling_service import InferenceProfiler
from backend.services.semantic_service import SemanticService
from backend.services.token_transmission_service import TokenTransmissionService
from backend.services.transmission_service import SatelliteTransmissionService
from backend.services.visualization_service import VisualizationService
from token_selection.utility_pruner import UtilityAwareTokenPruner
from transmission.energy_model import EnergyModel
from backend.utils.file_utils import unique_output_path
from backend.utils.image_utils import load_image_bytes, save_png
from backend.utils.tensor_utils import image_to_tensor, tensor_to_image

logger = logging.getLogger(__name__)


class CompressionService:
    def __init__(
        self,
        encoder_service: EncoderService,
        decoder_service: DecoderService,
        metrics_service: MetricsService,
        semantic_service: SemanticService,
        token_service: TokenTransmissionService,
        transmission_service: SatelliteTransmissionService,
        visualization_service: VisualizationService,
        output_dir: Path,
    ) -> None:
        self.encoder_service = encoder_service
        self.decoder_service = decoder_service
        self.metrics_service = metrics_service
        self.semantic_service = semantic_service
        self.token_service = token_service
        self.transmission_service = transmission_service
        self.visualization_service = visualization_service
        self.output_dir = output_dir
        self.utility_pruner = UtilityAwareTokenPruner()
        self.semantic_utility_metric = SemanticUtilityMetric()
        self.energy_model = EnergyModel()
        self.research_objective = ResearchObjective()
        self.detectors = {
            "wildfire_detection": WildfireDetector("models/checkpoints/wildfire_yolo.pt"),
            "flood_detection": FloodDetector("models/checkpoints/flood_yolo.pt"),
            "ship_detection": ShipDetector("models/checkpoints/ship_yolo.pt"),
        }

    def compress_image(
        self,
        image_bytes: bytes,
        filename: str = "upload.png",
        transmission_config: TransmissionConfig | None = None,
        semantic_method: str = "hybrid",
        mission: str = "wildfire_detection",
    ) -> CompressionResponse:
        config = transmission_config or TransmissionConfig()
        profiler = InferenceProfiler(self.encoder_service.device)

        with profiler.track("preprocessing"):
            original = load_image_bytes(image_bytes)
            tensor = image_to_tensor(original, self.encoder_service.device, self.encoder_service.stride)

        with profiler.track("encoding"):
            tokens = self.encoder_service.encode(tensor)
            profiler.token_count = int(tokens.numel())

        token_shape = tuple(tokens.shape[-2:])
        with profiler.track("semantic"):
            semantic_analysis = self.semantic_service.analyze(original, token_shape, method=semantic_method)
            detector_output = self._detect_mission_utility(original, token_shape, mission)
            utility_map = np.maximum(semantic_analysis.importance_map, detector_output.utility_map)
            detail_map = self.semantic_service.detail_map(original, token_shape)
            keep_mask, token_scores = self.utility_pruner.select(
                tokens,
                utility_map,
                config.semantic_keep_ratio,
                detail_map=detail_map,
            )
            semantic_tokens = int(keep_mask.sum())
            pruned_tokens = self.token_service.prune_tokens(tokens, keep_mask)
            token_entropy = self.token_service.token_entropy_bits(pruned_tokens)
            semantic_fidelity = self.token_service.semantic_fidelity_percent(utility_map, keep_mask)

        with profiler.track("transmission"):
            full_payload_kb = self.token_service.estimate_payload_kb(tokens)
            compressed_size_kb = self.token_service.estimate_payload_kb(pruned_tokens, keep_mask)
            transmission = self.transmission_service.simulate(
                full_payload_kb=full_payload_kb,
                semantic_payload_kb=compressed_size_kb,
                token_entropy_bits=token_entropy,
                semantic_fidelity_percent=semantic_fidelity,
                config=config,
            )

        with profiler.track("reconstruction"):
            reconstruction_tensor = self.decoder_service.decode(pruned_tokens)
            reconstruction = tensor_to_image(reconstruction_tensor)
            detector_after = self._detect_mission_utility(reconstruction, token_shape, mission)
            sus, sus_components = self.semantic_utility_metric.score_before_after(detector_output, detector_after)

        with profiler.track("postprocessing"):
            output_path = unique_output_path(self.output_dir)
            save_png(reconstruction, output_path)
            heatmap_path = self.visualization_service.save_semantic_heatmap(
                original,
                utility_map,
                semantic_analysis.regions,
            )
            token_mask_path = self.visualization_service.save_token_mask(keep_mask)

        original_size_kb = self._original_size_kb(image_bytes)
        compression_ratio = original_size_kb / compressed_size_kb if compressed_size_kb else 0.0
        total_tokens = int(tokens.numel())
        bandwidth_savings = max(0.0, (1.0 - compressed_size_kb / original_size_kb) * 100.0) if original_size_kb else 0.0
        profile = self._build_profile(profiler)
        lpips_value = self.metrics_service.lpips(original, reconstruction)
        distortion = 1.0 - max(0.0, min(1.0, self.metrics_service.ssim(original, reconstruction)))
        rate = compressed_size_kb / max(original_size_kb, 1e-9)
        energy = self.energy_model.estimate(
            payload_kb=compressed_size_kb,
            baseline_payload_kb=full_payload_kb,
            compute_latency_ms=profile.total_latency_ms,
        )
        objective = self.research_objective.evaluate(
            rate=rate,
            distortion=distortion,
            energy=min(1.0, energy.total_energy_j / 100.0),
            utility=sus / 100.0,
        )

        logger.info(
            "Compressed %s: %.2fKB -> %.2fKB, tokens=%d, semantic_tokens=%d",
            filename,
            original_size_kb,
            compressed_size_kb,
            total_tokens,
            semantic_tokens,
        )

        return CompressionResponse(
            compression_ratio=round(compression_ratio, 4),
            original_size_kb=round(original_size_kb, 4),
            compressed_size_kb=round(compressed_size_kb, 4),
            psnr=round(self.metrics_service.psnr(original, reconstruction), 4),
            ssim=round(self.metrics_service.ssim(original, reconstruction), 4),
            lpips=round(lpips_value, 6) if lpips_value is not None else None,
            reconstructed_image_path=str(output_path),
            semantic_token_count=semantic_tokens,
            total_token_count=total_tokens,
            bandwidth_savings_percent=round(bandwidth_savings, 2),
            bandwidth_saved_percent=round(bandwidth_savings, 2),
            transmission_time_saved=transmission.transmission_time_saved_sec,
            semantic_regions_detected=len(semantic_analysis.regions),
            semantic_regions=semantic_analysis.regions,
            transmission=transmission,
            profile=profile,
            inference_latency_ms=round(profile.total_latency_ms, 4),
            semantic_heatmap_path=str(heatmap_path),
            token_mask_path=str(token_mask_path),
            mission=mission,
            detector_backend=detector_output.backend,
            semantic_utility_score=round(sus, 4),
            objective_value=objective.objective,
            rate_component=objective.rate,
            distortion_component=objective.distortion,
            energy_component=objective.energy,
            utility_component=objective.utility,
            transmission_energy_j=energy.transmission_energy_j,
            compute_energy_j=energy.compute_energy_j,
            total_energy_j=energy.total_energy_j,
            detector_retention=round(sus_components.detector_retention, 6),
            object_retention=round(sus_components.object_retention, 6),
            relevance_retention=round(sus_components.relevance_retention, 6),
            region_preservation=round(sus_components.region_preservation, 6),
        )

    def _original_size_kb(self, image_bytes: bytes) -> float:
        return len(image_bytes) / 1024.0

    def analyze_semantic_regions(self, image_bytes: bytes, method: str = "hybrid") -> SemanticAnalysisResponse:
        original = load_image_bytes(image_bytes)
        tensor = image_to_tensor(original, self.encoder_service.device, self.encoder_service.stride)
        token_shape = (
            int(tensor.shape[-2] // self.encoder_service.stride),
            int(tensor.shape[-1] // self.encoder_service.stride),
        )
        analysis = self.semantic_service.analyze(original, token_shape, method=method)
        threshold = float(analysis.importance_map.mean())
        semantic_count = int((analysis.importance_map >= threshold).sum())
        total_tokens = int(analysis.importance_map.size)
        return SemanticAnalysisResponse(
            semantic_regions_detected=len(analysis.regions),
            token_shape=[token_shape[0], token_shape[1]],
            semantic_token_count=semantic_count,
            total_token_count=total_tokens,
            semantic_coverage_percent=round(semantic_count / max(total_tokens, 1) * 100.0, 2),
            regions=analysis.regions,
            method=analysis.method,
        )

    def _build_profile(self, profiler: InferenceProfiler) -> InferenceProfile:
        return InferenceProfile(
            preprocessing_latency_ms=round(profiler.timings.get("preprocessing", 0.0), 4),
            encoding_latency_ms=round(profiler.timings.get("encoding", 0.0), 4),
            semantic_latency_ms=round(profiler.timings.get("semantic", 0.0), 4),
            transmission_latency_ms=round(profiler.timings.get("transmission", 0.0), 4),
            reconstruction_latency_ms=round(profiler.timings.get("reconstruction", 0.0), 4),
            postprocessing_latency_ms=round(profiler.timings.get("postprocessing", 0.0), 4),
            total_latency_ms=round(profiler.total_ms(), 4),
            device=str(self.encoder_service.device),
            token_generation_rate=round(profiler.token_generation_rate(), 4),
        )

    def _detect_mission_utility(self, original, token_shape: tuple[int, int], mission: str):
        detector = self.detectors.get(mission, self.detectors["wildfire_detection"])
        output = detector.detect(original)
        utility = self.semantic_service.resize_importance(output.utility_map, token_shape)
        confidence = self.semantic_service.resize_importance(output.confidence_map, token_shape)
        relevance = self.semantic_service.resize_importance(output.relevance_map, token_shape)
        return type(output)(
            mission=output.mission,
            utility_map=utility,
            confidence_map=confidence,
            relevance_map=relevance,
            detections=output.detections,
            backend=output.backend,
        )
