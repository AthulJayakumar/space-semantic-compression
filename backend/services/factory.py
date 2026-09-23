"""backend.services.factory

Plain-English purpose: FastAPI web service and application orchestration.

This file is part of the public Space Semantic Compression research demo.
It is documented at module level so researchers, supervisors, and non-specialist
readers can understand where the file fits before reading implementation details.
"""

from functools import lru_cache

from backend.config import get_settings
from backend.services.benchmark_service import BenchmarkService
from backend.services.compression_service import CompressionService
from backend.services.decoder_service import DecoderService
from backend.services.encoder_service import EncoderService
from backend.services.metrics_service import MetricsService
from backend.services.semantic_service import SemanticService
from backend.services.token_transmission_service import TokenTransmissionService
from backend.services.transmission_service import SatelliteTransmissionService
from backend.services.visualization_service import VisualizationService


@lru_cache(maxsize=1)
def get_compression_service() -> CompressionService:
    settings = get_settings()
    encoder = EncoderService(
        settings.checkpoint_path,
        settings.device,
        expected_sha256=settings.checkpoint_sha256,
    )
    decoder = DecoderService(encoder)
    return CompressionService(
        encoder_service=encoder,
        decoder_service=decoder,
        metrics_service=MetricsService(),
        semantic_service=SemanticService(),
        token_service=TokenTransmissionService(),
        transmission_service=SatelliteTransmissionService(),
        visualization_service=VisualizationService(settings.output_dir),
        output_dir=settings.output_dir,
    )


@lru_cache(maxsize=1)
def get_benchmark_service() -> BenchmarkService:
    settings = get_settings()
    return BenchmarkService(get_compression_service(), settings.output_dir)
