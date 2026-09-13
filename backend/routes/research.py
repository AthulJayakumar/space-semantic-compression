"""backend.routes.research

Plain-English purpose: FastAPI web service and application orchestration.

This file is part of the public Space Semantic Compression research demo.
It is documented at module level so researchers, supervisors, and non-specialist
readers can understand where the file fits before reading implementation details.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from backend.schemas.response_schema import BenchmarkResponse, CompressionResponse, SemanticAnalysisResponse, TransmissionConfig
from backend.services.benchmark_service import BenchmarkService
from backend.services.compression_service import CompressionService
from backend.services.factory import get_benchmark_service, get_compression_service

logger = logging.getLogger(__name__)
router = APIRouter(tags=["research"])


@router.post("/analyze-semantic-regions", response_model=SemanticAnalysisResponse)
async def analyze_semantic_regions(
    image: UploadFile = File(...),
    method: str = Form("hybrid"),
    service: CompressionService = Depends(get_compression_service),
) -> SemanticAnalysisResponse:
    payload = await _read_image_payload(image)
    try:
        return service.analyze_semantic_regions(payload, method=method)
    except Exception as exc:
        logger.exception("Semantic region analysis failed")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Semantic analysis failed") from exc


@router.post("/simulate-transmission", response_model=CompressionResponse)
async def simulate_transmission(
    image: UploadFile = File(...),
    bandwidth_kbps: float = Form(256.0),
    latency_ms: float = Form(600.0),
    packet_loss_percent: float = Form(2.0),
    outage_probability: float = Form(0.05),
    packet_size_bytes: int = Form(1024),
    semantic_keep_ratio: float = Form(0.45),
    semantic_method: str = Form("hybrid"),
    mission: str = Form("wildfire_detection"),
    service: CompressionService = Depends(get_compression_service),
) -> CompressionResponse:
    payload = await _read_image_payload(image)
    config = TransmissionConfig(
        bandwidth_kbps=bandwidth_kbps,
        latency_ms=latency_ms,
        packet_loss_percent=packet_loss_percent,
        outage_probability=outage_probability,
        packet_size_bytes=packet_size_bytes,
        semantic_keep_ratio=semantic_keep_ratio,
    )
    try:
        return service.compress_image(payload, image.filename or "upload", config, semantic_method=semantic_method, mission=mission)
    except Exception as exc:
        logger.exception("Transmission simulation failed")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Transmission simulation failed") from exc


@router.post("/benchmark", response_model=BenchmarkResponse)
async def benchmark_image(
    image: UploadFile = File(...),
    service: BenchmarkService = Depends(get_benchmark_service),
) -> BenchmarkResponse:
    payload = await _read_image_payload(image)
    try:
        return service.run_image_benchmark(payload, image.filename or "benchmark.png")
    except Exception as exc:
        logger.exception("Benchmark failed")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Benchmark failed") from exc


async def _read_image_payload(image: UploadFile) -> bytes:
    if image.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Upload a JPEG, PNG, or WebP image.",
        )
    return await image.read()
