"""backend.routes.compress

Plain-English purpose: FastAPI web service and application orchestration.

This file is part of the public Space Semantic Compression research demo.
It is documented at module level so researchers, supervisors, and non-specialist
readers can understand where the file fits before reading implementation details.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from backend.config import get_settings
from backend.schemas.response_schema import CompressionResponse, TransmissionConfig
from backend.services.compression_service import CompressionService
from backend.services.factory import get_compression_service

logger = logging.getLogger(__name__)
router = APIRouter(tags=["compression"])


@router.post("/compress", response_model=CompressionResponse)
async def compress_image(
    image: UploadFile = File(...),
    semantic_keep_ratio: float = Form(0.45),
    bandwidth_kbps: float = Form(256.0),
    latency_ms: float = Form(600.0),
    packet_loss_percent: float = Form(2.0),
    outage_probability: float = Form(0.05),
    token_selection_mode: str = Form("mission_utility"),
    semantic_method: str = Form("hybrid"),
    mission: str = Form("wildfire_detection"),
    service: CompressionService = Depends(get_compression_service),
) -> CompressionResponse:
    if image.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Upload a JPEG, PNG, or WebP image.",
        )

    payload = await image.read()
    settings = get_settings()
    if len(payload) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image exceeds {settings.max_upload_mb}MB upload limit.",
        )

    try:
        config = TransmissionConfig(
            semantic_keep_ratio=semantic_keep_ratio,
            bandwidth_kbps=bandwidth_kbps,
            latency_ms=latency_ms,
            packet_loss_percent=packet_loss_percent,
            outage_probability=outage_probability,
            token_selection_mode=token_selection_mode,
        )
        return service.compress_image(payload, image.filename or "upload", config, semantic_method=semantic_method, mission=mission)
    except FileNotFoundError as exc:
        logger.exception("Compression checkpoint is unavailable")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Compression failed")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Compression failed") from exc
