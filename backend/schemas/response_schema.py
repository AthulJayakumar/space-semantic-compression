"""backend.schemas.response_schema

Plain-English purpose: FastAPI web service and application orchestration.

This file is part of the public Space Semantic Compression research demo.
It is documented at module level so researchers, supervisors, and non-specialist
readers can understand where the file fits before reading implementation details.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SemanticRegion(BaseModel):
    label: str
    bbox: list[int] = Field(..., description="[x, y, width, height]")
    confidence: float = Field(..., ge=0, le=1)
    priority: float = Field(..., ge=0, le=1)
    area_percent: float = Field(..., ge=0, le=100)


class TransmissionConfig(BaseModel):
    bandwidth_kbps: float = Field(default=256.0, gt=0)
    latency_ms: float = Field(default=600.0, ge=0)
    packet_loss_percent: float = Field(default=2.0, ge=0, le=100)
    outage_probability: float = Field(default=0.05, ge=0, le=1)
    packet_size_bytes: int = Field(default=1024, gt=0)
    semantic_keep_ratio: float = Field(default=0.45, gt=0, le=1)
    token_selection_mode: Literal["mission_utility", "reconstruction_balanced", "custom"] = "mission_utility"


class TransmissionStats(BaseModel):
    full_payload_kb: float = Field(..., ge=0)
    semantic_payload_kb: float = Field(..., ge=0)
    packets_total: int = Field(..., ge=0)
    packets_delivered: int = Field(..., ge=0)
    estimated_downlink_time_sec: float = Field(..., ge=0)
    baseline_downlink_time_sec: float = Field(..., ge=0)
    transmission_time_saved_sec: float
    semantic_fidelity_percent: float = Field(..., ge=0, le=100)
    token_entropy_bits: float = Field(..., ge=0)


class InferenceProfile(BaseModel):
    preprocessing_latency_ms: float = Field(..., ge=0)
    encoding_latency_ms: float = Field(..., ge=0)
    semantic_latency_ms: float = Field(..., ge=0)
    transmission_latency_ms: float = Field(..., ge=0)
    reconstruction_latency_ms: float = Field(..., ge=0)
    postprocessing_latency_ms: float = Field(..., ge=0)
    total_latency_ms: float = Field(..., ge=0)
    device: str
    token_generation_rate: float = Field(..., ge=0)


class CompressionResponse(BaseModel):
    compression_ratio: float = Field(..., ge=0)
    original_size_kb: float = Field(..., ge=0)
    compressed_size_kb: float = Field(..., ge=0)
    psnr: float
    ssim: float
    lpips: float | None = None
    reconstructed_image_path: str
    semantic_token_count: int = Field(..., ge=0)
    total_token_count: int = Field(..., ge=0)
    bandwidth_savings_percent: float
    bandwidth_saved_percent: float
    transmission_time_saved: float
    semantic_regions_detected: int = Field(..., ge=0)
    inference_latency_ms: float = Field(..., ge=0)
    semantic_regions: list[SemanticRegion] = Field(default_factory=list)
    transmission: TransmissionStats | None = None
    profile: InferenceProfile | None = None
    semantic_heatmap_path: str | None = None
    token_mask_path: str | None = None
    mission: str = "wildfire_detection"
    token_selection_mode: str = "mission_utility"
    detector_backend: str | None = None
    semantic_utility_score: float | None = None
    objective_value: float | None = None
    rate_component: float | None = None
    distortion_component: float | None = None
    energy_component: float | None = None
    utility_component: float | None = None
    transmission_energy_j: float | None = None
    compute_energy_j: float | None = None
    total_energy_j: float | None = None
    detector_retention: float | None = None
    object_retention: float | None = None
    relevance_retention: float | None = None
    region_preservation: float | None = None


class SemanticAnalysisResponse(BaseModel):
    semantic_regions_detected: int
    token_shape: list[int]
    semantic_token_count: int
    total_token_count: int
    semantic_coverage_percent: float
    regions: list[SemanticRegion]
    method: str


class BenchmarkEntry(BaseModel):
    model_name: str
    status: str
    compression_ratio: float | None = None
    bandwidth_saved_percent: float | None = None
    psnr: float | None = None
    ssim: float | None = None
    semantic_fidelity_percent: float | None = None
    token_entropy_bits: float | None = None
    lpips: float | None = None
    semantic_utility_score: float | None = None
    objective_value: float | None = None
    total_energy_j: float | None = None
    inference_latency_ms: float | None = None
    gpu_memory_mb: float | None = None
    notes: str | None = None


class BenchmarkResponse(BaseModel):
    benchmark_id: str
    entries: list[BenchmarkEntry]
    csv_path: str
    json_path: str
