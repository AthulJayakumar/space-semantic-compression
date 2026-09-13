"""tests.test_services

Plain-English purpose: Automated tests proving core services and research components work.

This file is part of the public Space Semantic Compression research demo.
It is documented at module level so researchers, supervisors, and non-specialist
readers can understand where the file fits before reading implementation details.
"""

from PIL import Image

import numpy as np
import torch

from backend.services.metrics_service import MetricsService
from backend.services.semantic_service import SemanticService
from backend.services.token_transmission_service import TokenTransmissionService
from backend.services.transmission_service import SatelliteTransmissionService
from backend.schemas.response_schema import TransmissionConfig


def test_metrics_identical_image():
    image = Image.new("RGB", (32, 32), color=(128, 64, 32))
    metrics = MetricsService()

    assert metrics.psnr(image, image) == float("inf")
    assert metrics.ssim(image, image) == 1.0


def test_semantic_service_counts_token_regions():
    image = Image.new("RGB", (64, 64), color="black")
    for x in range(16, 48):
        for y in range(16, 48):
            image.putpixel((x, y), (255, 255, 255))

    count = SemanticService().semantic_token_count(image, (8, 8))

    assert 0 < count <= 64


def test_semantic_analysis_returns_regions_and_importance_shape():
    image = Image.new("RGB", (64, 64), color="black")
    for x in range(20, 44):
        for y in range(20, 44):
            image.putpixel((x, y), (220, 220, 220))

    analysis = SemanticService().analyze(image, (8, 8))

    assert analysis.importance_map.shape == (8, 8)
    assert analysis.method == "hybrid"


def test_token_pruning_keeps_high_importance_cells():
    service = TokenTransmissionService()
    tokens = torch.arange(16).reshape(1, 4, 4)
    importance = np.zeros((4, 4), dtype="float32")
    importance[0, 0] = 1.0

    keep_mask = service.build_keep_mask(importance, keep_ratio=0.25)
    pruned = service.prune_tokens(tokens, keep_mask)

    assert keep_mask.sum() == 4
    assert keep_mask[0, 0]
    assert int(pruned[0, 0, 0]) == 0


def test_satellite_simulation_saves_time_for_smaller_payload():
    stats = SatelliteTransmissionService().simulate(
        full_payload_kb=100.0,
        semantic_payload_kb=25.0,
        token_entropy_bits=5.0,
        semantic_fidelity_percent=80.0,
        config=TransmissionConfig(bandwidth_kbps=100.0, latency_ms=500.0),
    )

    assert stats.transmission_time_saved_sec > 0
    assert stats.packets_total > 0


def test_transmission_config_defaults_to_mission_utility_mode():
    config = TransmissionConfig()

    assert config.token_selection_mode == "mission_utility"
