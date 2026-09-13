"""tests.test_research_components

Plain-English purpose: Automated tests proving core services and research components work.

This file is part of the public Space Semantic Compression research demo.
It is documented at module level so researchers, supervisors, and non-specialist
readers can understand where the file fits before reading implementation details.
"""

import numpy as np
import torch
from PIL import Image

from metrics.semantic_utility import SemanticUtilityMetric
from research.objective import ResearchObjective
from semantic_ai.detector_base import MissionDetectorOutput
from semantic_ai import FloodDetector, ShipDetector, WildfireDetector
from backend.services.semantic_service import SemanticService
from token_selection.utility_pruner import TokenSelectionWeights, UtilityAwareTokenPruner
from transmission.energy_model import EnergyModel


def test_semantic_utility_score_range():
    utility = np.zeros((4, 4), dtype="float32")
    utility[:2, :2] = 1.0
    keep = np.zeros((4, 4), dtype=bool)
    keep[:2, :2] = True

    score, components = SemanticUtilityMetric().score(utility, utility, utility, keep)

    assert 0 <= score <= 100
    assert components.object_retention == 1.0


def test_formal_sus_components_from_detector_outputs():
    before = MissionDetectorOutput(
        mission="wildfire_detection",
        utility_map=np.ones((4, 4), dtype="float32"),
        confidence_map=np.ones((4, 4), dtype="float32"),
        relevance_map=np.ones((4, 4), dtype="float32"),
        detections=[{"label": "fire"}, {"label": "smoke"}],
        backend="test",
    )
    after = MissionDetectorOutput(
        mission="wildfire_detection",
        utility_map=np.ones((4, 4), dtype="float32") * 0.5,
        confidence_map=np.ones((4, 4), dtype="float32") * 0.5,
        relevance_map=np.ones((4, 4), dtype="float32") * 0.5,
        detections=[{"label": "fire"}],
        backend="test",
    )

    score, components = SemanticUtilityMetric().score_before_after(before, after)

    assert 0 <= score <= 100
    assert components.detector_retention == 0.5
    assert components.object_retention == 0.5
    assert components.relevance_retention == 0.5


def test_utility_pruner_prefers_high_utility_token():
    tokens = torch.arange(16).reshape(1, 4, 4)
    utility = np.zeros((4, 4), dtype="float32")
    utility[3, 3] = 1.0

    keep, scores = UtilityAwareTokenPruner().select(tokens, utility, keep_ratio=0.25)

    assert keep.sum() == 4
    assert keep[3, 3]
    assert scores.shape == (4, 4)


def test_reconstruction_balanced_pruner_can_prioritize_structural_detail():
    tokens = torch.zeros((1, 4, 4), dtype=torch.long)
    utility = np.zeros((4, 4), dtype="float32")
    detail = np.zeros((4, 4), dtype="float32")
    detail[1, 2] = 1.0

    pruner = UtilityAwareTokenPruner(TokenSelectionWeights.reconstruction_balanced())
    keep, scores = pruner.select(tokens, utility, keep_ratio=0.0625, detail_map=detail)

    assert keep.sum() == 1
    assert keep[1, 2]
    assert scores[1, 2] == scores.max()


def test_default_pruner_is_mission_utility_mode():
    weights = TokenSelectionWeights()

    assert weights.delta_detail == 0.0
    assert weights == TokenSelectionWeights.mission_utility()


def test_semantic_service_detail_map_matches_token_shape():
    image = Image.new("RGB", (32, 32), color=(20, 20, 20))
    for x in range(16, 32):
        for y in range(32):
            image.putpixel((x, y), (220, 220, 220))

    detail = SemanticService().detail_map(image, (4, 4))

    assert detail.shape == (4, 4)
    assert detail.min() >= 0
    assert detail.max() <= 1
    assert detail.sum() > 0


def test_energy_and_objective_are_finite():
    energy = EnergyModel().estimate(payload_kb=10, baseline_payload_kb=20, compute_latency_ms=100)
    objective = ResearchObjective().evaluate(rate=0.5, distortion=0.2, energy=0.01, utility=0.8)

    assert energy.total_energy_j > 0
    assert energy.energy_saved_j > 0
    assert objective.objective < 1


def test_mission_detectors_return_normalized_maps():
    image = Image.new("RGB", (32, 32), color=(180, 80, 30))

    for detector in (WildfireDetector(save_visualizations=False), FloodDetector(), ShipDetector()):
        output = detector.detect(image)
        assert output.utility_map.shape == (32, 32)
        assert output.utility_map.min() >= 0
        assert output.utility_map.max() <= 1
