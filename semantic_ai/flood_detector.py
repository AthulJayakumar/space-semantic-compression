"""semantic_ai.flood_detector

Plain-English purpose: Mission-specific detectors that turn images into utility maps.

This file is part of the public Space Semantic Compression research demo.
It is documented at module level so researchers, supervisors, and non-specialist
readers can understand where the file fits before reading implementation details.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from semantic_ai.detector_base import MissionDetector, MissionDetectorOutput


class FloodDetector(MissionDetector):
    mission_name = "flood_detection"

    def _vision_detect(self, image: Image.Image) -> MissionDetectorOutput:
        arr = np.asarray(image).astype("float32") / 255.0
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        brightness = arr.mean(axis=2)
        blue_water = np.clip(b - 0.7 * r - 0.3 * g, 0.0, 1.0)
        muddy_water = np.clip((r + g) * 0.5 - b * 0.4, 0.0, 1.0) * (brightness < 0.62)
        confidence = self._normalize(0.65 * blue_water + 0.35 * muddy_water)
        mask = confidence > max(0.30, float(np.quantile(confidence, 0.88)))
        detections = self._connected_components(mask, "flood_water", confidence)
        relevance = self._normalize(confidence * (1.0 - np.clip(brightness - 0.85, 0.0, 1.0)))
        utility = self._normalize(0.60 * confidence + 0.40 * relevance)
        return MissionDetectorOutput(
            mission=self.mission_name,
            utility_map=utility,
            confidence_map=confidence,
            relevance_map=relevance,
            detections=detections,
            backend="vision_water_index",
        )

    def _label_relevance(self, label: str) -> float:
        return 1.0 if any(key in label.lower() for key in ("flood", "water", "river")) else 0.30
