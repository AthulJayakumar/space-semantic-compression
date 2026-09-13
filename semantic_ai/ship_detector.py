"""semantic_ai.ship_detector

Plain-English purpose: Mission-specific detectors that turn images into utility maps.

This file is part of the public Space Semantic Compression research demo.
It is documented at module level so researchers, supervisors, and non-specialist
readers can understand where the file fits before reading implementation details.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from semantic_ai.detector_base import MissionDetector, MissionDetectorOutput


class ShipDetector(MissionDetector):
    mission_name = "ship_detection"

    def _vision_detect(self, image: Image.Image) -> MissionDetectorOutput:
        arr = np.asarray(image).astype("float32") / 255.0
        gray = arr.mean(axis=2)
        water = (arr[..., 2] > arr[..., 1] * 0.95) & (arr[..., 2] > arr[..., 0] * 1.05)
        bright_objects = np.clip(gray - 0.62, 0.0, 1.0)
        local_contrast = self._local_contrast(gray)
        confidence = self._normalize((0.55 * bright_objects + 0.45 * local_contrast) * water.astype("float32"))
        mask = confidence > max(0.25, float(np.quantile(confidence, 0.94)))
        detections = self._connected_components(mask, "ship_or_wake", confidence)
        relevance = self._normalize(confidence + 0.2 * water.astype("float32"))
        utility = self._normalize(0.75 * confidence + 0.25 * relevance)
        return MissionDetectorOutput(
            mission=self.mission_name,
            utility_map=utility,
            confidence_map=confidence,
            relevance_map=relevance,
            detections=detections,
            backend="vision_ship_candidate",
        )

    def _local_contrast(self, gray: np.ndarray) -> np.ndarray:
        try:
            import cv2

            blur = cv2.GaussianBlur(gray, (9, 9), 0)
            return np.abs(gray - blur)
        except Exception:
            return np.abs(gray - gray.mean())

    def _label_relevance(self, label: str) -> float:
        return 1.0 if any(key in label.lower() for key in ("ship", "vessel", "boat")) else 0.25
