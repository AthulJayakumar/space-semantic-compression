"""semantic_ai.__init__

Plain-English purpose: Mission-specific detectors that turn images into utility maps.

This file is part of the public Space Semantic Compression research demo.
It is documented at module level so researchers, supervisors, and non-specialist
readers can understand where the file fits before reading implementation details.
"""

from semantic_ai.detector_base import DetectionResult, MissionDetectorOutput
from semantic_ai.flood_detector import FloodDetector
from semantic_ai.ship_detector import ShipDetector
from semantic_ai.wildfire_detector import WildfireDetector

__all__ = [
    "DetectionResult",
    "MissionDetectorOutput",
    "FloodDetector",
    "ShipDetector",
    "WildfireDetector",
]
