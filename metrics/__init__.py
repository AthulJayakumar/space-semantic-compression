"""metrics.__init__

Plain-English purpose: Research metrics such as Semantic Utility Score.

This file is part of the public Space Semantic Compression research demo.
It is documented at module level so researchers, supervisors, and non-specialist
readers can understand where the file fits before reading implementation details.
"""

from metrics.semantic_utility import SemanticUtilityComponents, SemanticUtilityMetric

__all__ = ["SemanticUtilityComponents", "SemanticUtilityMetric"]
