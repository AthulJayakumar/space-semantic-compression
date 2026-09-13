"""token_selection.__init__

Plain-English purpose: Algorithms that decide which learned image tokens are worth transmitting.

This file is part of the public Space Semantic Compression research demo.
It is documented at module level so researchers, supervisors, and non-specialist
readers can understand where the file fits before reading implementation details.
"""

from token_selection.learned_mode_selector import ModeConditionedSelectorConfig, ModeConditionedTokenScorer
from token_selection.utility_pruner import TokenSelectionWeights, UtilityAwareTokenPruner

__all__ = [
    "ModeConditionedSelectorConfig",
    "ModeConditionedTokenScorer",
    "TokenSelectionWeights",
    "UtilityAwareTokenPruner",
]
