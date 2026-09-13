"""evaluation.__init__

Plain-English purpose: Benchmark pipelines, statistics, and publication-oriented experiments.

This file is part of the public Space Semantic Compression research demo.
It is documented at module level so researchers, supervisors, and non-specialist
readers can understand where the file fits before reading implementation details.
"""

from evaluation.dataset_evaluator import DatasetEvaluator
from evaluation.statistics import StatisticalValidator

__all__ = ["DatasetEvaluator", "StatisticalValidator"]
