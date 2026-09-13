"""datasets.__init__

Plain-English purpose: Dataset loader code only; raw imagery is intentionally not committed.

This file is part of the public Space Semantic Compression research demo.
It is documented at module level so researchers, supervisors, and non-specialist
readers can understand where the file fits before reading implementation details.
"""

from datasets.wildfire_datasets import (
    DFireDataset,
    DatasetItem,
    FirmsWildfireDataset,
    FLAMEDataset,
    ModisActiveFireDataset,
    Sentinel2WildfireDataset,
    discover_wildfire_datasets,
)
from datasets.firms import FirmsDataset, FirmsDetection
from datasets.sentinel2 import Sentinel2Dataset, Sentinel2Item, discover_sentinel2_dataset

__all__ = [
    "DatasetItem",
    "DFireDataset",
    "FirmsDataset",
    "FirmsDetection",
    "FirmsWildfireDataset",
    "FLAMEDataset",
    "ModisActiveFireDataset",
    "Sentinel2Dataset",
    "Sentinel2Item",
    "Sentinel2WildfireDataset",
    "discover_wildfire_datasets",
    "discover_sentinel2_dataset",
]
