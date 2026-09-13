"""transmission.__init__

Plain-English purpose: Energy and communication models for constrained satellite links.

This file is part of the public Space Semantic Compression research demo.
It is documented at module level so researchers, supervisors, and non-specialist
readers can understand where the file fits before reading implementation details.
"""

from transmission.energy_model import EnergyEstimate, EnergyModel

__all__ = ["EnergyEstimate", "EnergyModel"]
