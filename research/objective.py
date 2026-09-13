"""research.objective

Plain-English purpose: Research objective definitions and mission framing.

This file is part of the public Space Semantic Compression research demo.
It is documented at module level so researchers, supervisors, and non-specialist
readers can understand where the file fits before reading implementation details.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ObjectiveComponents:
    rate: float
    distortion: float
    energy: float
    utility: float
    objective: float


class ResearchObjective:
    """Formal objective: L = lambda1*Rate + lambda2*Distortion + lambda3*Energy - lambda4*Utility."""

    def __init__(
        self,
        lambda_rate: float = 0.25,
        lambda_distortion: float = 0.25,
        lambda_energy: float = 0.20,
        lambda_utility: float = 0.30,
    ) -> None:
        self.lambda_rate = lambda_rate
        self.lambda_distortion = lambda_distortion
        self.lambda_energy = lambda_energy
        self.lambda_utility = lambda_utility

    def evaluate(
        self,
        rate: float,
        distortion: float,
        energy: float,
        utility: float,
    ) -> ObjectiveComponents:
        objective = (
            self.lambda_rate * rate
            + self.lambda_distortion * distortion
            + self.lambda_energy * energy
            - self.lambda_utility * utility
        )
        return ObjectiveComponents(
            rate=round(rate, 6),
            distortion=round(distortion, 6),
            energy=round(energy, 6),
            utility=round(utility, 6),
            objective=round(objective, 6),
        )
