"""transmission.energy_model

Plain-English purpose: Energy and communication models for constrained satellite links.

This file is part of the public Space Semantic Compression research demo.
It is documented at module level so researchers, supervisors, and non-specialist
readers can understand where the file fits before reading implementation details.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EnergyEstimate:
    transmission_energy_j: float
    compute_energy_j: float
    total_energy_j: float
    energy_per_image_j: float
    energy_per_bit_j: float
    energy_saved_j: float


class EnergyModel:
    """Simple reproducible mission energy estimator."""

    def __init__(
        self,
        radio_energy_per_bit_j: float = 50e-9,
        compute_power_w: float = 12.0,
    ) -> None:
        self.radio_energy_per_bit_j = radio_energy_per_bit_j
        self.compute_power_w = compute_power_w

    def estimate(
        self,
        payload_kb: float,
        baseline_payload_kb: float,
        compute_latency_ms: float,
    ) -> EnergyEstimate:
        payload_bits = payload_kb * 1024.0 * 8.0
        baseline_bits = baseline_payload_kb * 1024.0 * 8.0
        tx_energy = payload_bits * self.radio_energy_per_bit_j
        baseline_tx_energy = baseline_bits * self.radio_energy_per_bit_j
        compute_energy = self.compute_power_w * (compute_latency_ms / 1000.0)
        total = tx_energy + compute_energy
        return EnergyEstimate(
            transmission_energy_j=round(tx_energy, 6),
            compute_energy_j=round(compute_energy, 6),
            total_energy_j=round(total, 6),
            energy_per_image_j=round(total, 6),
            energy_per_bit_j=round(total / max(payload_bits, 1.0), 12),
            energy_saved_j=round(max(0.0, baseline_tx_energy - tx_energy), 6),
        )
