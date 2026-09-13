"""backend.services.transmission_service

Plain-English purpose: FastAPI web service and application orchestration.

This file is part of the public Space Semantic Compression research demo.
It is documented at module level so researchers, supervisors, and non-specialist
readers can understand where the file fits before reading implementation details.
"""

from __future__ import annotations

import math

from backend.schemas.response_schema import TransmissionConfig, TransmissionStats


class SatelliteTransmissionService:
    def simulate(
        self,
        full_payload_kb: float,
        semantic_payload_kb: float,
        token_entropy_bits: float,
        semantic_fidelity_percent: float,
        config: TransmissionConfig,
    ) -> TransmissionStats:
        packet_size_kb = config.packet_size_bytes / 1024.0
        packets_total = int(math.ceil(semantic_payload_kb / packet_size_kb)) if semantic_payload_kb else 0
        delivery_rate = (1.0 - config.packet_loss_percent / 100.0) * (1.0 - config.outage_probability)
        packets_delivered = int(round(packets_total * max(0.0, min(1.0, delivery_rate))))

        semantic_time = self._downlink_time_seconds(semantic_payload_kb, config.bandwidth_kbps, config.latency_ms)
        baseline_time = self._downlink_time_seconds(full_payload_kb, config.bandwidth_kbps, config.latency_ms)

        return TransmissionStats(
            full_payload_kb=round(full_payload_kb, 4),
            semantic_payload_kb=round(semantic_payload_kb, 4),
            packets_total=packets_total,
            packets_delivered=packets_delivered,
            estimated_downlink_time_sec=round(semantic_time, 4),
            baseline_downlink_time_sec=round(baseline_time, 4),
            transmission_time_saved_sec=round(baseline_time - semantic_time, 4),
            semantic_fidelity_percent=round(semantic_fidelity_percent, 2),
            token_entropy_bits=round(token_entropy_bits, 4),
        )

    def _downlink_time_seconds(self, payload_kb: float, bandwidth_kbps: float, latency_ms: float) -> float:
        payload_kbits = payload_kb * 8.0
        return payload_kbits / bandwidth_kbps + latency_ms / 1000.0
