"""communication.satellite_analysis

Plain-English purpose: Satellite downlink analysis and mission-scale savings estimates.

This file is part of the public Space Semantic Compression research demo.
It is documented at module level so researchers, supervisors, and non-specialist
readers can understand where the file fits before reading implementation details.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


class SatelliteCommunicationAnalyzer:
    """Estimate Sentinel-2 communication savings from compression outputs."""

    def __init__(self, downlink_mbps: float = 50.0, daily_images: int = 1000) -> None:
        self.downlink_mbps = downlink_mbps
        self.daily_images = daily_images

    def analyze_rows(self, rows: list[dict[str, object]]) -> list[dict[str, object]]:
        analyzed: list[dict[str, object]] = []
        for row in rows:
            original_mb = self._original_mb(row)
            bandwidth_saved = float(row.get("bandwidth_saved_percent") or row.get("bandwidth_saved_percent_mean") or 0.0)
            compressed_mb = original_mb * max(0.0, 1.0 - bandwidth_saved / 100.0)
            original_time = self._downlink_seconds(original_mb)
            compressed_time = self._downlink_seconds(compressed_mb)
            analyzed.append(
                {
                    **row,
                    "original_volume_mb": round(original_mb, 4),
                    "compressed_volume_mb": round(compressed_mb, 4),
                    "downlink_seconds_original": round(original_time, 4),
                    "downlink_seconds_compressed": round(compressed_time, 4),
                    "downlink_seconds_saved": round(original_time - compressed_time, 4),
                    "mission_daily_volume_saved_gb": round((original_mb - compressed_mb) * self.daily_images / 1024.0, 4),
                    "mission_daily_downlink_hours_saved": round((original_time - compressed_time) * self.daily_images / 3600.0, 4),
                }
            )
        return analyzed

    def export(self, rows: list[dict[str, object]], output_dir: str | Path) -> dict[str, str]:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        analyzed = self.analyze_rows(rows)
        csv_path = output / "satellite_communication_analysis.csv"
        json_path = output / "satellite_communication_analysis.json"
        if analyzed:
            keys = sorted({key for row in analyzed for key in row.keys()})
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=keys)
                writer.writeheader()
                writer.writerows(analyzed)
            json_path.write_text(json.dumps(analyzed, indent=2), encoding="utf-8")
        else:
            csv_path.write_text("", encoding="utf-8")
            json_path.write_text("[]", encoding="utf-8")
        return {"communication_csv": str(csv_path), "communication_json": str(json_path)}

    def _original_mb(self, row: dict[str, object]) -> float:
        value = row.get("original_size_mb") or row.get("original_volume_mb")
        if value is not None:
            return float(value)
        compression_ratio = float(row.get("compression_ratio") or row.get("compression_ratio_mean") or 1.0)
        compressed_kb = float(row.get("compressed_size_kb") or 1024.0)
        return max(compressed_kb * compression_ratio / 1024.0, 0.001)

    def _downlink_seconds(self, volume_mb: float) -> float:
        return volume_mb * 8.0 / max(self.downlink_mbps, 1e-9)
