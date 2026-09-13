"""scripts.run_retention_sweep

Plain-English purpose: Command-line runners for datasets, benchmarks, reports, and exports.

This file is part of the public Space Semantic Compression research demo.
It is documented at module level so researchers, supervisors, and non-specialist
readers can understand where the file fits before reading implementation details.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.schemas.response_schema import TransmissionConfig
from backend.services.factory import get_compression_service


def main() -> None:
    parser = argparse.ArgumentParser(description="Run utility-aware token retention sweep.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--mission", default="wildfire_detection")
    args = parser.parse_args()

    image_path = Path(args.image)
    service = get_compression_service()
    out_dir = Path("results")
    plots_dir = out_dir / "retention_sweep_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for keep in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        result = service.compress_image(
            image_path.read_bytes(),
            image_path.name,
            TransmissionConfig(semantic_keep_ratio=keep),
            mission=args.mission,
        )
        rows.append(
            {
                "keep_ratio": keep,
                "psnr": result.psnr,
                "ssim": result.ssim,
                "lpips": result.lpips,
                "sus": result.semantic_utility_score,
                "bandwidth_saved_percent": result.bandwidth_saved_percent,
                "compression_ratio": result.compression_ratio,
                "objective_value": result.objective_value,
                "energy_j": result.total_energy_j,
            }
        )
    csv_path = out_dir / "retention_sweep.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    _plot(rows, plots_dir)
    print(f"csv_path={csv_path}")
    print(f"plots_dir={plots_dir}")


def _plot(rows: list[dict[str, float]], plots_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    x = [row["keep_ratio"] for row in rows]
    for metric in ["psnr", "ssim", "lpips", "sus", "bandwidth_saved_percent", "compression_ratio", "objective_value"]:
        y = [row[metric] for row in rows]
        plt.figure(figsize=(6.0, 4.0), dpi=180)
        plt.plot(x, y, marker="o", linewidth=2)
        plt.xlabel("Token retention ratio")
        plt.ylabel(metric)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / f"{metric}.png")
        plt.close()


if __name__ == "__main__":
    main()
