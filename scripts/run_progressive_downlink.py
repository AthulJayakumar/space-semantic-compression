"""Evaluate token ordering when a simulated contact ends after a byte prefix."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.matched_rate import MatchedRateBenchmark, read_benchmark_manifest  # noqa: E402
from evaluation.progressive_downlink import evaluate_item, paired_comparisons, summarize, write_rows  # noqa: E402
from scripts.run_matched_rate_benchmark import build_service  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--detector-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="external_test")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    items = read_benchmark_manifest(args.manifest, split=args.split, limit=args.limit)
    service = build_service(args.checkpoint, args.device, args.output_dir / "artifacts", args.detector_checkpoint)
    benchmark = MatchedRateBenchmark(service, args.output_dir, image_size=args.image_size, skip_lpips=True)
    rows: list[dict[str, object]] = []
    for index, item in enumerate(items, 1):
        print(f"progressive {index}/{len(items)}: {item.sample_id}", flush=True)
        rows.extend(evaluate_item(benchmark, item))
        write_rows(args.output_dir / "progressive_rows.csv", rows)
    summary = summarize(rows)
    write_rows(args.output_dir / "progressive_event_summary.csv", summary)
    comparisons = paired_comparisons(rows)
    if comparisons:
        write_rows(args.output_dir / "progressive_event_paired_tests.csv", comparisons)
    print(f"Completed {len(items)} images and {len(summary)} event-level summary rows")


if __name__ == "__main__":
    main()
