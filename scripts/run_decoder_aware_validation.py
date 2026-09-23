"""Run prespecified decoder-aware selector variants on the HLS validation split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.decoder_aware_validation import DecoderAwareMatchedRateBenchmark  # noqa: E402
from evaluation.matched_rate import read_benchmark_manifest  # noqa: E402
from scripts.run_matched_rate_benchmark import build_service  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("results/validated_hls_500/validated_scene_manifest.csv"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/checkpoints/vqvae_s16k8_pruned_semantic_finetuned.pt"))
    parser.add_argument("--detector-checkpoint", type=Path, default=Path("models/checkpoints/wildfire_utility_segmentation_retrained.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/decoder_aware_hls_validation"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip-lpips", action="store_true", help="Faster SUS-focused development screen")
    args = parser.parse_args()
    items = read_benchmark_manifest(args.manifest, split="validation", limit=args.limit)
    service = build_service(args.checkpoint, args.device, args.output_dir / "artifacts", args.detector_checkpoint)
    runner = DecoderAwareMatchedRateBenchmark(service, args.output_dir, image_size=512, skip_lpips=args.skip_lpips)
    print(json.dumps(runner.run(items), indent=2))


if __name__ == "__main__":
    main()
