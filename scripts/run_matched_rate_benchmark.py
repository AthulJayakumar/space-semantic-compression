"""Run the publication matched-byte satellite compression comparison."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.compression_service import CompressionService  # noqa: E402
from backend.services.decoder_service import DecoderService  # noqa: E402
from backend.services.encoder_service import EncoderService  # noqa: E402
from backend.services.metrics_service import MetricsService  # noqa: E402
from backend.services.semantic_service import SemanticService  # noqa: E402
from backend.services.token_transmission_service import TokenTransmissionService  # noqa: E402
from backend.services.transmission_service import SatelliteTransmissionService  # noqa: E402
from backend.services.visualization_service import VisualizationService  # noqa: E402
from evaluation.matched_rate import MatchedRateBenchmark, read_benchmark_manifest  # noqa: E402
from semantic_ai import WildfireDetector  # noqa: E402
from token_selection.learned_mode_selector import ModeConditionedSelectorConfig, ModeConditionedTokenScorer  # noqa: E402
from token_selection.utility_pruner import TokenSelectionWeights  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Matched-rate JPEG, JPEG2000, and VQ-VAE comparison.")
    parser.add_argument("--manifest", type=Path, default=Path("results/validated_hls_500/validated_scene_manifest.csv"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/checkpoints/vqvae_s16k8_mixed_regularized_finetuned.pt"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/matched_rate_hls_500"))
    parser.add_argument(
        "--detector-checkpoint",
        type=Path,
        default=Path("models/checkpoints/wildfire_utility_segmentation.pt"),
        help="Wildfire detector checkpoint. Keep the default to reproduce registered results.",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip-lpips", action="store_true")
    parser.add_argument(
        "--utility-map-mode",
        choices=("hybrid_max", "detector_only", "weighted_blend", "detector_context"),
        default="hybrid_max",
    )
    parser.add_argument("--utility-alpha", type=float, default=0.65)
    parser.add_argument("--entropy-beta", type=float, default=0.25)
    parser.add_argument("--cost-gamma", type=float, default=0.10)
    parser.add_argument("--detail-delta", type=float, default=0.0)
    parser.add_argument("--semantic-blend-weight", type=float, default=0.25)
    parser.add_argument("--context-radius", type=int, default=1)
    parser.add_argument("--learned-selector-checkpoint", type=Path, default=None)
    args = parser.parse_args()

    items = read_benchmark_manifest(args.manifest, split=args.split, limit=args.limit)
    if not items:
        raise RuntimeError(f"No '{args.split}' items found in {args.manifest}")
    service = build_service(
        args.checkpoint,
        args.device,
        args.output_dir / "artifacts",
        detector_checkpoint=args.detector_checkpoint,
    )
    learned_selector = load_learned_selector(args.learned_selector_checkpoint, service.encoder_service.device)
    runner = MatchedRateBenchmark(
        service=service,
        output_dir=args.output_dir,
        image_size=args.image_size,
        skip_lpips=args.skip_lpips,
        utility_weights=TokenSelectionWeights(
            alpha_utility=args.utility_alpha,
            beta_entropy=args.entropy_beta,
            gamma_cost=args.cost_gamma,
            delta_detail=args.detail_delta,
        ),
        utility_map_mode=args.utility_map_mode,
        semantic_blend_weight=args.semantic_blend_weight,
        context_radius=args.context_radius,
        learned_selector=learned_selector,
        selector_name=(
            args.learned_selector_checkpoint.stem
            if args.learned_selector_checkpoint is not None
            else "fixed_weighted"
        ),
    )
    metadata = runner.run(items)
    print(json.dumps(metadata, indent=2))


def build_service(
    checkpoint: Path,
    device: str,
    output_dir: Path,
    detector_checkpoint: Path = Path("models/checkpoints/wildfire_utility_segmentation.pt"),
) -> CompressionService:
    encoder = EncoderService(checkpoint, device)
    service = CompressionService(
        encoder_service=encoder,
        decoder_service=DecoderService(encoder),
        metrics_service=MetricsService(),
        semantic_service=SemanticService(),
        token_service=TokenTransmissionService(),
        transmission_service=SatelliteTransmissionService(),
        visualization_service=VisualizationService(output_dir),
        output_dir=output_dir,
    )
    service.detectors["wildfire_detection"] = WildfireDetector(
        detector_checkpoint,
        output_dir=None,
        save_visualizations=False,
    )
    return service


def load_learned_selector(
    checkpoint_path: Path | None, device: torch.device
) -> ModeConditionedTokenScorer | None:
    if checkpoint_path is None:
        return None
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Learned selector checkpoint not found: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = ModeConditionedSelectorConfig(**checkpoint["config"])
    model = ModeConditionedTokenScorer(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


if __name__ == "__main__":
    main()
