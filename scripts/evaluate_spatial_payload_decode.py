"""Compare modal and spatial VQ decoding from identical transmitted payloads."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.utils.tensor_utils import image_to_tensor, tensor_to_image  # noqa: E402
from datasets.research_wildfire import load_rgb_image  # noqa: E402
from evaluation.decoder_aware_decision import paired_group_comparison  # noqa: E402
from evaluation.matched_rate import EncodedCandidate, MatchedRateBenchmark, read_benchmark_manifest  # noqa: E402
from scripts.run_matched_rate_benchmark import build_service  # noqa: E402
from token_selection.spatial_reconstruction import decode_nearest_filled_tokens, decode_spatially_filled_tokens  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("results/token_sus_selector/internal_geographic_split.csv"))
    parser.add_argument("--split", default="selector_validation")
    parser.add_argument("--reference-rows", type=Path, default=Path("results/task_aligned_internal_256/matched_rate_rows.csv"))
    parser.add_argument("--vqvae-checkpoint", type=Path, default=Path("models/checkpoints/vqvae_s16k8_pruned_semantic_finetuned.pt"))
    parser.add_argument("--detector-checkpoint", type=Path, default=Path("models/checkpoints/wildfire_utility_segmentation_retrained.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/spatial_payload_internal_256"))
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip-lpips", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    items = read_benchmark_manifest(args.manifest, split=args.split, limit=args.limit)
    with args.reference_rows.open(newline="", encoding="utf-8") as handle:
        reference_rows = list(csv.DictReader(handle))
    reference = {
        (row["sample_id"], int(row["budget_index"]), row["method"]): row
        for row in reference_rows
    }
    service = build_service(args.vqvae_checkpoint, "auto", args.output_dir / "artifacts", args.detector_checkpoint)
    runner = MatchedRateBenchmark(service, args.output_dir, image_size=args.image_size, skip_lpips=args.skip_lpips)
    rows: list[dict[str, object]] = []
    for index, item in enumerate(items, 1):
        original = runner._prepare_image(load_rgb_image(item.image_path))
        tensor = image_to_tensor(original, service.encoder_service.device, service.encoder_service.stride)
        with torch.inference_mode():
            tokens = service.encoder_service.encode(tensor)
        token_shape = tuple(tokens.shape[-2:])
        before = service._detect_mission_utility(original, token_shape, "wildfire_detection")
        semantic = service.semantic_service.analyze(original, token_shape)
        utility = runner.compose_utility_map(before.utility_map, semantic.importance_map)
        detail = service.semantic_service.detail_map(original, token_shape)
        ranking = runner._rankings(tokens, utility, detail, item.sample_id)["vqvae_utility"]
        for budget_index in range(3):
            baseline = reference[(item.sample_id, budget_index, "vqvae_fixed_utility")]
            keep_count = int(baseline["kept_tokens"])
            mask = np.zeros(tokens.numel(), dtype=bool)
            mask[ranking[:keep_count]] = True
            mask = mask.reshape(token_shape)
            payload = service.token_service.serialize_payload(tokens, mask)
            if len(payload) != int(baseline["encoded_bytes"]):
                raise ValueError(f"Payload bytes differ from reference for {item.sample_id} budget {budget_index}")
            received, received_mask = service.token_service.deserialize_payload(payload)
            received = received.to(service.encoder_service.device)
            with torch.inference_mode():
                modal = tensor_to_image(service.decoder_service.decode(received))
                spatial = tensor_to_image(
                    decode_spatially_filled_tokens(service.encoder_service.model, received, received_mask)
                )
                nearest = tensor_to_image(
                    decode_nearest_filled_tokens(service.encoder_service.model, received, received_mask)
                )
            for method, reconstructed in (
                ("vqvae_wire_modal", modal), ("vqvae_spatial_fill", spatial), ("vqvae_nearest_fill", nearest)
            ):
                candidate = EncodedCandidate(payload, reconstructed, keep_count / tokens.numel(), keep_count)
                rows.append(runner._measure(
                    item, original, before, token_shape, method, budget_index,
                    float(baseline["budget_level"]), int(baseline["target_bytes"]),
                    int(baseline["common_budget_low"]), int(baseline["common_budget_high"]), candidate,
                ))
            for codec in ("jpeg", "jpeg2000"):
                codec_row = dict(reference[(item.sample_id, budget_index, codec)])
                codec_row["budget_index"] = budget_index
                codec_row["budget_level"] = float(codec_row["budget_level"])
                for metric in ("lpips", "label_dice", "label_iou"):
                    if codec_row.get(metric) == "":
                        codec_row[metric] = None
                rows.append(codec_row)
        runner._write_csv(args.output_dir / "spatial_payload_rows.csv", rows)
        print(f"payload-decode {index}/{len(items)}: {item.sample_id}", flush=True)

    summary = runner.summarize(rows)
    runner._write_csv(args.output_dir / "spatial_payload_summary.csv", summary)
    comparisons = [
        paired_group_comparison(rows, candidate, control, metric, budget)
        for budget in (0, 1)
        for candidate in ("vqvae_spatial_fill", "vqvae_nearest_fill")
        for control in ("vqvae_wire_modal", "jpeg2000")
        for metric in ("sus", "detector_retention", "label_dice", "psnr", "ssim")
    ]
    runner._write_csv(args.output_dir / "spatial_payload_comparisons.csv", comparisons)
    metadata = {
        "images": len(items),
        "geographic_groups": len({item.geographic_group for item in items}),
        "wire_payload_roundtrip": True,
        "transmitted_bytes_identical_between_vq_decoders": True,
        "used_original_untransmitted_codes_for_reconstruction": False,
        "decoder_side_iterations": 4,
        "nearest_code_fill": True,
        "comparison_partition": args.split,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
