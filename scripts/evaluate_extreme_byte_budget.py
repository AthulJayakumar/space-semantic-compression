"""Compare wire-decoded VQ with resolution-adaptive classical codecs."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.utils.tensor_utils import image_to_tensor, tensor_to_image  # noqa: E402
from datasets.research_wildfire import load_rgb_image  # noqa: E402
from evaluation.decoder_aware_decision import paired_group_comparison  # noqa: E402
from evaluation.matched_rate import EncodedCandidate, MatchedRateBenchmark, read_benchmark_manifest  # noqa: E402
from scripts.run_matched_rate_benchmark import build_service  # noqa: E402


SIDES = (64, 96, 128, 192, 256, 384, 512)
JPEG_QUALITIES = (1, 10, 20, 35, 50, 65, 80, 95)
J2K_RATES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)


def classical_candidates(original: Image.Image, runner: MatchedRateBenchmark, maximum_bytes: int):
    """Choose distortion-optimal settings using only the original encoder input."""
    output: dict[str, list[tuple[EncodedCandidate, int, float]]] = {"jpeg_rdo": [], "jpeg2000_rdo": []}
    seen: dict[str, set[str]] = {method: set() for method in output}
    for side in sorted({*SIDES, min(original.size)}):
        if side > min(original.size):
            continue
        reduced = original.resize((side, side), Image.Resampling.LANCZOS)
        for method, parameters in (("jpeg_rdo", JPEG_QUALITIES), ("jpeg2000_rdo", J2K_RATES)):
            for parameter in parameters:
                buffer = io.BytesIO()
                if method == "jpeg_rdo":
                    reduced.save(buffer, format="JPEG", quality=parameter, optimize=True)
                else:
                    reduced.save(buffer, format="JPEG2000", quality_mode="rates", quality_layers=[float(parameter)])
                payload = buffer.getvalue()
                if len(payload) > maximum_bytes:
                    continue
                digest = hashlib.sha256(payload).hexdigest()
                if digest in seen[method]:
                    continue
                seen[method].add(digest)
                decoded = Image.open(io.BytesIO(payload)).convert("RGB").resize(original.size, Image.Resampling.LANCZOS)
                candidate = EncodedCandidate(payload, decoded, float(parameter))
                output[method].append((candidate, side, runner.service.metrics_service.psnr(original, decoded)))
    return output


def choose_rdo(candidates: list[tuple[EncodedCandidate, int, float]], target_bytes: int):
    eligible = [item for item in candidates if item[0].size <= target_bytes]
    if not eligible:
        raise RuntimeError(f"Classical codec has no feasible setting at {target_bytes} bytes")
    return max(eligible, key=lambda item: (item[2], -item[0].size))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("results/token_sus_selector/internal_geographic_split.csv"))
    parser.add_argument("--split", default="selector_validation")
    parser.add_argument("--checkpoint", type=Path, default=Path("models/checkpoints/vqvae_s16k8_pruned_semantic_finetuned.pt"))
    parser.add_argument("--detector-checkpoint", type=Path, default=Path("models/checkpoints/wildfire_utility_segmentation_retrained.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/extreme_rate_internal_512"))
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--budgets", nargs="+", type=int, default=[800, 1200, 1600, 2000, 2500])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip-lpips", action="store_true")
    args = parser.parse_args()
    if sorted(set(args.budgets)) != args.budgets or min(args.budgets) < 500:
        raise ValueError("Budgets must be unique ascending byte counts of at least 500")
    items = read_benchmark_manifest(args.manifest, split=args.split, limit=args.limit)
    service = build_service(args.checkpoint, "auto", args.output_dir / "artifacts", args.detector_checkpoint)
    runner = MatchedRateBenchmark(service, args.output_dir, image_size=args.image_size, skip_lpips=args.skip_lpips)
    rows: list[dict[str, object]] = []
    for index, item in enumerate(items, 1):
        original = runner._prepare_image(load_rgb_image(item.image_path))
        tensor = image_to_tensor(original, service.encoder_service.device, service.encoder_service.stride)
        with torch.inference_mode():
            tokens = service.encoder_service.encode(tensor)
        shape = tuple(tokens.shape[-2:])
        before = service._detect_mission_utility(original, shape, "wildfire_detection")
        semantic = service.semantic_service.analyze(original, shape)
        utility = runner.compose_utility_map(before.utility_map, semantic.importance_map)
        detail = service.semantic_service.detail_map(original, shape)
        ranking = runner._rankings(tokens, utility, detail, item.sample_id)["vqvae_utility"]
        classical = classical_candidates(original, runner, max(args.budgets))

        def token_candidate(count: int) -> EncodedCandidate:
            mask = np.zeros(tokens.numel(), dtype=bool)
            mask[ranking[:count]] = True
            payload = service.token_service.serialize_payload(tokens, mask.reshape(shape))
            return EncodedCandidate(payload, None, count / tokens.numel(), count)

        for budget_index, budget in enumerate(args.budgets):
            vq = runner._calibrate(budget, 1, tokens.numel(), token_candidate, increasing=True)
            mask = np.zeros(tokens.numel(), dtype=bool)
            mask[ranking[:int(vq.keep_count)]] = True
            received, received_mask = service.token_service.deserialize_payload(vq.payload)
            if not np.array_equal(received_mask, mask.reshape(shape)):
                raise RuntimeError("Receiver mask differs from transmitted mask")
            with torch.inference_mode():
                decoded = tensor_to_image(service.decoder_service.decode(received))
            vq = EncodedCandidate(vq.payload, decoded, vq.parameter, vq.keep_count)
            choices = {"vqvae_fixed_utility": (vq, args.image_size, service.metrics_service.psnr(original, decoded))}
            choices.update({method: choose_rdo(options, budget) for method, options in classical.items()})
            for method, (candidate, side, rdo_psnr) in choices.items():
                row = runner._measure(item, original, before, shape, method, budget_index,
                                      budget / args.image_size**2, budget, args.budgets[0],
                                      args.budgets[-1], candidate)
                row["encoded_side"] = side
                row["encoder_rdo_psnr"] = rdo_psnr
                rows.append(row)
        runner._write_csv(args.output_dir / "extreme_rate_rows.csv", rows)
        print(f"extreme-rate {index}/{len(items)}: {item.sample_id}", flush=True)

    summary = runner.summarize(rows)
    runner._write_csv(args.output_dir / "extreme_rate_summary.csv", summary)
    comparisons = [
        paired_group_comparison(rows, "vqvae_fixed_utility", control, metric, budget_index)
        for budget_index in range(len(args.budgets))
        for control in ("jpeg_rdo", "jpeg2000_rdo")
        for metric in ("sus", "detector_retention", "label_dice", "psnr", "ssim")
    ]
    runner._write_csv(args.output_dir / "extreme_rate_comparisons.csv", comparisons)
    metadata = {
        "images": len(items),
        "geographic_groups": len({item.geographic_group for item in items}),
        "budget_bytes": args.budgets,
        "classical_encoder_selection": "highest original-image PSNR under byte budget across predefined scales and quality settings",
        "wire_roundtrip": True,
        "development_partition": args.split,
        "skip_lpips": args.skip_lpips,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
