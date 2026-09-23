"""Screen two prespecified coverage-aware rankings at frozen validation byte targets."""

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
from token_selection.spatial_stratification import stratified_order  # noqa: E402
from token_selection.utility_pruner import UtilityAwareTokenPruner  # noqa: E402


VARIANTS = {"vqvae_stratified4": 4, "vqvae_stratified8": 8}
CONTROLS = ("vqvae_utility", "vqvae_random", "vqvae_entropy")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("results/validated_hls_500/validated_scene_manifest.csv"))
    parser.add_argument("--reference-dir", type=Path, default=Path("results/decoder_aware_hls_validation"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/spatial_selector_hls_validation"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/checkpoints/vqvae_s16k8_pruned_semantic_finetuned.pt"))
    parser.add_argument("--detector-checkpoint", type=Path, default=Path("models/checkpoints/wildfire_utility_segmentation_retrained.pt"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    items = read_benchmark_manifest(args.manifest, split="validation", limit=args.limit)
    selected_ids = {item.sample_id for item in items}
    reference = [
        row for row in read_csv(args.reference_dir / "matched_rate_rows.csv")
        if row["budget_index"] == "0" and row["sample_id"] in selected_ids
    ]
    targets = {row["sample_id"]: row for row in reference if row["method"] == "jpeg"}
    service = build_service(args.checkpoint, args.device, args.output_dir / "artifacts", args.detector_checkpoint)
    runner = MatchedRateBenchmark(service, args.output_dir, image_size=512, skip_lpips=True)
    rows: list[dict[str, object]] = []
    for index, item in enumerate(items, 1):
        print(f"spatial screen {index}/{len(items)}: {item.sample_id}", flush=True)
        original = runner._prepare_image(load_rgb_image(item.image_path))
        tensor = image_to_tensor(original, service.encoder_service.device, service.encoder_service.stride)
        with torch.inference_mode():
            tokens = service.encoder_service.encode(tensor)
        shape = tuple(tokens.shape[-2:])
        before = service._detect_mission_utility(original, shape, "wildfire_detection")
        semantic = service.semantic_service.analyze(original, shape)
        utility = runner.compose_utility_map(before.utility_map, semantic.importance_map)
        scores = UtilityAwareTokenPruner(runner.utility_weights).score_tokens(tokens, utility)
        target = targets[item.sample_id]
        budget = int(target["target_bytes"])
        for name, block_size in VARIANTS.items():
            order = stratified_order(scores, block_size)

            def encode(count: int) -> EncodedCandidate:
                count = int(np.clip(count, 1, tokens.numel()))
                mask = np.zeros(tokens.numel(), dtype=bool)
                mask[order[:count]] = True
                mask = mask.reshape(shape)
                pruned = service.token_service.prune_tokens(tokens, mask)
                payload = service.token_service.serialize_payload(pruned, mask)
                return EncodedCandidate(payload, None, count / tokens.numel(), count)

            encoded = runner._calibrate(budget, 1, tokens.numel(), encode, increasing=True)
            mask = np.zeros(tokens.numel(), dtype=bool)
            mask[order[: int(encoded.keep_count or 1)]] = True
            mask = mask.reshape(shape)
            pruned = service.token_service.prune_tokens(tokens, mask)
            with torch.inference_mode():
                reconstructed = tensor_to_image(service.decoder_service.decode(pruned))
            rows.append(runner._measure(
                item, original, before, shape, name, 0, 0.0, budget,
                int(target["common_budget_low"]), int(target["common_budget_high"]),
                EncodedCandidate(encoded.payload, reconstructed, encoded.parameter, encoded.keep_count),
            ))
        write_csv(args.output_dir / "spatial_rows.csv", rows)

    combined = reference + rows
    comparisons = [
        paired_group_comparison(combined, candidate, control, "sus")
        for candidate in VARIANTS for control in CONTROLS
    ]
    comparisons.extend(
        paired_group_comparison(combined, candidate, "vqvae_random", "detector_retention")
        for candidate in VARIANTS
    )
    write_csv(args.output_dir / "geography_clustered_comparisons.csv", comparisons)
    eligible = []
    for candidate in VARIANTS:
        sus = [row for row in comparisons if row["candidate"] == candidate and row["metric"] == "sus"]
        detector = next(row for row in comparisons if row["candidate"] == candidate and row["metric"] == "detector_retention")
        if all(float(row["ci_low"]) > 0 for row in sus) and float(detector["ci_low"]) >= -0.02:
            eligible.append((candidate, min(float(row["ci_low"]) for row in sus)))
    selected = max(eligible, key=lambda pair: pair[1])[0] if eligible else None
    decision = {"decision": "advance_to_reserved_replication" if selected else "no_go", "selected_candidate": selected,
                "comparison_count": len(comparisons), "selection_partition": "HLS geographic validation"}
    (args.output_dir / "advancement_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
