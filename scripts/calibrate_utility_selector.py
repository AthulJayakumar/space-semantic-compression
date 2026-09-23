"""Tune utility-map composition and token-ranking weights on validation data only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.utils.tensor_utils import image_to_tensor, tensor_to_image  # noqa: E402
from datasets.research_wildfire import load_rgb_image  # noqa: E402
from evaluation.matched_rate import (  # noqa: E402
    EncodedCandidate,
    MatchedRateBenchmark,
    bootstrap_mean_ci,
    read_benchmark_manifest,
)
from evaluation.statistics import StatisticalValidator  # noqa: E402
from scripts.run_matched_rate_benchmark import build_service  # noqa: E402
from token_selection.utility_pruner import TokenSelectionWeights, UtilityAwareTokenPruner  # noqa: E402


@dataclass(frozen=True)
class SelectorConfig:
    name: str
    map_mode: str
    utility_weight: float
    entropy_weight: float
    context_radius: int = 0
    semantic_blend_weight: float = 0.25


CONFIGS = (
    SelectorConfig("random", "random", 0.0, 0.0),
    SelectorConfig("entropy_only", "detector_only", 0.0, 1.0),
    SelectorConfig("current_hybrid_u65_e25", "hybrid_max", 0.65, 0.25),
    SelectorConfig("detector_only_u80_e20", "detector_only", 0.80, 0.20),
    SelectorConfig("detector_only_u60_e40", "detector_only", 0.60, 0.40),
    SelectorConfig("detector_context_r1_u80_e20", "detector_context", 0.80, 0.20, context_radius=1),
    SelectorConfig("detector_context_r2_u80_e20", "detector_context", 0.80, 0.20, context_radius=2),
    SelectorConfig("detector_context_r1_u60_e40", "detector_context", 0.60, 0.40, context_radius=1),
    SelectorConfig("weighted_blend_25_u80_e20", "weighted_blend", 0.80, 0.20, semantic_blend_weight=0.25),
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate the semantic token selector on held-out validation scenes.")
    parser.add_argument("--manifest", type=Path, default=Path("results/validated_hls_500/validated_scene_manifest.csv"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/checkpoints/vqvae_s16k8_mixed_regularized_finetuned.pt"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/utility_selector_calibration"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    items = read_benchmark_manifest(args.manifest, split="validation", limit=args.limit)
    if not items:
        raise RuntimeError("No validation items were found; selector tuning must not use the test partition.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    service = build_service(args.checkpoint, args.device, args.output_dir / "artifacts")
    runner = MatchedRateBenchmark(
        service=service,
        output_dir=args.output_dir,
        image_size=args.image_size,
        budget_levels=(0.0,),
        skip_lpips=True,
    )

    rows: list[dict[str, object]] = []
    exclusions: list[dict[str, str]] = []
    for index, item in enumerate(items, start=1):
        print(f"selector calibration {index}/{len(items)}: {item.sample_id}", flush=True)
        try:
            rows.extend(evaluate_item(runner, item))
        except Exception as exc:
            exclusions.append({"sample_id": item.sample_id, "reason": str(exc)})
        write_csv(args.output_dir / "selector_calibration_rows.csv", rows)
        write_csv(args.output_dir / "excluded_samples.csv", exclusions)

    summary = summarize(rows)
    comparisons = paired_comparisons(rows)
    eligible = [row for row in summary if row["selector"] not in {"random", "entropy_only"}]
    best = max(eligible, key=lambda row: (float(row["sus_mean"]), float(row["ssim_mean"])))
    selected = next(config for config in CONFIGS if config.name == best["selector"])
    selection = {
        "selection_partition": "validation",
        "selection_objective": "maximum mean SUS with mean SSIM as deterministic tie-break",
        "validation_items": len({str(row["sample_id"]) for row in rows}),
        "excluded_items": len(exclusions),
        "selected_config": selected.__dict__,
        "validation_metrics": best,
    }
    write_csv(args.output_dir / "selector_calibration_summary.csv", summary)
    write_csv(args.output_dir / "selector_calibration_statistics.csv", comparisons)
    (args.output_dir / "selected_selector.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    (args.output_dir / "selector_calibration_report.md").write_text(
        build_report(summary, comparisons, selection), encoding="utf-8"
    )
    print(json.dumps(selection, indent=2))


def evaluate_item(runner: MatchedRateBenchmark, item) -> list[dict[str, object]]:
    original = runner._prepare_image(load_rgb_image(item.image_path))
    tensor = image_to_tensor(original, runner.service.encoder_service.device, runner.service.encoder_service.stride)
    with torch.inference_mode():
        tokens = runner.service.encoder_service.encode(tensor)
    token_shape = tuple(tokens.shape[-2:])
    before = runner.service._detect_mission_utility(original, token_shape, "wildfire_detection")
    semantic = runner.service.semantic_service.analyze(original, token_shape)
    entropy_weights = TokenSelectionWeights(alpha_utility=0.0, beta_entropy=1.0, gamma_cost=0.0, delta_detail=0.0)
    entropy = UtilityAwareTokenPruner(entropy_weights).score_tokens(tokens, np.zeros(token_shape, dtype="float32"))
    rankings: dict[str, np.ndarray] = {}

    seed_bytes = hashlib.sha256(f"{runner.seed}:{item.sample_id}".encode("utf-8")).digest()[:8]
    rankings["random"] = np.random.default_rng(int.from_bytes(seed_bytes, "little")).permutation(tokens.numel())
    for config in CONFIGS[1:]:
        utility = utility_map_for_config(runner, before.utility_map, semantic.importance_map, config)
        score = runner._normalize_map(config.utility_weight * utility + config.entropy_weight * entropy)
        rankings[config.name] = np.argsort(-score.reshape(-1), kind="stable")

    cache: dict[tuple[str, int], EncodedCandidate] = {}

    def token_candidate(selector: str, keep_count: int) -> EncodedCandidate:
        key = (selector, int(keep_count))
        if key in cache:
            return cache[key]
        keep_count = int(np.clip(keep_count, 1, tokens.numel()))
        mask = np.zeros(tokens.numel(), dtype=bool)
        mask[rankings[selector][:keep_count]] = True
        mask = mask.reshape(token_shape)
        pruned = runner.service.token_service.prune_tokens(tokens, mask)
        payload = runner.service.token_service.serialize_payload(pruned, mask)
        candidate = EncodedCandidate(payload, None, keep_count / tokens.numel(), keep_count)
        cache[key] = candidate
        return candidate

    jpeg_min = runner._jpeg(original, 1)
    jpeg_max = runner._jpeg(original, 95)
    j2k_min = runner._jpeg2000(original, 512)
    j2k_max = runner._jpeg2000(original, 1)
    token_min = [token_candidate(config.name, 1).size for config in CONFIGS]
    token_max = [token_candidate(config.name, tokens.numel()).size for config in CONFIGS]
    common_low = max(jpeg_min.size, j2k_min.size, *token_min)
    common_high = min(jpeg_max.size, j2k_max.size, *token_max)
    if common_high < common_low:
        raise RuntimeError(f"No shared byte interval: {common_low}>{common_high}")

    output: list[dict[str, object]] = []
    for config in CONFIGS:
        candidate = runner._calibrate(
            common_low,
            1,
            tokens.numel(),
            lambda count, name=config.name: token_candidate(name, count),
            increasing=True,
        )
        mask = np.zeros(tokens.numel(), dtype=bool)
        mask[rankings[config.name][: int(candidate.keep_count or 1)]] = True
        mask = mask.reshape(token_shape)
        pruned = runner.service.token_service.prune_tokens(tokens, mask)
        with torch.inference_mode():
            reconstruction = tensor_to_image(runner.service.decoder_service.decode(pruned))
        measured = runner._measure(
            item=item,
            original=original,
            before=before,
            token_shape=token_shape,
            method=config.name,
            budget_index=0,
            budget_level=0.0,
            target_bytes=common_low,
            common_low=common_low,
            common_high=common_high,
            candidate=EncodedCandidate(candidate.payload, reconstruction, candidate.parameter, candidate.keep_count),
        )
        measured["selector"] = config.name
        measured["utility_map_mode"] = config.map_mode
        measured["utility_weight"] = config.utility_weight
        measured["entropy_weight"] = config.entropy_weight
        measured["context_radius"] = config.context_radius
        output.append(measured)
    return output


def utility_map_for_config(
    runner: MatchedRateBenchmark,
    detector_map: np.ndarray,
    semantic_map: np.ndarray,
    config: SelectorConfig,
) -> np.ndarray:
    if config.map_mode == "detector_only":
        return runner._normalize_map(detector_map)
    if config.map_mode == "hybrid_max":
        return np.maximum(runner._normalize_map(detector_map), runner._normalize_map(semantic_map))
    if config.map_mode == "weighted_blend":
        detector = runner._normalize_map(detector_map)
        semantic = runner._normalize_map(semantic_map)
        weight = config.semantic_blend_weight
        return runner._normalize_map((1.0 - weight) * detector + weight * semantic)
    if config.map_mode == "detector_context":
        return runner._max_filter(runner._normalize_map(detector_map), config.context_radius)
    raise ValueError(f"Unsupported map mode: {config.map_mode}")


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for selector in sorted({str(row["selector"]) for row in rows}):
        subset = [row for row in rows if row["selector"] == selector]
        record: dict[str, object] = {"selector": selector, "n": len(subset)}
        for metric in ("sus", "detector_retention", "psnr", "ssim", "actual_bytes", "kept_tokens"):
            values = np.asarray([float(row[metric]) for row in subset], dtype="float64")
            low, high = bootstrap_mean_ci(values, seed=20260921)
            record[f"{metric}_mean"] = float(values.mean())
            record[f"{metric}_std"] = float(values.std(ddof=1)) if values.size > 1 else 0.0
            record[f"{metric}_ci_low"] = low
            record[f"{metric}_ci_high"] = high
        output.append(record)
    return output


def paired_comparisons(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    validator = StatisticalValidator()
    by_selector = {
        selector: {str(row["sample_id"]): row for row in rows if row["selector"] == selector}
        for selector in {str(row["selector"]) for row in rows}
    }
    output: list[dict[str, object]] = []
    for candidate in [config.name for config in CONFIGS if config.name not in {"random", "entropy_only"}]:
        for baseline in ("current_hybrid_u65_e25", "random", "entropy_only"):
            if candidate == baseline:
                continue
            ids = sorted(set(by_selector[candidate]) & set(by_selector[baseline]))
            baseline_values = [float(by_selector[baseline][sample_id]["sus"]) for sample_id in ids]
            candidate_values = [float(by_selector[candidate][sample_id]["sus"]) for sample_id in ids]
            t_result = validator.paired_t_test(baseline_values, candidate_values)
            w_result = validator.wilcoxon(baseline_values, candidate_values)
            differences = np.asarray(candidate_values) - np.asarray(baseline_values)
            low, high = bootstrap_mean_ci(differences, seed=20260921)
            output.append(
                {
                    "candidate": candidate,
                    "baseline": baseline,
                    "n": len(ids),
                    "mean_paired_sus_difference": float(differences.mean()),
                    "difference_ci_low": low,
                    "difference_ci_high": high,
                    "paired_t_p_value": t_result.p_value,
                    "wilcoxon_p_value": w_result.p_value,
                    "cohens_dz": t_result.effect_size,
                }
            )
    return output


def build_report(
    summary: list[dict[str, object]], comparisons: list[dict[str, object]], selection: dict[str, object]
) -> str:
    best = selection["validation_metrics"]
    config = selection["selected_config"]
    lines = [
        "# Utility Selector Calibration",
        "",
        "The selector was calibrated only on the geography-separated validation partition. The test partition was not inspected during model selection.",
        "",
        f"Selected configuration: `{config['name']}` with validation SUS {float(best['sus_mean']):.2f} "
        f"(95% CI {float(best['sus_ci_low']):.2f}-{float(best['sus_ci_high']):.2f}).",
        "",
        "## Validation summary",
        "",
        "| selector | n | SUS | detector retention | PSNR | SSIM | bytes | tokens |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(summary, key=lambda item: float(item["sus_mean"]), reverse=True):
        lines.append(
            f"| {row['selector']} | {row['n']} | {float(row['sus_mean']):.2f} | "
            f"{float(row['detector_retention_mean']):.3f} | {float(row['psnr_mean']):.2f} | "
            f"{float(row['ssim_mean']):.3f} | {float(row['actual_bytes_mean']):.1f} | "
            f"{float(row['kept_tokens_mean']):.1f} |"
        )
    lines.extend(
        [
            "",
            "## Selection rule",
            "",
            "The primary objective was mean SUS. Mean SSIM was used only as a deterministic tie-break. Random and entropy-only selectors were references and were not eligible for promotion.",
            "",
            "Full paired tests are stored in `selector_calibration_statistics.csv`. The selected configuration must be evaluated once on the untouched test partition before any performance claim is made.",
        ]
    )
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
