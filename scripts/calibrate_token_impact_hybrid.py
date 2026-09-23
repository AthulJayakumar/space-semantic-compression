"""Calibrate learned, fixed-utility, and entropy score hybrids on validation data."""

from __future__ import annotations

import argparse
import csv
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
from scripts.run_matched_rate_benchmark import build_service, load_learned_selector  # noqa: E402
from token_selection.learned_mode_selector import local_token_entropy_numpy  # noqa: E402
from token_selection.utility_pruner import TokenSelectionWeights, UtilityAwareTokenPruner  # noqa: E402


@dataclass(frozen=True)
class HybridSpec:
    name: str
    fixed_weight: float
    entropy_weight: float
    learned_weight: float


HYBRIDS = (
    HybridSpec("hybrid_fixed75_learned25", 0.75, 0.0, 0.25),
    HybridSpec("hybrid_fixed50_learned50", 0.50, 0.0, 0.50),
    HybridSpec("hybrid_entropy75_learned25", 0.0, 0.75, 0.25),
    HybridSpec("hybrid_entropy50_learned50", 0.0, 0.50, 0.50),
    HybridSpec("hybrid_fixed25_entropy25_learned50", 0.25, 0.25, 0.50),
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate decoder-impact selector score hybrids.")
    parser.add_argument("--manifest", type=Path, default=Path("results/validated_hls_500/validated_scene_manifest.csv"))
    parser.add_argument(
        "--vqvae-checkpoint",
        type=Path,
        default=Path("models/checkpoints/vqvae_s16k8_mixed_regularized_finetuned.pt"),
    )
    parser.add_argument(
        "--selector-checkpoint",
        type=Path,
        default=Path("models/checkpoints/token_impact_selector_hls.pt"),
    )
    parser.add_argument(
        "--reference-results",
        type=Path,
        default=Path("results/token_impact_selector_validation"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/token_impact_hybrid_calibration"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference_rows = [
        normalize_reference_row(row)
        for row in read_csv(args.reference_results / "matched_rate_rows.csv")
        if int(row["budget_index"]) in {0, 1}
    ]
    targets = {
        (row["sample_id"], int(row["budget_index"])): row
        for row in reference_rows
        if row["method"] == "jpeg"
    }
    service = build_service(args.vqvae_checkpoint, args.device, args.output_dir / "artifacts")
    learned = load_learned_selector(args.selector_checkpoint, service.encoder_service.device)
    if learned is None:
        raise RuntimeError("A learned selector checkpoint is required.")
    runner = MatchedRateBenchmark(service, args.output_dir, skip_lpips=True)
    items = read_benchmark_manifest(args.manifest, split="validation", limit=args.limit)
    hybrid_rows: list[dict[str, object]] = []
    exclusions: list[dict[str, str]] = []
    for index, item in enumerate(items, start=1):
        print(f"hybrid calibration {index}/{len(items)}: {item.sample_id}", flush=True)
        try:
            hybrid_rows.extend(evaluate_item(runner, learned, item, targets))
        except Exception as exc:
            exclusions.append({"sample_id": item.sample_id, "reason": str(exc)})
        write_csv(args.output_dir / "hybrid_rows.csv", hybrid_rows)
        write_csv(args.output_dir / "excluded_samples.csv", exclusions)

    combined = reference_rows + [{key: value for key, value in row.items()} for row in hybrid_rows]
    summary = runner.summarize(combined)
    candidate_scores = rank_candidates(summary)
    selected = max(candidate_scores, key=lambda row: float(row["mean_low_middle_sus"]))
    comparisons = paired_average_rate_tests(combined, str(selected["selector"]))
    entropy_test = next(row for row in comparisons if row["baseline"] == "vqvae_entropy")
    fixed_test = next(row for row in comparisons if row["baseline"] == "vqvae_fixed_utility")
    go = float(entropy_test["difference_ci_low"]) > 0.0 and float(fixed_test["difference_ci_low"]) > 0.0
    decision = {
        "decision": "go_for_test_evaluation" if go else "no_go_for_test_evaluation",
        "selection_partition": "geographic validation",
        "test_partition_used": False,
        "validation_items": len(items) - len(exclusions),
        "selected_hybrid": selected,
        "go_rule": "paired average low/middle SUS bootstrap CI must be above zero versus fixed and entropy selectors",
        "comparison_vs_entropy": entropy_test,
        "comparison_vs_fixed": fixed_test,
    }
    write_csv(args.output_dir / "hybrid_summary.csv", summary)
    write_csv(args.output_dir / "candidate_scores.csv", candidate_scores)
    write_csv(args.output_dir / "selected_hybrid_statistics.csv", comparisons)
    (args.output_dir / "hybrid_selection.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    (args.output_dir / "hybrid_calibration_report.md").write_text(
        build_report(summary, candidate_scores, comparisons, decision), encoding="utf-8"
    )
    print(json.dumps(decision, indent=2))


def evaluate_item(runner, learned, item, targets) -> list[dict[str, object]]:
    original = runner._prepare_image(load_rgb_image(item.image_path))
    tensor = image_to_tensor(original, runner.service.encoder_service.device, runner.service.encoder_service.stride)
    with torch.inference_mode():
        tokens = runner.service.encoder_service.encode(tensor)
    token_shape = tuple(tokens.shape[-2:])
    before = runner.service._detect_mission_utility(original, token_shape, "wildfire_detection")
    utility = runner._normalize_map(before.utility_map)
    detail = runner.service.semantic_service.detail_map(original, token_shape)
    entropy = local_token_entropy_numpy(tokens, token_shape)
    fixed = UtilityAwareTokenPruner(TokenSelectionWeights.mission_utility()).score_tokens(tokens, utility)
    device = next(learned.parameters()).device
    with torch.inference_mode():
        learned_scores = (
            learned.score(
                tokens.to(device),
                torch.from_numpy(utility).to(device),
                torch.from_numpy(entropy).to(device),
                torch.from_numpy(detail).to(device),
                "mission_utility",
            )[0]
            .detach()
            .cpu()
            .numpy()
        )
    base_scores = {
        "fixed": runner._normalize_map(fixed),
        "entropy": runner._normalize_map(entropy),
        "learned": runner._normalize_map(learned_scores),
    }
    rankings = {
        spec.name: np.argsort(
            -runner._normalize_map(
                spec.fixed_weight * base_scores["fixed"]
                + spec.entropy_weight * base_scores["entropy"]
                + spec.learned_weight * base_scores["learned"]
            ).reshape(-1),
            kind="stable",
        )
        for spec in HYBRIDS
    }
    cache: dict[tuple[str, int], EncodedCandidate] = {}

    def candidate(selector: str, keep_count: int) -> EncodedCandidate:
        key = (selector, int(keep_count))
        if key in cache:
            return cache[key]
        keep_count = int(np.clip(keep_count, 1, tokens.numel()))
        mask = np.zeros(tokens.numel(), dtype=bool)
        mask[rankings[selector][:keep_count]] = True
        mask = mask.reshape(token_shape)
        pruned = runner.service.token_service.prune_tokens(tokens, mask)
        encoded = EncodedCandidate(
            runner.service.token_service.serialize_payload(pruned, mask),
            None,
            keep_count / tokens.numel(),
            keep_count,
        )
        cache[key] = encoded
        return encoded

    rows: list[dict[str, object]] = []
    for budget_index in (0, 1):
        target_row = targets[(item.sample_id, budget_index)]
        target_bytes = int(float(target_row["target_bytes"]))
        for spec in HYBRIDS:
            encoded = runner._calibrate(
                target_bytes,
                1,
                tokens.numel(),
                lambda count, name=spec.name: candidate(name, count),
                increasing=True,
            )
            mask = np.zeros(tokens.numel(), dtype=bool)
            mask[rankings[spec.name][: int(encoded.keep_count or 1)]] = True
            mask = mask.reshape(token_shape)
            pruned = runner.service.token_service.prune_tokens(tokens, mask)
            with torch.inference_mode():
                reconstruction = tensor_to_image(runner.service.decoder_service.decode(pruned))
            measured = runner._measure(
                item=item,
                original=original,
                before=before,
                token_shape=token_shape,
                method=spec.name,
                budget_index=budget_index,
                budget_level=float(target_row["budget_level"]),
                target_bytes=target_bytes,
                common_low=int(float(target_row["common_budget_low"])),
                common_high=int(float(target_row["common_budget_high"])),
                candidate=EncodedCandidate(encoded.payload, reconstruction, encoded.parameter, encoded.keep_count),
            )
            measured.update(
                {
                    "fixed_score_weight": spec.fixed_weight,
                    "entropy_score_weight": spec.entropy_weight,
                    "learned_score_weight": spec.learned_weight,
                }
            )
            rows.append(measured)
    return rows


def rank_candidates(summary) -> list[dict[str, object]]:
    output = []
    for spec in HYBRIDS:
        low = next(row for row in summary if row["method"] == spec.name and int(row["budget_index"]) == 0)
        middle = next(row for row in summary if row["method"] == spec.name and int(row["budget_index"]) == 1)
        output.append(
            {
                "selector": spec.name,
                "low_sus": float(low["sus_mean"]),
                "middle_sus": float(middle["sus_mean"]),
                "mean_low_middle_sus": (float(low["sus_mean"]) + float(middle["sus_mean"])) / 2.0,
                "low_detector_retention": float(low["detector_retention_mean"]),
                "middle_detector_retention": float(middle["detector_retention_mean"]),
                "low_psnr": float(low["psnr_mean"]),
                "middle_psnr": float(middle["psnr_mean"]),
            }
        )
    return output


def paired_average_rate_tests(rows, candidate: str) -> list[dict[str, object]]:
    validator = StatisticalValidator()
    methods = ("vqvae_fixed_utility", "vqvae_entropy", "vqvae_random", "vqvae_utility")
    values: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        if int(row["budget_index"]) not in {0, 1}:
            continue
        values.setdefault(str(row["method"]), {}).setdefault(str(row["sample_id"]), []).append(float(row["sus"]))
    output = []
    candidate_values = {sample_id: float(np.mean(items)) for sample_id, items in values[candidate].items()}
    for baseline in methods:
        baseline_values = {sample_id: float(np.mean(items)) for sample_id, items in values[baseline].items()}
        ids = sorted(set(candidate_values) & set(baseline_values))
        base = [baseline_values[sample_id] for sample_id in ids]
        selected = [candidate_values[sample_id] for sample_id in ids]
        differences = np.asarray(selected) - np.asarray(base)
        t_result = validator.paired_t_test(base, selected)
        w_result = validator.wilcoxon(base, selected)
        low, high = bootstrap_mean_ci(differences, seed=20260921)
        output.append(
            {
                "baseline": baseline,
                "candidate": candidate,
                "n": len(ids),
                "baseline_mean_low_middle_sus": float(np.mean(base)),
                "candidate_mean_low_middle_sus": float(np.mean(selected)),
                "mean_paired_difference": float(differences.mean()),
                "difference_ci_low": low,
                "difference_ci_high": high,
                "paired_t_p_value": t_result.p_value,
                "wilcoxon_p_value": w_result.p_value,
                "cohens_dz": t_result.effect_size,
            }
        )
    return output


def build_report(summary, candidate_scores, comparisons, decision) -> str:
    selected = decision["selected_hybrid"]
    lines = [
        "# Token-Impact Hybrid Calibration",
        "",
        "Five predeclared hybrids combined fixed mission utility, local entropy, and learned decoder-impact scores. Every candidate reused the exact per-image low and middle byte targets from the completed 98-scene validation benchmark.",
        "",
        "## Candidate ranking",
        "",
        "| selector | low SUS | middle SUS | mean SUS | low PSNR | middle PSNR |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(candidate_scores, key=lambda item: float(item["mean_low_middle_sus"]), reverse=True):
        lines.append(
            f"| {row['selector']} | {float(row['low_sus']):.2f} | {float(row['middle_sus']):.2f} | "
            f"{float(row['mean_low_middle_sus']):.2f} | {float(row['low_psnr']):.2f} | {float(row['middle_psnr']):.2f} |"
        )
    lines.extend(["", f"Selected hybrid: `{selected['selector']}`.", "", "## Paired average-rate tests", ""])
    for row in comparisons:
        lines.append(
            f"- Versus {row['baseline']}: {float(row['mean_paired_difference']):+.2f} SUS, "
            f"95% CI {float(row['difference_ci_low']):+.2f} to {float(row['difference_ci_high']):+.2f}, "
            f"paired t-test p={float(row['paired_t_p_value']):.4g}."
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"**{str(decision['decision']).replace('_', ' ').title()}.** The test partition remains unused. The predeclared go rule requires a positive paired bootstrap confidence interval against both fixed utility and entropy selection before test evaluation.",
        ]
    )
    return "\n".join(lines)


def normalize_reference_row(row: dict[str, str]) -> dict[str, object]:
    """Match CSV key types to rows measured in the current process."""
    normalized: dict[str, object] = {
        key: (None if isinstance(value, str) and not value.strip() else value)
        for key, value in row.items()
    }
    normalized["budget_index"] = int(row["budget_index"])
    return normalized


def read_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows) -> None:
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
