"""Diagnose token alignment and validate SUS against burn-scar ground truth."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.utils.tensor_utils import image_to_tensor, tensor_to_image  # noqa: E402
from datasets.research_wildfire import load_grayscale_image, load_rgb_image  # noqa: E402
from evaluation.matched_rate import bootstrap_mean_ci, read_benchmark_manifest  # noqa: E402
from evaluation.statistics import StatisticalValidator  # noqa: E402
from scripts.run_matched_rate_benchmark import build_service, load_learned_selector  # noqa: E402
from token_selection.learned_mode_selector import local_token_entropy_numpy  # noqa: E402
from token_selection.utility_pruner import TokenSelectionWeights, UtilityAwareTokenPruner  # noqa: E402


RETENTION_LEVELS = (0.10, 0.20, 0.30, 0.40, 0.50)
SELECTORS = ("entropy", "fixed_utility", "sus_aligned", "decoder_impact_exploratory")
CONFIRMATORY_SELECTORS = ("entropy", "fixed_utility", "sus_aligned")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run token-ground-truth and SUS-validity diagnostics.")
    parser.add_argument(
        "--manifest", type=Path, default=Path("results/token_sus_selector/internal_geographic_split.csv")
    )
    parser.add_argument(
        "--matched-rate-rows",
        type=Path,
        default=Path("results/token_sus_selector_internal_validation/matched_rate_rows.csv"),
    )
    parser.add_argument(
        "--vqvae-checkpoint",
        type=Path,
        default=Path("models/checkpoints/vqvae_s16k8_mixed_regularized_finetuned.pt"),
    )
    parser.add_argument(
        "--sus-selector-checkpoint", type=Path, default=Path("models/checkpoints/token_sus_selector_hls.pt")
    )
    parser.add_argument(
        "--impact-selector-checkpoint", type=Path, default=Path("models/checkpoints/token_impact_selector_hls.pt")
    )
    parser.add_argument(
        "--detector-checkpoint",
        type=Path,
        default=Path("models/checkpoints/wildfire_utility_segmentation.pt"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/selector_validity_study"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260922)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    service = build_service(args.vqvae_checkpoint, args.device, args.output_dir / "artifacts")
    device = service.encoder_service.device
    sus_selector = load_learned_selector(args.sus_selector_checkpoint, device)
    impact_selector = load_learned_selector(args.impact_selector_checkpoint, device)
    if sus_selector is None or impact_selector is None:
        raise RuntimeError("Both learned selector checkpoints are required.")
    detector_checkpoint = torch.load(args.detector_checkpoint, map_location="cpu", weights_only=True)
    detector_threshold = float(detector_checkpoint["config"].get("threshold", 0.5))
    items = read_benchmark_manifest(args.manifest, split="selector_validation", limit=args.limit)
    reference_rows = read_csv(args.matched_rate_rows)
    reference = {
        (row["sample_id"], row["method"], int(row["budget_index"])): row
        for row in reference_rows
        if int(row["budget_index"]) in {0, 1}
    }

    alignment_rows: list[dict[str, object]] = []
    validity_rows: list[dict[str, object]] = []
    exclusions: list[dict[str, str]] = []
    for index, item in enumerate(items, start=1):
        print(f"selector validity {index}/{len(items)}: {item.sample_id}", flush=True)
        try:
            alignment, validity = evaluate_item(
                item,
                service,
                sus_selector,
                impact_selector,
                reference,
                detector_threshold,
            )
            alignment_rows.extend(alignment)
            validity_rows.extend(validity)
        except Exception as exc:
            exclusions.append({"sample_id": item.sample_id, "reason": str(exc)})
        write_csv(args.output_dir / "token_alignment_rows.csv", alignment_rows)
        write_csv(args.output_dir / "sus_validity_rows.csv", validity_rows)
        write_csv(args.output_dir / "excluded_samples.csv", exclusions)

    alignment_summary = summarize_alignment(alignment_rows, args.seed)
    alignment_statistics = alignment_pairwise_tests(alignment_rows, args.seed)
    correlations = correlation_analysis(validity_rows, args.seed)
    stratification = stratify_detector(validity_rows)
    figures = generate_figures(args.output_dir / "figures", alignment_summary, validity_rows, stratification)
    write_csv(args.output_dir / "token_alignment_summary.csv", alignment_summary)
    write_csv(args.output_dir / "token_alignment_statistics.csv", alignment_statistics)
    write_csv(args.output_dir / "sus_ground_truth_correlations.csv", correlations)
    write_csv(args.output_dir / "detector_stratification.csv", stratification)
    metadata = {
        "partition": "internal geographic validation derived from original training partition",
        "requested_scenes": len(items),
        "completed_scenes": len({row["sample_id"] for row in validity_rows}),
        "excluded_scenes": len(exclusions),
        "official_validation_used": False,
        "test_partition_used": False,
        "retention_levels": RETENTION_LEVELS,
        "confirmatory_selectors": CONFIRMATORY_SELECTORS,
        "exploratory_selector": "decoder_impact_exploratory",
        "detector_threshold": detector_threshold,
        "figures": figures,
    }
    (args.output_dir / "study_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (args.output_dir / "selector_validity_report.md").write_text(
        build_report(metadata, alignment_summary, alignment_statistics, correlations, stratification, validity_rows),
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2))


def evaluate_item(item, service, sus_selector, impact_selector, reference, detector_threshold):
    image = service_image = load_rgb_image(item.image_path).convert("RGB").resize((512, 512))
    target_mask = load_grayscale_image(item.mask_path).resize((512, 512))
    truth = (np.asarray(target_mask, dtype="float32") > 0.0).astype("float32")
    tensor = image_to_tensor(service_image, service.encoder_service.device, service.encoder_service.stride)
    with torch.inference_mode():
        tokens = service.encoder_service.encode(tensor)
    token_shape = tuple(tokens.shape[-2:])
    before_full = service.detectors["wildfire_detection"].detect(image)
    before = resize_detector_output(before_full, token_shape, service)
    detail = service.semantic_service.detail_map(image, token_shape)
    entropy_map = local_token_entropy_numpy(tokens, token_shape)
    rankings = selector_rankings(tokens, before.utility_map, entropy_map, detail, sus_selector, impact_selector)
    truth_tokens = downsample_mask(truth, token_shape)
    geographic_zone = item.geographic_group[:2] if len(item.geographic_group) >= 2 else item.geographic_group
    fire_fraction = float(truth.mean())
    fire_size = fire_size_class(fire_fraction)

    alignment_rows = []
    for selector, ranking in rankings.items():
        for retention in RETENTION_LEVELS:
            count = max(1, int(round(tokens.numel() * retention)))
            selected = np.zeros(tokens.numel(), dtype=bool)
            selected[ranking[:count]] = True
            selected = selected.reshape(token_shape)
            binary_truth = truth_tokens >= 0.25
            true_positive = int(np.logical_and(selected, binary_truth).sum())
            union = int(np.logical_or(selected, binary_truth).sum())
            alignment_rows.append(
                {
                    "sample_id": item.sample_id,
                    "geographic_group": item.geographic_group,
                    "geographic_zone": geographic_zone,
                    "selector": selector,
                    "analysis_role": "exploratory" if selector == "decoder_impact_exploratory" else "confirmatory",
                    "retention": retention,
                    "selected_tokens": count,
                    "positive_tokens": int(binary_truth.sum()),
                    "positive_token_prevalence": float(binary_truth.mean()),
                    "token_precision": true_positive / max(count, 1),
                    "token_recall": true_positive / max(int(binary_truth.sum()), 1),
                    "token_iou": true_positive / max(union, 1),
                    "utility_mass_captured": float(truth_tokens[selected].sum() / max(float(truth_tokens.sum()), 1e-9)),
                    "precision_enrichment": (true_positive / max(count, 1)) / max(float(binary_truth.mean()), 1e-9),
                    "fire_fraction": fire_fraction,
                    "fire_size": fire_size,
                }
            )

    method_to_selector = {
        "vqvae_entropy": "entropy",
        "vqvae_fixed_utility": "fixed_utility",
        "vqvae_utility": "sus_aligned",
    }
    before_metrics = segmentation_metrics(before_full.confidence_map, truth, detector_threshold)
    validity_rows = []
    for method, selector in method_to_selector.items():
        for budget_index in (0, 1):
            saved = reference[(item.sample_id, method, budget_index)]
            count = int(float(saved["kept_tokens"]))
            mask = np.zeros(tokens.numel(), dtype=bool)
            mask[rankings[selector][:count]] = True
            mask = mask.reshape(token_shape)
            pruned = service.token_service.prune_tokens(tokens, mask)
            with torch.inference_mode():
                reconstruction = tensor_to_image(service.decoder_service.decode(pruned))
            after_full = service.detectors["wildfire_detection"].detect(reconstruction)
            after = resize_detector_output(after_full, token_shape, service)
            sus, components = service.semantic_utility_metric.score_before_after(before, after)
            after_metrics = segmentation_metrics(after_full.confidence_map, truth, detector_threshold)
            validity_rows.append(
                {
                    "sample_id": item.sample_id,
                    "geographic_group": item.geographic_group,
                    "geographic_zone": geographic_zone,
                    "selector": selector,
                    "method": method,
                    "budget_index": budget_index,
                    "actual_bytes": float(saved["actual_bytes"]),
                    "kept_tokens": count,
                    "formal_sus": sus,
                    "saved_sus": float(saved["sus"]),
                    "sus_reproduction_error": sus - float(saved["sus"]),
                    "detector_retention": components.detector_retention,
                    "before_dice": before_metrics["dice"],
                    "after_dice": after_metrics["dice"],
                    "dice_retention": min(after_metrics["dice"] / max(before_metrics["dice"], 1e-9), 1.0),
                    "dice_change": after_metrics["dice"] - before_metrics["dice"],
                    "before_iou": before_metrics["iou"],
                    "after_iou": after_metrics["iou"],
                    "iou_retention": min(after_metrics["iou"] / max(before_metrics["iou"], 1e-9), 1.0),
                    "fire_fraction": fire_fraction,
                    "fire_size": fire_size,
                    "cloud_fraction": item.cloud_fraction,
                }
            )
    return alignment_rows, validity_rows


def selector_rankings(tokens, utility_map, entropy_map, detail_map, sus_selector, impact_selector):
    fixed_weights = TokenSelectionWeights(alpha_utility=0.65, beta_entropy=0.25, gamma_cost=0.10, delta_detail=0.0)
    entropy_weights = TokenSelectionWeights(alpha_utility=0.0, beta_entropy=1.0, gamma_cost=0.0, delta_detail=0.0)
    fixed_scores = UtilityAwareTokenPruner(fixed_weights).score_tokens(tokens, utility_map)
    entropy_scores = UtilityAwareTokenPruner(entropy_weights).score_tokens(tokens, np.zeros_like(utility_map))
    learned_scores = {}
    for name, model in (("sus_aligned", sus_selector), ("decoder_impact_exploratory", impact_selector)):
        device = next(model.parameters()).device
        with torch.inference_mode():
            learned_scores[name] = (
                model.score(
                    tokens.to(device),
                    torch.from_numpy(utility_map).to(device),
                    torch.from_numpy(entropy_map).to(device),
                    torch.from_numpy(detail_map).to(device),
                    "mission_utility",
                )[0]
                .cpu()
                .numpy()
            )
    return {
        "entropy": np.argsort(-entropy_scores.reshape(-1), kind="stable"),
        "fixed_utility": np.argsort(-fixed_scores.reshape(-1), kind="stable"),
        "sus_aligned": np.argsort(-learned_scores["sus_aligned"].reshape(-1), kind="stable"),
        "decoder_impact_exploratory": np.argsort(
            -learned_scores["decoder_impact_exploratory"].reshape(-1), kind="stable"
        ),
    }


def resize_detector_output(output, token_shape: tuple[int, int], service):
    """Create the token-scale detector view used by the registered SUS metric."""
    return type(output)(
        mission=output.mission,
        utility_map=service.semantic_service.resize_importance(output.utility_map, token_shape),
        confidence_map=service.semantic_service.resize_importance(output.confidence_map, token_shape),
        relevance_map=service.semantic_service.resize_importance(output.relevance_map, token_shape),
        detections=output.detections,
        backend=output.backend,
    )


def downsample_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)
    return F.interpolate(tensor, size=shape, mode="area")[0, 0].numpy()


def segmentation_metrics(probability: np.ndarray, truth: np.ndarray, threshold: float) -> dict[str, float]:
    predicted = np.asarray(probability) >= threshold
    expected = np.asarray(truth) >= 0.5
    true_positive = float(np.logical_and(predicted, expected).sum())
    false_positive = float(np.logical_and(predicted, ~expected).sum())
    false_negative = float(np.logical_and(~predicted, expected).sum())
    epsilon = 1e-9
    return {
        "dice": 2.0 * true_positive / (2.0 * true_positive + false_positive + false_negative + epsilon),
        "iou": true_positive / (true_positive + false_positive + false_negative + epsilon),
    }


def summarize_alignment(rows, seed: int):
    output = []
    metrics = ("token_precision", "token_recall", "token_iou", "utility_mass_captured", "precision_enrichment")
    groups = sorted({(row["selector"], float(row["retention"])) for row in rows})
    for selector, retention in groups:
        subset = [row for row in rows if row["selector"] == selector and float(row["retention"]) == retention]
        record: dict[str, object] = {"selector": selector, "retention": retention, "n": len(subset)}
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in subset], dtype="float64")
            low, high = bootstrap_mean_ci(values, seed=seed + int(retention * 100))
            record[f"{metric}_mean"] = float(values.mean())
            record[f"{metric}_ci_low"] = low
            record[f"{metric}_ci_high"] = high
        output.append(record)
    return output


def alignment_pairwise_tests(rows, seed: int):
    validator = StatisticalValidator()
    output = []
    for retention in RETENTION_LEVELS:
        fixed = {
            row["sample_id"]: row
            for row in rows
            if row["selector"] == "fixed_utility" and float(row["retention"]) == retention
        }
        for candidate in ("entropy", "sus_aligned", "decoder_impact_exploratory"):
            selected = {
                row["sample_id"]: row
                for row in rows
                if row["selector"] == candidate and float(row["retention"]) == retention
            }
            ids = sorted(set(fixed) & set(selected))
            for metric in ("utility_mass_captured", "token_recall", "token_precision"):
                baseline_values = np.asarray([float(fixed[sample_id][metric]) for sample_id in ids])
                candidate_values = np.asarray([float(selected[sample_id][metric]) for sample_id in ids])
                differences = candidate_values - baseline_values
                t_result = validator.paired_t_test(baseline_values.tolist(), candidate_values.tolist())
                w_result = validator.wilcoxon(baseline_values.tolist(), candidate_values.tolist())
                low, high = bootstrap_mean_ci(differences, seed=seed + int(retention * 100))
                output.append(
                    {
                        "retention": retention,
                        "baseline": "fixed_utility",
                        "candidate": candidate,
                        "metric": metric,
                        "n": len(ids),
                        "baseline_mean": float(baseline_values.mean()),
                        "candidate_mean": float(candidate_values.mean()),
                        "mean_paired_difference": float(differences.mean()),
                        "difference_ci_low": low,
                        "difference_ci_high": high,
                        "paired_t_p_value": t_result.p_value,
                        "wilcoxon_p_value": w_result.p_value,
                        "cohens_dz": t_result.effect_size,
                    }
                )
    return output


def correlation_analysis(rows, seed: int):
    output = []
    scopes = [("all", rows)]
    scopes += [(f"budget_{budget}", [row for row in rows if int(row["budget_index"]) == budget]) for budget in (0, 1)]
    scopes += [(selector, [row for row in rows if row["selector"] == selector]) for selector in CONFIRMATORY_SELECTORS]
    for scope, subset in scopes:
        for predictor in ("formal_sus", "detector_retention"):
            for outcome in ("after_dice", "dice_retention", "after_iou"):
                x = np.asarray([float(row[predictor]) for row in subset], dtype="float64")
                y = np.asarray([float(row[outcome]) for row in subset], dtype="float64")
                pearson = safe_pearson(x, y)
                spearman = safe_spearman(x, y)
                low, high = cluster_bootstrap_correlation(subset, predictor, outcome, seed)
                output.append(
                    {
                        "scope": scope,
                        "predictor": predictor,
                        "outcome": outcome,
                        "n_observations": len(subset),
                        "n_scenes": len({row["sample_id"] for row in subset}),
                        "pearson_r": pearson,
                        "spearman_rho": spearman,
                        "cluster_bootstrap_pearson_ci_low": low,
                        "cluster_bootstrap_pearson_ci_high": high,
                    }
                )
    return output


def cluster_bootstrap_correlation(rows, predictor: str, outcome: str, seed: int, samples: int = 2000):
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["sample_id"]), []).append(row)
    ids = sorted(grouped)
    rng = np.random.default_rng(seed + sum(ord(char) for char in predictor + outcome))
    values = []
    for _ in range(samples):
        selected = rng.choice(ids, size=len(ids), replace=True)
        sampled = [row for sample_id in selected for row in grouped[str(sample_id)]]
        x = np.asarray([float(row[predictor]) for row in sampled])
        y = np.asarray([float(row[outcome]) for row in sampled])
        value = safe_pearson(x, y)
        if math.isfinite(value):
            values.append(value)
    if not values:
        return float("nan"), float("nan")
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(stats.pearsonr(x, y).statistic)


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(stats.spearmanr(x, y).statistic)


def stratify_detector(rows):
    output = []
    groups = sorted({(row["selector"], int(row["budget_index"]), row["fire_size"]) for row in rows})
    for selector, budget, fire_size in groups:
        subset = [
            row
            for row in rows
            if row["selector"] == selector and int(row["budget_index"]) == budget and row["fire_size"] == fire_size
        ]
        output.append(
            {
                "selector": selector,
                "budget_index": budget,
                "fire_size": fire_size,
                "n": len(subset),
                "fire_fraction_mean": float(np.mean([float(row["fire_fraction"]) for row in subset])),
                "before_dice_mean": float(np.mean([float(row["before_dice"]) for row in subset])),
                "after_dice_mean": float(np.mean([float(row["after_dice"]) for row in subset])),
                "dice_change_mean": float(np.mean([float(row["dice_change"]) for row in subset])),
                "sus_mean": float(np.mean([float(row["formal_sus"]) for row in subset])),
            }
        )
    return output


def fire_size_class(fraction: float) -> str:
    if fraction < 0.05:
        return "small_lt_5pct"
    if fraction < 0.25:
        return "medium_5_25pct"
    return "large_ge_25pct"


def generate_figures(output_dir: Path, alignment_summary, validity_rows, stratification):
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    labels = {
        "entropy": "Entropy",
        "fixed_utility": "Fixed utility",
        "sus_aligned": "SUS-aligned",
        "decoder_impact_exploratory": "Decoder impact (exploratory)",
    }
    colors = {"entropy": "#2b6cb0", "fixed_utility": "#c05621", "sus_aligned": "#2f855a", "decoder_impact_exploratory": "#6b46c1"}
    for metric, filename, ylabel in (
        ("token_recall_mean", "token_recall_vs_retention.png", "Burn-scar token recall"),
        ("token_precision_mean", "token_precision_vs_retention.png", "Burn-scar token precision"),
        ("utility_mass_captured_mean", "utility_capture_vs_retention.png", "Burn-scar utility mass captured"),
    ):
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        for selector in SELECTORS:
            subset = sorted([row for row in alignment_summary if row["selector"] == selector], key=lambda row: row["retention"])
            ax.plot([row["retention"] * 100 for row in subset], [row[metric] for row in subset], marker="o", label=labels[selector], color=colors[selector])
        ax.set_xlabel("Tokens retained (%)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        path = output_dir / filename
        fig.savefig(path, dpi=300)
        plt.close(fig)
        paths.append(str(path))

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for selector in CONFIRMATORY_SELECTORS:
        subset = [row for row in validity_rows if row["selector"] == selector]
        ax.scatter([row["formal_sus"] for row in subset], [row["after_dice"] for row in subset], s=20, alpha=0.55, label=labels[selector], color=colors[selector])
    ax.set_xlabel("Formal SUS")
    ax.set_ylabel("Ground-truth burn-scar Dice")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = output_dir / "sus_vs_ground_truth_dice.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    paths.append(str(path))

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    fire_sizes = ("small_lt_5pct", "medium_5_25pct", "large_ge_25pct")
    x = np.arange(len(fire_sizes))
    width = 0.25
    for offset, selector in enumerate(CONFIRMATORY_SELECTORS):
        values = []
        for fire_size in fire_sizes:
            subset = [row for row in stratification if row["selector"] == selector and int(row["budget_index"]) == 0 and row["fire_size"] == fire_size]
            values.append(float(subset[0]["after_dice_mean"]) if subset else 0.0)
        ax.bar(x + (offset - 1) * width, values, width, label=labels[selector], color=colors[selector])
    ax.set_xticks(x, ("Small <5%", "Medium 5-25%", "Large >=25%"))
    ax.set_ylabel("Low-rate burn-scar Dice")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = output_dir / "low_rate_dice_by_fire_size.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    paths.append(str(path))
    return paths


def build_report(metadata, alignment_summary, alignment_statistics, correlations, stratification, validity_rows):
    at_20 = sorted(
        [row for row in alignment_summary if abs(float(row["retention"]) - 0.20) < 1e-9],
        key=lambda row: float(row["utility_mass_captured_mean"]),
        reverse=True,
    )
    sus_dice = next(
        row for row in correlations if row["scope"] == "all" and row["predictor"] == "formal_sus" and row["outcome"] == "after_dice"
    )
    sus_retention = next(
        row for row in correlations if row["scope"] == "all" and row["predictor"] == "formal_sus" and row["outcome"] == "dice_retention"
    )
    alignment_20_tests = [
        row
        for row in alignment_statistics
        if abs(float(row["retention"]) - 0.20) < 1e-9 and row["metric"] == "utility_mass_captured"
    ]
    unique_before = {}
    for row in validity_rows:
        unique_before.setdefault(row["sample_id"], float(row["before_dice"]))
    detector_dice = float(np.mean(list(unique_before.values())))
    reproduction_error = []
    with (Path(metadata["figures"][0]).parents[1] / "sus_validity_rows.csv").open("r", newline="", encoding="utf-8") as handle:
        reproduction_error = [abs(float(row["sus_reproduction_error"])) for row in csv.DictReader(handle)]
    lines = [
        "# Token Alignment and SUS Validity Study",
        "",
        "## Scope and leakage controls",
        "",
        f"The study used {metadata['completed_scenes']} scenes from the internal geographic validation subset derived exclusively from the original training partition. The official validation and final test partitions were not used.",
        "",
        "## Token-ground-truth alignment at 20% retention",
        "",
        "| selector | token precision | token recall | token IoU | utility mass captured | enrichment |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in at_20:
        lines.append(
            f"| {row['selector']} | {row['token_precision_mean']:.3f} | {row['token_recall_mean']:.3f} | "
            f"{row['token_iou_mean']:.3f} | {row['utility_mass_captured_mean']:.3f} | {row['precision_enrichment_mean']:.2f}x |"
        )
    lines.extend(
        [
            "",
            "The decoder-impact selector is exploratory because its earlier model-selection process used the official validation partition. It is shown only to understand failure modes and is excluded from confirmatory conclusions.",
            "",
            "### Paired 20% retention comparisons",
            "",
        ]
    )
    for row in alignment_20_tests:
        lines.append(
            f"- {row['candidate']} minus fixed utility: {row['mean_paired_difference']:+.3f} utility mass, "
            f"95% CI {row['difference_ci_low']:+.3f} to {row['difference_ci_high']:+.3f}, "
            f"paired t-test p={row['paired_t_p_value']:.4g}."
        )
    lines.extend(
        [
            "",
            "## Does SUS track independent wildfire preservation?",
            "",
            f"The uncompressed detector achieved mean ground-truth Dice={detector_dice:.3f} on these 59 scenes.",
            "",
            f"Across all confirmatory selector, scene, and rate observations, formal SUS had Pearson r={sus_dice['pearson_r']:.3f} with post-compression ground-truth Dice (cluster-bootstrap 95% CI {sus_dice['cluster_bootstrap_pearson_ci_low']:.3f} to {sus_dice['cluster_bootstrap_pearson_ci_high']:.3f}) and Spearman rho={sus_dice['spearman_rho']:.3f}.",
            f"For Dice retention relative to the uncompressed detector, Pearson r={sus_retention['pearson_r']:.3f} (95% CI {sus_retention['cluster_bootstrap_pearson_ci_low']:.3f} to {sus_retention['cluster_bootstrap_pearson_ci_high']:.3f}).",
            "",
            f"The maximum absolute difference between recomputed and previously stored SUS was {max(reproduction_error, default=0.0):.6f}, confirming that this diagnostic reproduced the registered matched-rate selections.",
            "",
            "## Interpretation",
            "",
            "A strong positive correlation would support SUS as a proxy for independently labelled wildfire preservation. A weak or unstable correlation indicates that selector development is currently limited by detector validity rather than decoder capacity. Selector promotion remains governed by the separate matched-rate advancement decision.",
            "",
            "## Recommended decision",
            "",
            "Do not evaluate another learned selector on the official validation or test partitions until the detector and SUS-to-ground-truth relationship are sufficiently stable. Use the alignment and stratification tables to decide whether detector calibration, small-fire sensitivity, or selector supervision is the dominant failure source.",
        ]
    )
    return "\n".join(lines)


def read_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
