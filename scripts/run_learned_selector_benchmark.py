"""Benchmark learned token selection against the fixed mission-utility selector.

This script evaluates whether the optional learned selector improves end-to-end
compression evidence after training. It deliberately uses the same validation
split as `train_mode_conditioned_token_scorer.py` so the reported images were
not used for selector training.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.factory import get_compression_service
from backend.utils.tensor_utils import image_to_tensor, tensor_to_image
from token_selection.learned_mode_selector import (
    ModeConditionedSelectorConfig,
    ModeConditionedTokenScorer,
    local_token_entropy_numpy,
)
from token_selection.utility_pruner import TokenSelectionWeights, UtilityAwareTokenPruner


@dataclass(frozen=True)
class SelectorSpec:
    name: str
    kind: str
    checkpoint_path: Path | None = None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run held-out learned-selector compression benchmark.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("datasets/sentinel2_full_patch_256/images"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/model_improvement_step10_learned_selector_benchmark"))
    parser.add_argument("--keep-ratio", type=float, default=0.8)
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip-lpips", action="store_true")
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        help="Learned selector checkpoint as NAME=PATH. May be passed multiple times.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    service = get_compression_service()
    device = resolve_device(args.device)
    selectors = build_selectors(args.checkpoint)
    learned_models = {
        spec.name: load_learned_model(spec.checkpoint_path, device)
        for spec in selectors
        if spec.kind == "learned" and spec.checkpoint_path is not None and spec.checkpoint_path.exists()
    }
    if not learned_models:
        raise FileNotFoundError("No learned selector checkpoints were found for benchmarking.")

    image_paths = validation_images(args.dataset_dir, args.validation_fraction, args.seed, args.limit)
    if not image_paths:
        raise ValueError(f"No validation images found in {args.dataset_dir}")

    rows: list[dict[str, object]] = []
    for index, image_path in enumerate(image_paths, start=1):
        print(f"benchmarking {index}/{len(image_paths)} {image_path.name}")
        rows.extend(evaluate_image(image_path, selectors, learned_models, service, device, args.keep_ratio, args.skip_lpips))
        write_rows(args.output_dir / "learned_selector_benchmark_rows.csv", rows)

    summary_rows = summarize(rows)
    paired_rows = paired_statistics(rows, baseline_selector="fixed_mission_utility")
    write_rows(args.output_dir / "learned_selector_benchmark_summary.csv", summary_rows)
    write_rows(args.output_dir / "learned_selector_paired_statistics.csv", paired_rows)
    write_json(args.output_dir / "learned_selector_benchmark_summary.json", summary_rows)
    write_report(args.output_dir / "learned_selector_benchmark_report.md", args, rows, summary_rows, paired_rows)
    make_plots(args.output_dir, summary_rows)
    print(json.dumps({"rows": len(rows), "images": len(image_paths), "output_dir": str(args.output_dir)}, indent=2))


def build_selectors(checkpoint_args: list[str]) -> list[SelectorSpec]:
    selectors = [SelectorSpec(name="fixed_mission_utility", kind="fixed")]
    if checkpoint_args:
        for item in checkpoint_args:
            if "=" not in item:
                raise ValueError("--checkpoint must use NAME=PATH format")
            name, raw_path = item.split("=", 1)
            selectors.append(SelectorSpec(name=name.strip(), kind="learned", checkpoint_path=Path(raw_path.strip())))
        return selectors

    defaults = [
        ("learned_1500_patch", Path("models/checkpoints/mode_conditioned_token_scorer_sentinel2_1500.pt")),
        ("learned_full_3224_patch", Path("models/checkpoints/mode_conditioned_token_scorer_sentinel2_full.pt")),
    ]
    selectors.extend(SelectorSpec(name=name, kind="learned", checkpoint_path=path) for name, path in defaults if path.exists())
    return selectors


def evaluate_image(
    image_path: Path,
    selectors: list[SelectorSpec],
    learned_models: dict[str, ModeConditionedTokenScorer],
    service,
    device: torch.device,
    keep_ratio: float,
    skip_lpips: bool,
) -> list[dict[str, object]]:
    image_bytes = image_path.read_bytes()
    original = Image.open(image_path).convert("RGB")
    tensor = image_to_tensor(original, service.encoder_service.device, service.encoder_service.stride)
    tokens = service.encoder_service.encode(tensor)
    token_shape = tuple(tokens.shape[-2:])

    semantic = service.semantic_service.analyze(original, token_shape)
    detector_before = service._detect_mission_utility(original, token_shape, "wildfire_detection")
    utility_map = np.maximum(semantic.importance_map, detector_before.utility_map).astype("float32")
    detail_map = service.semantic_service.detail_map(original, token_shape)
    entropy_map = local_token_entropy_numpy(tokens, token_shape)
    fixed_pruner = UtilityAwareTokenPruner(TokenSelectionWeights.mission_utility())

    rows: list[dict[str, object]] = []
    for spec in selectors:
        started = time.perf_counter()
        if spec.kind == "fixed":
            keep_mask, _ = fixed_pruner.select(tokens, utility_map, keep_ratio)
        else:
            model = learned_models.get(spec.name)
            if model is None:
                continue
            keep_mask, _ = learned_mask(model, tokens, utility_map, entropy_map, detail_map, keep_ratio, device)

        pruned_tokens = service.token_service.prune_tokens(tokens, keep_mask)
        reconstruction = tensor_to_image(service.decoder_service.decode(pruned_tokens))
        detector_after = service._detect_mission_utility(reconstruction, token_shape, "wildfire_detection")
        sus, sus_components = service.semantic_utility_metric.score_before_after(detector_before, detector_after)
        compressed_size_kb = service.token_service.estimate_payload_kb(pruned_tokens, keep_mask)
        original_size_kb = len(image_bytes) / 1024.0
        full_payload_kb = service.token_service.estimate_payload_kb(tokens)
        token_entropy = service.token_service.token_entropy_bits(pruned_tokens)
        semantic_fidelity = service.token_service.semantic_fidelity_percent(utility_map, keep_mask)
        lpips_value = None if skip_lpips else service.metrics_service.lpips(original, reconstruction)

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        rows.append(
            {
                "image": image_path.name,
                "selector": spec.name,
                "keep_ratio": keep_ratio,
                "semantic_utility_score": sus,
                "detector_retention": sus_components.detector_retention,
                "object_retention": sus_components.object_retention,
                "relevance_retention": sus_components.relevance_retention,
                "region_preservation": sus_components.region_preservation,
                "psnr": service.metrics_service.psnr(original, reconstruction),
                "ssim": service.metrics_service.ssim(original, reconstruction),
                "lpips": lpips_value,
                "compression_ratio": original_size_kb / compressed_size_kb if compressed_size_kb else 0.0,
                "bandwidth_saved_percent": max(0.0, (1.0 - compressed_size_kb / original_size_kb) * 100.0) if original_size_kb else 0.0,
                "original_size_kb": original_size_kb,
                "compressed_size_kb": compressed_size_kb,
                "full_payload_kb": full_payload_kb,
                "semantic_token_count": int(np.asarray(keep_mask).sum()),
                "total_token_count": int(tokens.numel()),
                "token_entropy_bits": token_entropy,
                "semantic_fidelity_percent": semantic_fidelity,
                "inference_latency_ms": elapsed_ms,
            }
        )
    return rows


def learned_mask(
    model: ModeConditionedTokenScorer,
    tokens: torch.Tensor,
    utility_map: np.ndarray,
    entropy_map: np.ndarray,
    detail_map: np.ndarray,
    keep_ratio: float,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        keep_mask, scores = model.select(
            tokens.detach().cpu().long().to(device),
            torch.from_numpy(utility_map).to(device),
            torch.from_numpy(entropy_map).to(device),
            torch.from_numpy(detail_map).to(device),
            "mission_utility",
            keep_ratio,
        )
    return keep_mask.squeeze(0).detach().cpu().numpy().astype(bool), scores.squeeze(0).detach().cpu().numpy()


def validation_images(dataset_dir: Path, validation_fraction: float, seed: int, limit: int | None) -> list[Path]:
    images = sorted(
        [
            *dataset_dir.rglob("*.png"),
            *dataset_dir.rglob("*.jpg"),
            *dataset_dir.rglob("*.jpeg"),
            *dataset_dir.rglob("*.webp"),
        ]
    )
    if len(images) < 2:
        return images
    fraction = float(np.clip(validation_fraction, 0.0, 0.8))
    rng = np.random.default_rng(seed)
    indices = np.arange(len(images))
    rng.shuffle(indices)
    validation_count = max(1, int(round(len(images) * fraction)))
    validation_indices = sorted(indices[:validation_count].tolist())
    selected = [images[index] for index in validation_indices]
    return selected[:limit] if limit else selected


def load_learned_model(path: Path, device: torch.device) -> ModeConditionedTokenScorer:
    checkpoint = torch.load(path, map_location=device)
    config = ModeConditionedSelectorConfig(**checkpoint.get("config", {}))
    model = ModeConditionedTokenScorer(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    selectors = sorted({str(row["selector"]) for row in rows})
    metrics = [
        "semantic_utility_score",
        "detector_retention",
        "psnr",
        "ssim",
        "lpips",
        "compression_ratio",
        "bandwidth_saved_percent",
        "semantic_fidelity_percent",
        "token_entropy_bits",
        "inference_latency_ms",
    ]
    summary: list[dict[str, object]] = []
    for selector in selectors:
        subset = [row for row in rows if row["selector"] == selector]
        item: dict[str, object] = {"selector": selector, "n_images": len(subset)}
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in subset if row.get(metric) is not None], dtype="float64")
            if values.size == 0:
                item[f"{metric}_mean"] = None
                item[f"{metric}_std"] = None
                continue
            item[f"{metric}_mean"] = float(values.mean())
            item[f"{metric}_std"] = float(values.std(ddof=1)) if values.size > 1 else 0.0
        summary.append(item)
    return summary


def paired_statistics(rows: list[dict[str, object]], baseline_selector: str) -> list[dict[str, object]]:
    try:
        from scipy import stats
    except Exception:
        return []

    metrics = ["semantic_utility_score", "detector_retention", "psnr", "ssim", "lpips"]
    selectors = sorted({str(row["selector"]) for row in rows if row["selector"] != baseline_selector})
    rows_by_key = {(str(row["image"]), str(row["selector"])): row for row in rows}
    images = sorted({str(row["image"]) for row in rows if row["selector"] == baseline_selector})
    output: list[dict[str, object]] = []

    for selector in selectors:
        for metric in metrics:
            baseline_values: list[float] = []
            candidate_values: list[float] = []
            for image in images:
                baseline_row = rows_by_key.get((image, baseline_selector))
                candidate_row = rows_by_key.get((image, selector))
                if baseline_row is None or candidate_row is None:
                    continue
                baseline_value = baseline_row.get(metric)
                candidate_value = candidate_row.get(metric)
                if baseline_value is None or candidate_value is None:
                    continue
                baseline_values.append(float(baseline_value))
                candidate_values.append(float(candidate_value))
            if len(baseline_values) < 2:
                continue

            baseline_array = np.asarray(baseline_values, dtype="float64")
            candidate_array = np.asarray(candidate_values, dtype="float64")
            diff = candidate_array - baseline_array
            t_stat, t_p = stats.ttest_rel(candidate_array, baseline_array)
            try:
                wilcoxon_stat, wilcoxon_p = stats.wilcoxon(candidate_array, baseline_array, zero_method="wilcox")
            except ValueError:
                wilcoxon_stat, wilcoxon_p = 0.0, 1.0
            ci_low, ci_high = bootstrap_ci(diff)
            output.append(
                {
                    "baseline": baseline_selector,
                    "candidate": selector,
                    "metric": metric,
                    "n": int(diff.size),
                    "baseline_mean": float(baseline_array.mean()),
                    "candidate_mean": float(candidate_array.mean()),
                    "mean_difference_candidate_minus_baseline": float(diff.mean()),
                    "cohens_d_paired": float(diff.mean() / diff.std(ddof=1)) if diff.size > 1 and diff.std(ddof=1) > 1e-12 else 0.0,
                    "paired_t_statistic": float(t_stat),
                    "paired_t_p_value": float(t_p),
                    "wilcoxon_statistic": float(wilcoxon_stat),
                    "wilcoxon_p_value": float(wilcoxon_p),
                    "bootstrap_95ci_low": ci_low,
                    "bootstrap_95ci_high": ci_high,
                }
            )
    return output


def bootstrap_ci(values: np.ndarray, confidence: float = 0.95, rounds: int = 2000, seed: int = 1234) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    if values.size == 0:
        return 0.0, 0.0
    means = np.empty(rounds, dtype="float64")
    for index in range(rounds):
        sample = rng.choice(values, size=values.size, replace=True)
        means[index] = sample.mean()
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def write_report(
    path: Path,
    args: argparse.Namespace,
    rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
    paired_rows: list[dict[str, object]],
) -> None:
    best_sus = max(summary_rows, key=lambda row: float(row.get("semantic_utility_score_mean") or 0.0))
    best_psnr = max(summary_rows, key=lambda row: float(row.get("psnr_mean") or 0.0))
    lines = [
        "# Learned Selector Held-Out Benchmark",
        "",
        "## Purpose",
        "This benchmark evaluates whether learned token-priority selectors improve end-to-end compression results on held-out Sentinel-2 patches.",
        "",
        "## Configuration",
        f"- Dataset directory: `{args.dataset_dir}`",
        f"- Held-out images evaluated: {len({row['image'] for row in rows})}",
        f"- Token retention ratio: {args.keep_ratio}",
        f"- Validation fraction: {args.validation_fraction}",
        f"- Seed: {args.seed}",
        f"- LPIPS skipped: {args.skip_lpips}",
        "",
        "## Summary",
        "| Selector | SUS | Detector Retention | PSNR | SSIM | LPIPS | Compression Ratio | Bandwidth Saved |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {selector} | {sus:.3f} | {det:.4f} | {psnr:.3f} | {ssim:.4f} | {lpips} | {cr:.2f} | {bw:.2f}% |".format(
                selector=row["selector"],
                sus=float(row.get("semantic_utility_score_mean") or 0.0),
                det=float(row.get("detector_retention_mean") or 0.0),
                psnr=float(row.get("psnr_mean") or 0.0),
                ssim=float(row.get("ssim_mean") or 0.0),
                lpips="n/a" if row.get("lpips_mean") is None else f"{float(row['lpips_mean']):.4f}",
                cr=float(row.get("compression_ratio_mean") or 0.0),
                bw=float(row.get("bandwidth_saved_percent_mean") or 0.0),
            )
        )
    lines.extend(
        [
            "",
            "## Paired Comparison Against Fixed Mission Utility",
            "| Candidate | Metric | Mean Difference | Paired t-test p | Wilcoxon p | 95% CI |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in paired_rows:
        lines.append(
            "| {candidate} | {metric} | {diff:.6f} | {tp:.6g} | {wp:.6g} | [{lo:.6f}, {hi:.6f}] |".format(
                candidate=row["candidate"],
                metric=row["metric"],
                diff=float(row["mean_difference_candidate_minus_baseline"]),
                tp=float(row["paired_t_p_value"]),
                wp=float(row["wilcoxon_p_value"]),
                lo=float(row["bootstrap_95ci_low"]),
                hi=float(row["bootstrap_95ci_high"]),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            f"- Best mean SUS: `{best_sus['selector']}`.",
            f"- Best mean PSNR: `{best_psnr['selector']}`.",
            "- A learned selector should only be promoted if it improves mission metrics such as SUS and detector retention, not only training loss.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_plots(output_dir: Path, summary_rows: list[dict[str, object]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    labels = [str(row["selector"]) for row in summary_rows]
    sus = [float(row.get("semantic_utility_score_mean") or 0.0) for row in summary_rows]
    psnr = [float(row.get("psnr_mean") or 0.0) for row in summary_rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(labels, sus, color="#2f6f5e")
    axes[0].set_title("Mean SUS")
    axes[0].tick_params(axis="x", rotation=25)
    axes[1].bar(labels, psnr, color="#315f9f")
    axes[1].set_title("Mean PSNR")
    axes[1].tick_params(axis="x", rotation=25)
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "learned_selector_summary.png", dpi=200)
    plt.close(fig)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


if __name__ == "__main__":
    main()
