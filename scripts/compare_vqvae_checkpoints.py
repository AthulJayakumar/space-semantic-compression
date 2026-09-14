"""Compare VQ-VAE checkpoints on the same Earth Observation datasets.

This script is designed for model-promotion checks. It evaluates whether a
fine-tuned encoder/decoder improves end-to-end semantic compression evidence
against the original checkpoint while keeping the token selector fixed.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
import sys
import time

import numpy as np
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
from backend.utils.tensor_utils import image_to_tensor, tensor_to_image  # noqa: E402
from datasets.research_wildfire import load_rgb_image, looks_like_mask  # noqa: E402
from token_selection.utility_pruner import TokenSelectionWeights, UtilityAwareTokenPruner  # noqa: E402


@dataclass(frozen=True)
class CheckpointSpec:
    name: str
    path: Path


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    path: Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare original and fine-tuned VQ-VAE checkpoints.")
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="Checkpoint spec as NAME=PATH. Pass at least two.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        help="Dataset spec as NAME=PATH.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/model_improvement_step17_cross_dataset_vqvae_check"))
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--keep-ratio", type=float, default=0.8)
    parser.add_argument("--mission", default="wildfire_detection")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip-lpips", action="store_true")
    args = parser.parse_args()

    checkpoints = parse_checkpoints(args.checkpoint)
    datasets = parse_datasets(args.dataset)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    services = {spec.name: build_service(spec.path, args.device, args.output_dir / "outputs" / spec.name) for spec in checkpoints}
    pruner = UtilityAwareTokenPruner(TokenSelectionWeights.mission_utility())

    rows: list[dict[str, object]] = []
    for dataset in datasets:
        image_paths = discover_images(dataset.path, args.limit)
        if not image_paths:
            print(f"warning: no images found for {dataset.name} at {dataset.path}")
            continue
        for index, image_path in enumerate(image_paths, start=1):
            print(f"{dataset.name}: {index}/{len(image_paths)} {image_path.name}")
            for checkpoint in checkpoints:
                row = evaluate_image(
                    dataset_name=dataset.name,
                    checkpoint_name=checkpoint.name,
                    image_path=image_path,
                    service=services[checkpoint.name],
                    pruner=pruner,
                    keep_ratio=args.keep_ratio,
                    mission=args.mission,
                    skip_lpips=args.skip_lpips,
                )
                rows.append(row)
        write_rows(args.output_dir / "vqvae_checkpoint_comparison_rows.csv", rows)

    summary = summarize(rows)
    stats = paired_statistics(rows, baseline_checkpoint=checkpoints[0].name)
    write_rows(args.output_dir / "vqvae_checkpoint_comparison_summary.csv", summary)
    write_rows(args.output_dir / "vqvae_checkpoint_paired_statistics.csv", stats)
    (args.output_dir / "vqvae_checkpoint_comparison_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(args.output_dir / "vqvae_checkpoint_comparison_report.md", args, summary, stats)
    print(json.dumps({"rows": len(rows), "output_dir": str(args.output_dir)}, indent=2))


def evaluate_image(
    dataset_name: str,
    checkpoint_name: str,
    image_path: Path,
    service: CompressionService,
    pruner: UtilityAwareTokenPruner,
    keep_ratio: float,
    mission: str,
    skip_lpips: bool,
) -> dict[str, object]:
    started = time.perf_counter()
    original = load_rgb_image(image_path)
    image_bytes = image_path.read_bytes()
    tensor = image_to_tensor(original, service.encoder_service.device, service.encoder_service.stride)
    tokens = service.encoder_service.encode(tensor)
    token_shape = tuple(tokens.shape[-2:])
    semantic = service.semantic_service.analyze(original, token_shape)
    detector_before = service._detect_mission_utility(original, token_shape, mission)
    utility_map = np.maximum(semantic.importance_map, detector_before.utility_map).astype("float32")
    keep_mask, _ = pruner.select(tokens, utility_map, keep_ratio)
    pruned_tokens = service.token_service.prune_tokens(tokens, keep_mask)
    reconstruction = tensor_to_image(service.decoder_service.decode(pruned_tokens))
    detector_after = service._detect_mission_utility(reconstruction, token_shape, mission)
    sus, components = service.semantic_utility_metric.score_before_after(detector_before, detector_after)
    compressed_size_kb = service.token_service.estimate_payload_kb(pruned_tokens, keep_mask)
    original_size_kb = len(image_bytes) / 1024.0
    lpips_value = None if skip_lpips else service.metrics_service.lpips(original, reconstruction)
    return {
        "dataset": dataset_name,
        "checkpoint": checkpoint_name,
        "image": image_path.name,
        "semantic_utility_score": sus,
        "detector_retention": components.detector_retention,
        "object_retention": components.object_retention,
        "relevance_retention": components.relevance_retention,
        "region_preservation": components.region_preservation,
        "psnr": service.metrics_service.psnr(original, reconstruction),
        "ssim": service.metrics_service.ssim(original, reconstruction),
        "lpips": lpips_value,
        "compression_ratio": original_size_kb / compressed_size_kb if compressed_size_kb else 0.0,
        "bandwidth_saved_percent": max(0.0, (1.0 - compressed_size_kb / original_size_kb) * 100.0) if original_size_kb else 0.0,
        "semantic_token_count": int(np.asarray(keep_mask).sum()),
        "total_token_count": int(tokens.numel()),
        "inference_latency_ms": (time.perf_counter() - started) * 1000.0,
    }


def build_service(checkpoint_path: Path, device: str, output_dir: Path) -> CompressionService:
    encoder = EncoderService(checkpoint_path, device)
    decoder = DecoderService(encoder)
    return CompressionService(
        encoder_service=encoder,
        decoder_service=decoder,
        metrics_service=MetricsService(),
        semantic_service=SemanticService(),
        token_service=TokenTransmissionService(),
        transmission_service=SatelliteTransmissionService(),
        visualization_service=VisualizationService(output_dir),
        output_dir=output_dir,
    )


def discover_images(dataset_dir: Path, limit: int | None) -> list[Path]:
    images = sorted(
        [
            *dataset_dir.rglob("*.png"),
            *dataset_dir.rglob("*.jpg"),
            *dataset_dir.rglob("*.jpeg"),
            *dataset_dir.rglob("*.webp"),
            *dataset_dir.rglob("*.tif"),
            *dataset_dir.rglob("*.tiff"),
        ]
    )
    images = [path for path in images if not looks_like_mask(path)]
    return images[:limit] if limit else images


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    metrics = [
        "semantic_utility_score",
        "detector_retention",
        "psnr",
        "ssim",
        "lpips",
        "compression_ratio",
        "bandwidth_saved_percent",
        "inference_latency_ms",
    ]
    output: list[dict[str, object]] = []
    keys = sorted({(str(row["dataset"]), str(row["checkpoint"])) for row in rows})
    for dataset, checkpoint in keys:
        subset = [row for row in rows if row["dataset"] == dataset and row["checkpoint"] == checkpoint]
        item: dict[str, object] = {"dataset": dataset, "checkpoint": checkpoint, "n_images": len(subset)}
        for metric in metrics:
            values = [float(row[metric]) for row in subset if row.get(metric) is not None]
            if values:
                arr = np.asarray(values, dtype="float64")
                item[f"{metric}_mean"] = float(arr.mean())
                item[f"{metric}_std"] = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
        output.append(item)
    return output


def paired_statistics(rows: list[dict[str, object]], baseline_checkpoint: str) -> list[dict[str, object]]:
    metrics = ["semantic_utility_score", "detector_retention", "psnr", "ssim", "compression_ratio", "bandwidth_saved_percent"]
    output: list[dict[str, object]] = []
    try:
        from scipy import stats
    except Exception:
        stats = None
    datasets = sorted({str(row["dataset"]) for row in rows})
    checkpoints = sorted({str(row["checkpoint"]) for row in rows if row["checkpoint"] != baseline_checkpoint})
    by_key = {(row["dataset"], row["checkpoint"], row["image"]): row for row in rows}
    for dataset in datasets:
        images = sorted({str(row["image"]) for row in rows if row["dataset"] == dataset})
        for checkpoint in checkpoints:
            for metric in metrics:
                baseline_values: list[float] = []
                candidate_values: list[float] = []
                for image in images:
                    baseline = by_key.get((dataset, baseline_checkpoint, image))
                    candidate = by_key.get((dataset, checkpoint, image))
                    if baseline is None or candidate is None or baseline.get(metric) is None or candidate.get(metric) is None:
                        continue
                    baseline_values.append(float(baseline[metric]))
                    candidate_values.append(float(candidate[metric]))
                if len(baseline_values) < 2:
                    continue
                base = np.asarray(baseline_values, dtype="float64")
                cand = np.asarray(candidate_values, dtype="float64")
                diff = cand - base
                if stats is not None:
                    t_stat, t_p = stats.ttest_rel(cand, base)
                    try:
                        w_stat, w_p = stats.wilcoxon(cand, base, zero_method="wilcox")
                    except ValueError:
                        w_stat, w_p = 0.0, 1.0
                else:
                    t_stat, t_p, w_stat, w_p = 0.0, 1.0, 0.0, 1.0
                ci_low, ci_high = bootstrap_ci(diff)
                output.append(
                    {
                        "dataset": dataset,
                        "baseline": baseline_checkpoint,
                        "candidate": checkpoint,
                        "metric": metric,
                        "n": int(diff.size),
                        "baseline_mean": float(base.mean()),
                        "candidate_mean": float(cand.mean()),
                        "mean_difference_candidate_minus_baseline": float(diff.mean()),
                        "paired_t_statistic": float(t_stat),
                        "paired_t_p_value": float(t_p),
                        "wilcoxon_statistic": float(w_stat),
                        "wilcoxon_p_value": float(w_p),
                        "bootstrap_95ci_low": ci_low,
                        "bootstrap_95ci_high": ci_high,
                    }
                )
    return output


def bootstrap_ci(values: np.ndarray, confidence: float = 0.95, rounds: int = 2000, seed: int = 1234) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(rounds, dtype="float64")
    for index in range(rounds):
        means[index] = rng.choice(values, size=values.size, replace=True).mean()
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))


def write_report(path: Path, args: argparse.Namespace, summary: list[dict[str, object]], stats: list[dict[str, object]]) -> None:
    lines = [
        "# Cross-Dataset VQ-VAE Checkpoint Comparison",
        "",
        "## Purpose",
        "Compare the original VQ-VAE checkpoint against the satellite fine-tuned checkpoint while keeping the mission-utility token selector fixed.",
        "",
        "## Configuration",
        f"- Token retention: {args.keep_ratio}",
        f"- Per-dataset limit: {args.limit}",
        f"- LPIPS skipped: {args.skip_lpips}",
        "",
        "## Summary",
        "| Dataset | Checkpoint | Images | SUS | Detector Retention | PSNR | SSIM | Compression Ratio | Bandwidth Saved |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {dataset} | {checkpoint} | {n} | {sus:.3f} | {det:.4f} | {psnr:.3f} | {ssim:.4f} | {cr:.2f} | {bw:.2f}% |".format(
                dataset=row["dataset"],
                checkpoint=row["checkpoint"],
                n=int(row["n_images"]),
                sus=float(row.get("semantic_utility_score_mean") or 0.0),
                det=float(row.get("detector_retention_mean") or 0.0),
                psnr=float(row.get("psnr_mean") or 0.0),
                ssim=float(row.get("ssim_mean") or 0.0),
                cr=float(row.get("compression_ratio_mean") or 0.0),
                bw=float(row.get("bandwidth_saved_percent_mean") or 0.0),
            )
        )
    lines.extend(
        [
            "",
            "## Paired Statistics",
            "| Dataset | Candidate | Metric | Mean Difference | t-test p | 95% CI |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for row in stats:
        if row["metric"] not in {"semantic_utility_score", "detector_retention", "psnr", "ssim"}:
            continue
        lines.append(
            "| {dataset} | {candidate} | {metric} | {diff:.6f} | {p:.6g} | [{lo:.6f}, {hi:.6f}] |".format(
                dataset=row["dataset"],
                candidate=row["candidate"],
                metric=row["metric"],
                diff=float(row["mean_difference_candidate_minus_baseline"]),
                p=float(row["paired_t_p_value"]),
                lo=float(row["bootstrap_95ci_low"]),
                hi=float(row["bootstrap_95ci_high"]),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "The fine-tuned checkpoint should only be promoted if it improves satellite reconstruction and utility without unacceptable detector-retention loss on other validation datasets.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_checkpoints(values: list[str]) -> list[CheckpointSpec]:
    specs = []
    for value in values:
        name, path = parse_name_path(value, "--checkpoint")
        specs.append(CheckpointSpec(name=name, path=Path(path)))
    if len(specs) < 2:
        raise ValueError("Pass at least two checkpoints for comparison.")
    return specs


def parse_datasets(values: list[str]) -> list[DatasetSpec]:
    specs = []
    for value in values:
        name, path = parse_name_path(value, "--dataset")
        specs.append(DatasetSpec(name=name, path=Path(path)))
    return specs


def parse_name_path(value: str, flag: str) -> tuple[str, str]:
    if "=" not in value:
        raise ValueError(f"{flag} must use NAME=PATH format.")
    name, path = value.split("=", 1)
    return name.strip(), path.strip()


if __name__ == "__main__":
    main()
