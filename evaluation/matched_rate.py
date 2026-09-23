"""Matched-byte evaluation for classical and semantic image codecs.

Every method is calibrated independently for each image.  The target budgets
are chosen from the byte range that all methods can reach, which prevents a
method from receiving a hidden rate advantage.  All reported payload sizes are
measured serialized byte counts.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
from PIL import Image

from backend.services.compression_service import CompressionService
from backend.utils.tensor_utils import image_to_tensor, tensor_to_image
from datasets.research_wildfire import load_rgb_image
from datasets.research_wildfire import load_grayscale_image
from evaluation.label_fidelity import binary_overlap
from evaluation.statistics import StatisticalValidator
from token_selection.learned_mode_selector import ModeConditionedTokenScorer, local_token_entropy_numpy
from token_selection.utility_pruner import TokenSelectionWeights, UtilityAwareTokenPruner


METHODS = ("jpeg", "jpeg2000", "vqvae_random", "vqvae_entropy", "vqvae_utility")
METRICS = ("sus", "detector_retention", "psnr", "ssim", "lpips", "label_dice", "label_iou", "actual_bytes", "encoded_bytes")


@dataclass(frozen=True)
class BenchmarkItem:
    sample_id: str
    image_path: Path
    mask_path: Path | None = None
    dataset: str = "unknown"
    split: str = "test"
    source_split: str = "unknown"
    geographic_group: str = "unknown"
    event_group: str = "unknown"
    label_source: str = "none"
    image_unit: str = "independent_tile"
    cloud_fraction: float | None = None
    smoke_label_status: str = "unavailable"
    valid_mask_path: Path | None = None


@dataclass(frozen=True)
class EncodedCandidate:
    payload: bytes
    reconstruction: Image.Image | None
    parameter: float
    keep_count: int | None = None

    @property
    def size(self) -> int:
        return len(self.payload)


class MatchedRateBenchmark:
    """Run paired, matched-byte codec comparisons on one held-out split."""

    def __init__(
        self,
        service: CompressionService,
        output_dir: Path,
        image_size: int = 512,
        budget_levels: tuple[float, ...] = (0.0, 0.5, 1.0),
        seed: int = 20260919,
        skip_lpips: bool = False,
        utility_weights: TokenSelectionWeights | None = None,
        utility_map_mode: str = "hybrid_max",
        semantic_blend_weight: float = 0.25,
        context_radius: int = 1,
        learned_selector: ModeConditionedTokenScorer | None = None,
        selector_name: str = "fixed_weighted",
    ) -> None:
        self.service = service
        self.output_dir = output_dir
        self.image_size = image_size
        self.budget_levels = budget_levels
        self.seed = seed
        self.skip_lpips = skip_lpips
        self.utility_weights = utility_weights or TokenSelectionWeights.mission_utility()
        self.utility_map_mode = utility_map_mode
        self.semantic_blend_weight = float(np.clip(semantic_blend_weight, 0.0, 1.0))
        self.context_radius = max(0, int(context_radius))
        self.learned_selector = learned_selector
        self.selector_name = selector_name
        self.methods = (
            (*METHODS, "vqvae_fixed_utility") if learned_selector is not None else METHODS
        )
        self.validator = StatisticalValidator()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._truth_masks: dict[tuple[Path, tuple[int, int]], np.ndarray] = {}

    def run(self, items: list[BenchmarkItem]) -> dict[str, object]:
        rows: list[dict[str, object]] = []
        exclusions: list[dict[str, object]] = []
        started = time.perf_counter()
        for index, item in enumerate(items, start=1):
            print(f"matched-rate {index}/{len(items)}: {item.sample_id}", flush=True)
            try:
                image_rows = self.evaluate_item(item)
                rows.extend(image_rows)
            except Exception as exc:
                exclusions.append({"sample_id": item.sample_id, "image_path": str(item.image_path), "reason": str(exc)})
            self._write_csv(self.output_dir / "matched_rate_rows.csv", rows)
            self._write_csv(self.output_dir / "excluded_samples.csv", exclusions)

        summary = self.summarize(rows)
        statistics = self.statistical_tests(rows)
        confirmatory_rows = [row for row in rows if row.get("source_split") == "validation"]
        confirmatory_summary = self.summarize(confirmatory_rows)
        confirmatory_statistics = self.statistical_tests(confirmatory_rows)
        pareto = self.pareto_frontier(confirmatory_summary or summary)
        figures = self.generate_figures(summary, pareto)
        self._write_csv(self.output_dir / "matched_rate_summary.csv", summary)
        self._write_csv(self.output_dir / "matched_rate_statistics.csv", statistics)
        self._write_csv(self.output_dir / "confirmatory_source_validation_summary.csv", confirmatory_summary)
        self._write_csv(self.output_dir / "confirmatory_source_validation_statistics.csv", confirmatory_statistics)
        self._write_csv(self.output_dir / "rate_utility_pareto.csv", pareto)
        payload = {
            "requested_items": len(items),
            "completed_items": len({str(row["sample_id"]) for row in rows}),
            "confirmatory_items": len({str(row["sample_id"]) for row in confirmatory_rows}),
            "excluded_items": len(exclusions),
            "rows": len(rows),
            "methods": list(self.methods),
            "budget_levels": list(self.budget_levels),
            "image_size": self.image_size,
            "elapsed_seconds": time.perf_counter() - started,
            "skip_lpips": self.skip_lpips,
            "wire_payload_roundtrip": True,
            "utility_map_mode": self.utility_map_mode,
            "semantic_blend_weight": self.semantic_blend_weight,
            "context_radius": self.context_radius,
            "utility_weights": {
                "alpha_utility": self.utility_weights.alpha_utility,
                "beta_entropy": self.utility_weights.beta_entropy,
                "gamma_cost": self.utility_weights.gamma_cost,
                "delta_detail": self.utility_weights.delta_detail,
            },
            "selector_name": self.selector_name,
            "figures": figures,
        }
        (self.output_dir / "run_metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.write_report(payload, summary, statistics, confirmatory_summary, confirmatory_statistics, exclusions)
        return payload

    def evaluate_item(self, item: BenchmarkItem) -> list[dict[str, object]]:
        original = self._prepare_image(load_rgb_image(item.image_path))
        tensor = image_to_tensor(original, self.service.encoder_service.device, self.service.encoder_service.stride)
        with torch.inference_mode():
            tokens = self.service.encoder_service.encode(tensor)
        token_shape = tuple(tokens.shape[-2:])
        before = self.service._detect_mission_utility(original, token_shape, "wildfire_detection")
        semantic = self.service.semantic_service.analyze(original, token_shape)
        utility_map = self.compose_utility_map(before.utility_map, semantic.importance_map)
        detail_map = self.service.semantic_service.detail_map(original, token_shape)
        rankings = self._rankings(tokens, utility_map, detail_map, item.sample_id)

        codec_cache: dict[tuple[str, int], EncodedCandidate] = {}

        def token_candidate(method: str, keep_count: int) -> EncodedCandidate:
            key = (method, keep_count)
            if key in codec_cache:
                return codec_cache[key]
            keep_count = int(np.clip(keep_count, 1, tokens.numel()))
            mask = np.zeros(tokens.numel(), dtype=bool)
            mask[rankings[method][:keep_count]] = True
            mask = mask.reshape(token_shape)
            payload = self.service.token_service.serialize_payload(tokens, mask)
            candidate = EncodedCandidate(payload, None, keep_count / tokens.numel(), keep_count)
            codec_cache[key] = candidate
            return candidate

        jpeg_min = self._jpeg(original, 1)
        jpeg_max = self._jpeg(original, 95)
        j2k_min = self._jpeg2000(original, 512)
        j2k_max = self._jpeg2000(original, 1)
        token_min_sizes = [token_candidate(method, 1).size for method in rankings]
        token_max_sizes = [token_candidate(method, tokens.numel()).size for method in rankings]
        common_low = max([jpeg_min.size, j2k_min.size, *token_min_sizes])
        common_high = min([jpeg_max.size, j2k_max.size, *token_max_sizes])
        if common_high < common_low:
            raise RuntimeError(
                f"No common byte interval: lower={common_low}, upper={common_high}. "
                "The image cannot support a fair matched-rate comparison."
            )

        rows: list[dict[str, object]] = []
        for budget_index, level in enumerate(self.budget_levels):
            target = int(round(common_low + float(level) * (common_high - common_low)))
            candidates: dict[str, EncodedCandidate] = {
                "jpeg": self._calibrate(target, 1, 95, lambda q: self._jpeg(original, q), increasing=True),
                "jpeg2000": self._calibrate(target, 1, 512, lambda r: self._jpeg2000(original, r), increasing=False),
            }
            for method in rankings:
                candidates[method] = self._calibrate(
                    target,
                    1,
                    tokens.numel(),
                    lambda n, method=method: token_candidate(method, n),
                    increasing=True,
                )
            for method in rankings:
                candidate = candidates[method]
                if candidate.keep_count is None:
                    raise RuntimeError(f"Missing keep count for {method}")
                mask = np.zeros(tokens.numel(), dtype=bool)
                mask[rankings[method][: candidate.keep_count]] = True
                mask = mask.reshape(token_shape)
                received, received_mask = self.service.token_service.deserialize_payload(candidate.payload)
                if not np.array_equal(received_mask, mask):
                    raise RuntimeError("Decoded token mask does not match transmitted selection")
                with torch.inference_mode():
                    reconstruction = tensor_to_image(self.service.decoder_service.decode(received))
                candidates[method] = EncodedCandidate(
                    payload=candidate.payload,
                    reconstruction=reconstruction,
                    parameter=candidate.parameter,
                    keep_count=candidate.keep_count,
                )
            for method, candidate in candidates.items():
                rows.append(
                    self._measure(
                        item=item,
                        original=original,
                        before=before,
                        token_shape=token_shape,
                        method=method,
                        budget_index=budget_index,
                        budget_level=level,
                        target_bytes=target,
                        common_low=common_low,
                        common_high=common_high,
                        candidate=candidate,
                    )
                )
        return rows

    def _rankings(
        self,
        tokens: torch.Tensor,
        utility_map: np.ndarray,
        detail_map: np.ndarray,
        sample_id: str,
    ) -> dict[str, np.ndarray]:
        total = tokens.numel()
        seed_bytes = hashlib.sha256(f"{self.seed}:{sample_id}".encode("utf-8")).digest()[:8]
        random_order = np.random.default_rng(int.from_bytes(seed_bytes, "little")).permutation(total)
        entropy_weights = TokenSelectionWeights(alpha_utility=0.0, beta_entropy=1.0, gamma_cost=0.0, delta_detail=0.0)
        entropy_scores = UtilityAwareTokenPruner(entropy_weights).score_tokens(tokens, np.zeros_like(utility_map))
        fixed_utility_scores = UtilityAwareTokenPruner(self.utility_weights).score_tokens(tokens, utility_map)
        if self.learned_selector is None:
            utility_scores = fixed_utility_scores
        else:
            device = next(self.learned_selector.parameters()).device
            entropy_map = local_token_entropy_numpy(tokens, utility_map.shape)
            with torch.inference_mode():
                utility_scores = (
                    self.learned_selector.score(
                        tokens.to(device),
                        torch.from_numpy(utility_map).to(device),
                        torch.from_numpy(entropy_map).to(device),
                        torch.from_numpy(detail_map).to(device),
                        "mission_utility",
                    )[0]
                    .detach()
                    .cpu()
                    .numpy()
                )
        rankings = {
            "vqvae_random": random_order,
            "vqvae_entropy": np.argsort(-entropy_scores.reshape(-1), kind="stable"),
            "vqvae_utility": np.argsort(-utility_scores.reshape(-1), kind="stable"),
        }
        if self.learned_selector is not None:
            rankings["vqvae_fixed_utility"] = np.argsort(
                -fixed_utility_scores.reshape(-1), kind="stable"
            )
        return rankings

    def compose_utility_map(self, detector_map: np.ndarray, semantic_map: np.ndarray) -> np.ndarray:
        """Compose token-scale importance without changing the detector or SUS definition."""
        detector = self._normalize_map(detector_map)
        semantic = self._normalize_map(semantic_map)
        if self.utility_map_mode == "detector_only":
            return detector
        if self.utility_map_mode == "weighted_blend":
            weight = self.semantic_blend_weight
            return self._normalize_map((1.0 - weight) * detector + weight * semantic)
        if self.utility_map_mode == "detector_context":
            return self._max_filter(detector, self.context_radius)
        if self.utility_map_mode == "hybrid_max":
            return np.maximum(detector, semantic).astype("float32")
        raise ValueError(f"Unsupported utility map mode: {self.utility_map_mode}")

    @staticmethod
    def _normalize_map(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype="float32")
        lo = float(values.min()) if values.size else 0.0
        hi = float(values.max()) if values.size else 0.0
        if hi - lo < 1e-8:
            return np.zeros_like(values, dtype="float32")
        return ((values - lo) / (hi - lo)).astype("float32")

    @staticmethod
    def _max_filter(values: np.ndarray, radius: int) -> np.ndarray:
        if radius <= 0:
            return values.astype("float32")
        padded = np.pad(values, radius, mode="edge")
        windows = [
            padded[row : row + values.shape[0], col : col + values.shape[1]]
            for row in range(2 * radius + 1)
            for col in range(2 * radius + 1)
        ]
        return np.maximum.reduce(windows).astype("float32")

    def _measure(
        self,
        item: BenchmarkItem,
        original: Image.Image,
        before,
        token_shape: tuple[int, int],
        method: str,
        budget_index: int,
        budget_level: float,
        target_bytes: int,
        common_low: int,
        common_high: int,
        candidate: EncodedCandidate,
    ) -> dict[str, object]:
        if candidate.reconstruction is None:
            raise RuntimeError(f"Missing reconstruction for {method}")
        after = self.service._detect_mission_utility(candidate.reconstruction, token_shape, "wildfire_detection")
        sus, components = self.service.semantic_utility_metric.score_before_after(before, after)
        label_dice = label_iou = None
        label_precision = label_recall = label_false_positive_rate = None
        predicted_positive_fraction = truth_positive_fraction = None
        label_valid_fraction = None
        if item.mask_path is not None:
            mask_key = (item.mask_path, original.size)
            if mask_key not in self._truth_masks:
                mask = load_grayscale_image(item.mask_path).resize(original.size, Image.Resampling.NEAREST)
                self._truth_masks[mask_key] = np.asarray(mask, dtype="uint8") > 0
            detector = self.service.detectors["wildfire_detection"]
            full_after = detector.detect(candidate.reconstruction)
            config = getattr(detector, "_supervised_config", None) or {}
            threshold = float(config.get("threshold", 0.5))
            predicted = full_after.confidence_map >= threshold
            truth = self._truth_masks[mask_key]
            if item.valid_mask_path is not None:
                valid_key = (item.valid_mask_path, original.size)
                if valid_key not in self._truth_masks:
                    valid_image = load_grayscale_image(item.valid_mask_path).resize(original.size, Image.Resampling.NEAREST)
                    self._truth_masks[valid_key] = np.asarray(valid_image, dtype="uint8") > 0
                valid = self._truth_masks[valid_key]
                if not np.any(valid):
                    raise ValueError(f"No valid label pixels for {item.sample_id}")
                label_valid_fraction = float(np.mean(valid))
                predicted = predicted[valid]
                truth = truth[valid]
            label_dice, label_iou = binary_overlap(predicted, truth)
            true_positive = int(np.count_nonzero(predicted & truth))
            false_positive = int(np.count_nonzero(predicted & ~truth))
            false_negative = int(np.count_nonzero(~predicted & truth))
            true_negative = int(np.count_nonzero(~predicted & ~truth))
            label_precision = true_positive / max(true_positive + false_positive, 1)
            label_recall = true_positive / max(true_positive + false_negative, 1)
            label_false_positive_rate = false_positive / max(false_positive + true_negative, 1)
            predicted_positive_fraction = float(np.mean(predicted))
            truth_positive_fraction = float(np.mean(truth))
        lpips = None if self.skip_lpips else self.service.metrics_service.lpips(original, candidate.reconstruction)
        raw_bytes = original.width * original.height * 3
        if candidate.size > target_bytes:
            raise RuntimeError(f"Calibrated payload exceeds target: {candidate.size} > {target_bytes}")
        padding_bytes = target_bytes - candidate.size
        return {
            "sample_id": item.sample_id,
            "dataset": item.dataset,
            "split": item.split,
            "source_split": item.source_split,
            "geographic_group": item.geographic_group,
            "event_group": item.event_group,
            "label_source": item.label_source,
            "image_unit": item.image_unit,
            "cloud_fraction": item.cloud_fraction,
            "smoke_label_status": item.smoke_label_status,
            "method": method,
            "budget_index": budget_index,
            "budget_level": budget_level,
            "target_bytes": target_bytes,
            "encoded_bytes": candidate.size,
            "padding_bytes": padding_bytes,
            "actual_bytes": target_bytes,
            "byte_error": 0,
            "absolute_byte_error_percent": 0.0,
            "codec_underfill_percent": padding_bytes / max(target_bytes, 1) * 100.0,
            "common_budget_low": common_low,
            "common_budget_high": common_high,
            "bits_per_pixel": target_bytes * 8.0 / (original.width * original.height),
            "raw_compression_ratio": raw_bytes / max(target_bytes, 1),
            "bandwidth_saved_vs_raw_percent": (1.0 - target_bytes / raw_bytes) * 100.0,
            "codec_parameter": candidate.parameter,
            "kept_tokens": candidate.keep_count,
            "total_tokens": int(token_shape[0] * token_shape[1]),
            "sus": sus,
            "detector_retention": components.detector_retention,
            "object_retention": components.object_retention,
            "relevance_retention": components.relevance_retention,
            "region_preservation": components.region_preservation,
            "psnr": self.service.metrics_service.psnr(original, candidate.reconstruction),
            "ssim": self.service.metrics_service.ssim(original, candidate.reconstruction),
            "lpips": lpips,
            "label_dice": label_dice,
            "label_iou": label_iou,
            "label_precision": label_precision,
            "label_recall": label_recall,
            "label_false_positive_rate": label_false_positive_rate,
            "predicted_positive_fraction": predicted_positive_fraction,
            "truth_positive_fraction": truth_positive_fraction,
            "label_valid_fraction": label_valid_fraction,
            "detector_backend": before.backend,
        }

    def _calibrate(
        self,
        target: int,
        low: int,
        high: int,
        encoder: Callable[[int], EncodedCandidate],
        increasing: bool,
    ) -> EncodedCandidate:
        cache: dict[int, EncodedCandidate] = {}

        def get(value: int) -> EncodedCandidate:
            value = int(np.clip(value, low, high))
            if value not in cache:
                cache[value] = encoder(value)
            return cache[value]

        left, right = low, high
        while left <= right:
            middle = (left + right) // 2
            size = get(middle).size
            if size == target:
                break
            move_right = size < target if increasing else size > target
            if move_right:
                left = middle + 1
            else:
                right = middle - 1
        probes = {low, high, left, right, (left + right) // 2}
        for value in list(probes):
            probes.update(range(max(low, value - 3), min(high, value + 3) + 1))
        candidates = [get(value) for value in probes if low <= value <= high]
        within_budget = [candidate for candidate in candidates if candidate.size <= target]
        if not within_budget:
            raise RuntimeError(f"Codec cannot reach target budget of {target} bytes")
        return max(within_budget, key=lambda candidate: candidate.size)

    def _jpeg(self, image: Image.Image, quality: int) -> EncodedCandidate:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=int(quality), optimize=True)
        payload = buffer.getvalue()
        return EncodedCandidate(payload, Image.open(io.BytesIO(payload)).convert("RGB"), float(quality))

    def _jpeg2000(self, image: Image.Image, rate: int) -> EncodedCandidate:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG2000", quality_mode="rates", quality_layers=[float(rate)])
        payload = buffer.getvalue()
        return EncodedCandidate(payload, Image.open(io.BytesIO(payload)).convert("RGB"), float(rate))

    def _prepare_image(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        width, height = image.size
        side = min(width, height)
        left = (width - side) // 2
        top = (height - side) // 2
        image = image.crop((left, top, left + side, top + side))
        if image.size != (self.image_size, self.image_size):
            image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        return image

    def summarize(self, rows: list[dict[str, object]]) -> list[dict[str, object]]:
        summary: list[dict[str, object]] = []
        groups = sorted({(str(row["method"]), int(row["budget_index"])) for row in rows})
        for method, budget_index in groups:
            subset = [row for row in rows if row["method"] == method and row["budget_index"] == budget_index]
            record: dict[str, object] = {
                "method": method,
                "budget_index": budget_index,
                "budget_level": subset[0]["budget_level"],
                "n": len(subset),
            }
            for metric in (*METRICS, "bits_per_pixel", "raw_compression_ratio", "absolute_byte_error_percent", "codec_underfill_percent"):
                values = [float(row[metric]) for row in subset if row.get(metric) is not None and math.isfinite(float(row[metric]))]
                if not values:
                    continue
                arr = np.asarray(values, dtype="float64")
                low, high = bootstrap_mean_ci(arr, seed=self.seed + budget_index)
                record[f"{metric}_mean"] = float(arr.mean())
                record[f"{metric}_std"] = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
                record[f"{metric}_ci_low"] = low
                record[f"{metric}_ci_high"] = high
            summary.append(record)
        return summary

    def statistical_tests(self, rows: list[dict[str, object]]) -> list[dict[str, object]]:
        output: list[dict[str, object]] = []
        comparisons = ["jpeg", "jpeg2000", "vqvae_random", "vqvae_entropy"]
        if any(row.get("method") == "vqvae_fixed_utility" for row in rows):
            comparisons.append("vqvae_fixed_utility")
        for budget_index in sorted({int(row["budget_index"]) for row in rows}):
            budget_rows = [row for row in rows if int(row["budget_index"]) == budget_index]
            for baseline in comparisons:
                for metric in ("sus", "detector_retention", "psnr", "ssim", "lpips", "label_dice", "label_iou"):
                    paired = self._paired_values(budget_rows, baseline, "vqvae_utility", metric)
                    if len(paired) < 2:
                        continue
                    base = [pair[0] for pair in paired]
                    candidate = [pair[1] for pair in paired]
                    differences = np.asarray(candidate) - np.asarray(base)
                    t_result = self.validator.paired_t_test(base, candidate)
                    w_result = self.validator.wilcoxon(base, candidate)
                    ci_low, ci_high = bootstrap_mean_ci(differences, seed=self.seed + budget_index)
                    output.append(
                        {
                            "budget_index": budget_index,
                            "baseline": baseline,
                            "candidate": "vqvae_utility",
                            "metric": metric,
                            "n": len(paired),
                            "baseline_mean": float(np.mean(base)),
                            "candidate_mean": float(np.mean(candidate)),
                            "mean_paired_difference": float(differences.mean()),
                            "paired_t_statistic": t_result.statistic,
                            "paired_t_p_value": t_result.p_value,
                            "wilcoxon_statistic": w_result.statistic,
                            "wilcoxon_p_value": w_result.p_value,
                            "cohens_dz": t_result.effect_size,
                            "difference_ci_low": ci_low,
                            "difference_ci_high": ci_high,
                        }
                    )
        return output

    def _paired_values(
        self,
        rows: list[dict[str, object]],
        baseline: str,
        candidate: str,
        metric: str,
    ) -> list[tuple[float, float]]:
        by_key = {(str(row["sample_id"]), str(row["method"])): row for row in rows}
        samples = sorted({str(row["sample_id"]) for row in rows})
        output: list[tuple[float, float]] = []
        for sample in samples:
            base_row = by_key.get((sample, baseline))
            candidate_row = by_key.get((sample, candidate))
            if base_row is None or candidate_row is None:
                continue
            if base_row.get(metric) is None or candidate_row.get(metric) is None:
                continue
            base_value = float(base_row[metric])
            candidate_value = float(candidate_row[metric])
            if math.isfinite(base_value) and math.isfinite(candidate_value):
                output.append((base_value, candidate_value))
        return output

    def pareto_frontier(self, summary: list[dict[str, object]]) -> list[dict[str, object]]:
        candidates = [row for row in summary if row.get("actual_bytes_mean") is not None and row.get("sus_mean") is not None]
        output: list[dict[str, object]] = []
        for row in candidates:
            rate = float(row["actual_bytes_mean"])
            utility = float(row["sus_mean"])
            dominated = any(
                float(other["actual_bytes_mean"]) <= rate
                and float(other["sus_mean"]) >= utility
                and (float(other["actual_bytes_mean"]) < rate or float(other["sus_mean"]) > utility)
                for other in candidates
            )
            item = dict(row)
            item["pareto_optimal"] = not dominated
            output.append(item)
        return output

    def generate_figures(self, summary: list[dict[str, object]], pareto: list[dict[str, object]]) -> list[str]:
        try:
            import matplotlib.pyplot as plt
        except Exception:
            return []
        figure_dir = self.output_dir / "figures"
        figure_dir.mkdir(parents=True, exist_ok=True)
        outputs: list[str] = []
        for metric, ylabel, filename in (
            ("sus_mean", "Semantic Utility Score", "rate_vs_sus.png"),
            ("detector_retention_mean", "Detector retention", "rate_vs_detector_retention.png"),
            ("psnr_mean", "PSNR (dB)", "rate_vs_psnr.png"),
            ("ssim_mean", "SSIM", "rate_vs_ssim.png"),
            ("lpips_mean", "LPIPS (lower is better)", "rate_vs_lpips.png"),
            ("label_dice_mean", "Burn-scar label Dice", "rate_vs_label_dice.png"),
            ("label_iou_mean", "Burn-scar label IoU", "rate_vs_label_iou.png"),
        ):
            if not any(row.get(metric) is not None for row in summary):
                continue
            fig, ax = plt.subplots(figsize=(7.0, 4.6), dpi=220)
            for method in self.methods:
                points = sorted(
                    [row for row in summary if row["method"] == method and row.get(metric) is not None],
                    key=lambda row: float(row["actual_bytes_mean"]),
                )
                if not points:
                    continue
                ax.plot(
                    [float(row["actual_bytes_mean"]) for row in points],
                    [float(row[metric]) for row in points],
                    marker="o",
                    linewidth=1.8,
                    label=method.replace("_", " "),
                )
            ax.set_xlabel("Per-image byte budget (including unused capacity)")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.25)
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles, labels, fontsize=8)
            fig.tight_layout()
            path = figure_dir / filename
            fig.savefig(path, bbox_inches="tight")
            plt.close(fig)
            outputs.append(str(path))

        fig, ax = plt.subplots(figsize=(7.0, 4.6), dpi=220)
        for method in self.methods:
            points = sorted(
                [row for row in pareto if row["method"] == method],
                key=lambda row: float(row["actual_bytes_mean"]),
            )
            if points:
                ax.plot(
                    [float(row["actual_bytes_mean"]) for row in points],
                    [float(row["sus_mean"]) for row in points],
                    marker="o",
                    linewidth=1.2,
                    label=method.replace("_", " "),
                )
        frontier = sorted([row for row in pareto if row["pareto_optimal"]], key=lambda row: float(row["actual_bytes_mean"]))
        if frontier:
            ax.plot(
                [float(row["actual_bytes_mean"]) for row in frontier],
                [float(row["sus_mean"]) for row in frontier],
                color="black",
                linestyle="--",
                linewidth=2.0,
                label="Pareto frontier",
            )
        ax.set_xlabel("Per-image byte budget (including unused capacity)")
        ax.set_ylabel("Semantic Utility Score")
        ax.grid(True, alpha=0.25)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, fontsize=8)
        fig.tight_layout()
        path = figure_dir / "rate_utility_pareto.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        outputs.append(str(path))
        return outputs

    def write_report(
        self,
        metadata: dict[str, object],
        summary: list[dict[str, object]],
        statistics: list[dict[str, object]],
        confirmatory_summary: list[dict[str, object]],
        confirmatory_statistics: list[dict[str, object]],
        exclusions: list[dict[str, object]],
    ) -> None:
        detector_backends = sorted({str(row.get("detector_backend")) for row in self._read_csv(self.output_dir / "matched_rate_rows.csv")})
        lines = [
            "# Matched-Rate Satellite Compression Report",
            "",
            "## Experimental design",
            "",
            "JPEG, JPEG2000, random token selection, entropy token selection, and utility-aware token selection were calibrated per image to a common feasible byte interval. Low, middle, and high operating points were evaluated inside that interval. Each codec is constrained not to exceed the target; unused capacity is counted as hypothetical padding for equal-budget accounting. No physical packet-padding or downlink transmission was performed.",
            "",
            f"- Requested held-out items: {metadata['requested_items']}",
            f"- Completed held-out items: {metadata['completed_items']}",
            f"- Items also in the provider source-validation split: {metadata['confirmatory_items']}",
            f"- Excluded items: {metadata['excluded_items']}",
            f"- Standardized evaluation resolution: {self.image_size} x {self.image_size}",
            f"- Detector backend(s): {', '.join(detector_backends) if detector_backends else 'unknown'}",
            f"- LPIPS measured: {not self.skip_lpips}",
            "- `encoded_bytes` is the measured serialized codec stream; legacy column `actual_bytes` is the common target budget including unused capacity, not observed bytes transmitted.",
            "- VQ-VAE payloads include selected code indices and the serialized selection mask; the shared decoder checkpoint is assumed to be pre-deployed.",
            "",
            "## Provider source-validation overlap",
            "",
            "These rows are available only when the manifest records provider source-validation membership; an empty table does not invalidate the full-split results below.",
            "",
            markdown_table(confirmatory_summary, ["method", "budget_index", "n", "actual_bytes_mean", "encoded_bytes_mean", "codec_underfill_percent_mean", "sus_mean", "detector_retention_mean", "label_dice_mean", "label_iou_mean", "psnr_mean", "ssim_mean", "lpips_mean"]),
            "",
            "## Source-validation overlap paired tests",
            "",
            markdown_table(confirmatory_statistics, ["budget_index", "baseline", "candidate", "metric", "n", "mean_paired_difference", "paired_t_p_value", "wilcoxon_p_value", "cohens_dz", "difference_ci_low", "difference_ci_high"]),
            "",
            "## Full requested split results",
            "",
            markdown_table(summary, ["method", "budget_index", "n", "actual_bytes_mean", "encoded_bytes_mean", "codec_underfill_percent_mean", "sus_mean", "detector_retention_mean", "label_dice_mean", "label_iou_mean", "psnr_mean", "ssim_mean", "lpips_mean"]),
            "",
            "## Full requested split paired tests",
            "",
            markdown_table(statistics, ["budget_index", "baseline", "candidate", "metric", "n", "mean_paired_difference", "paired_t_p_value", "wilcoxon_p_value", "cohens_dz", "difference_ci_low", "difference_ci_high"]),
            "",
            "## Interpretation safeguards",
            "",
            "- Statistical tests are paired by source image and operating point.",
            "- Positive effect sizes mean utility-aware selection produced a larger metric value; for LPIPS, a negative difference is favourable.",
            "- SUS is detector-dependent and is reported alongside the detector backend.",
            "- The dataset integrity report must be cited with these results; patches are not treated as independent scenes.",
            "- Smoke-label coverage is reported as unavailable where no independent smoke annotation exists.",
            "",
            "## Exclusions",
            "",
            markdown_table(exclusions, ["sample_id", "image_path", "reason"]) if exclusions else "No exclusions.",
        ]
        (self.output_dir / "matched_rate_report.md").write_text("\n".join(lines), encoding="utf-8")

    def _write_csv(self, path: Path, rows: list[dict[str, object]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        fields = sorted({key for row in rows for key in row})
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def _read_csv(self, path: Path) -> list[dict[str, str]]:
        if not path.exists() or path.stat().st_size == 0:
            return []
        with path.open("r", newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))


def read_benchmark_manifest(path: Path, split: str = "test", limit: int | None = None) -> list[BenchmarkItem]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    items: list[BenchmarkItem] = []
    for row in rows:
        if split and row.get("split") != split:
            continue
        mask_text = row.get("mask_path", "").strip()
        valid_mask_text = row.get("valid_mask_path", "").strip()
        cloud_text = row.get("cloud_fraction", "").strip()
        items.append(
            BenchmarkItem(
                sample_id=row["sample_id"],
                image_path=Path(row["image_path"]),
                mask_path=Path(mask_text) if mask_text else None,
                valid_mask_path=Path(valid_mask_text) if valid_mask_text else None,
                dataset=row.get("dataset", "unknown"),
                split=row.get("split", "test"),
                source_split=row.get("source_split", "unknown"),
                geographic_group=row.get("geographic_group", "unknown"),
                event_group=row.get("event_group", "unknown"),
                label_source=row.get("label_source", "none"),
                image_unit=row.get("image_unit", "independent_tile"),
                cloud_fraction=float(cloud_text) if cloud_text else None,
                smoke_label_status=row.get("smoke_label_status", "unavailable"),
            )
        )
        if limit and len(items) >= limit:
            break
    return items


def bootstrap_mean_ci(values: Iterable[float], seed: int = 1234, samples: int = 2000) -> tuple[float, float]:
    array = np.asarray(list(values), dtype="float64")
    if array.size == 0:
        return float("nan"), float("nan")
    if array.size == 1:
        value = float(array[0])
        return value, value
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(samples, array.size))
    means = array[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def markdown_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    if not rows:
        return "No rows generated."
    available = [column for column in columns if any(column in row for row in rows)]
    lines = ["| " + " | ".join(available) + " |", "| " + " | ".join(["---"] * len(available)) + " |"]
    for row in rows:
        values = []
        for column in available:
            value = row.get(column, "")
            values.append(f"{value:.6g}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)
