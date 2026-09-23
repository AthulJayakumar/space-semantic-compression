"""Incremental VQ-VAE token stream for interrupted-contact experiments."""

from __future__ import annotations

import csv
import struct
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy import stats

from evaluation.clustered_matched_rate import bootstrap_ci
from evaluation.matched_rate import BenchmarkItem, MatchedRateBenchmark
from backend.utils.tensor_utils import image_to_tensor, tensor_to_image
from datasets.research_wildfire import load_rgb_image


HEADER = struct.Struct("<4sHHHH")
RECORD = struct.Struct("<HH")
MAGIC = b"SVQ1"
RETENTION = (0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.0)
METHODS = ("vqvae_random", "vqvae_entropy", "vqvae_utility")


def serialize_progressive(tokens: torch.Tensor, ranking: np.ndarray) -> bytes:
    """One header followed by independently decodable position/code records."""
    if tokens.ndim != 3 or tokens.shape[0] != 1:
        raise ValueError("Progressive research streams encode exactly one image at a time")
    codes = tokens.detach().cpu().numpy().reshape(-1)
    height, width = tokens.shape[-2:]
    total = height * width
    if total > 65535 or int(codes.max()) > 65535:
        raise ValueError("Current research stream supports at most 65535 positions and code IDs")
    if sorted(np.asarray(ranking).tolist()) != list(range(total)):
        raise ValueError("Ranking must contain every token position exactly once")
    values, counts = np.unique(codes, return_counts=True)
    fallback = int(values[np.argmax(counts)])
    chunks = [HEADER.pack(MAGIC, height, width, fallback, total)]
    chunks.extend(RECORD.pack(int(position), int(codes[position])) for position in ranking)
    return b"".join(chunks)


def decode_prefix(stream: bytes, byte_budget: int) -> tuple[torch.Tensor, int]:
    """Decode all complete token records received before contact termination."""
    if byte_budget < HEADER.size or len(stream) < HEADER.size:
        raise ValueError("Contact ended before the stream header was received")
    magic, height, width, fallback, total = HEADER.unpack_from(stream)
    if magic != MAGIC or total != height * width:
        raise ValueError("Invalid progressive stream header")
    count = min(total, max(0, (min(byte_budget, len(stream)) - HEADER.size) // RECORD.size))
    codes = np.full(total, fallback, dtype=np.int64)
    seen: set[int] = set()
    for index in range(count):
        position, value = RECORD.unpack_from(stream, HEADER.size + index * RECORD.size)
        if position >= total or position in seen:
            raise ValueError("Invalid or repeated token position")
        seen.add(position)
        codes[position] = value
    return torch.from_numpy(codes.reshape(1, height, width)), HEADER.size + RECORD.size * count


def evaluate_item(benchmark: MatchedRateBenchmark, item: BenchmarkItem) -> list[dict[str, object]]:
    service = benchmark.service
    original = benchmark._prepare_image(load_rgb_image(item.image_path))
    tensor = image_to_tensor(original, service.encoder_service.device, service.encoder_service.stride)
    with torch.inference_mode():
        tokens = service.encoder_service.encode(tensor)
    token_shape = tuple(tokens.shape[-2:])
    before = service._detect_mission_utility(original, token_shape, "wildfire_detection")
    semantic = service.semantic_service.analyze(original, token_shape)
    utility = benchmark.compose_utility_map(before.utility_map, semantic.importance_map)
    detail = service.semantic_service.detail_map(original, token_shape)
    rankings = benchmark._rankings(tokens, utility, detail, item.sample_id)
    rows: list[dict[str, object]] = []
    for method in METHODS:
        stream = serialize_progressive(tokens, rankings[method])
        for retention in RETENTION:
            count = max(1, int(round(tokens.numel() * retention)))
            budget = HEADER.size + RECORD.size * count
            received, used_bytes = decode_prefix(stream, budget)
            with torch.inference_mode():
                reconstruction = tensor_to_image(service.decoder_service.decode(received.to(tokens.device)))
            after = service._detect_mission_utility(reconstruction, token_shape, "wildfire_detection")
            sus, components = service.semantic_utility_metric.score_before_after(before, after)
            rows.append({
                "sample_id": item.sample_id,
                "event_group": item.event_group,
                "method": method,
                "retention": retention,
                "tokens_received": count,
                "cumulative_bytes": used_bytes,
                "sus": sus,
                "detector_retention": components.detector_retention,
                "psnr": service.metrics_service.psnr(original, reconstruction),
            })
    return rows


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_event: dict[tuple[str, float, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        event = str(row["event_group"])
        if event in ("", "unknown"):
            raise ValueError("Event grouping is required for clustered inference")
        by_event[(str(row["method"]), float(row["retention"]), event)].append(row)
    output: list[dict[str, object]] = []
    for method in METHODS:
        for retention in RETENTION:
            events = sorted(event for name, level, event in by_event if name == method and level == retention)
            if not events:
                continue
            record: dict[str, object] = {"method": method, "retention": retention, "n_events": len(events)}
            for metric in ("cumulative_bytes", "sus", "detector_retention", "psnr"):
                values = np.asarray([
                    np.mean([float(row[metric]) for row in by_event[(method, retention, event)]])
                    for event in events
                ])
                low, high = bootstrap_ci(values)
                record[f"{metric}_mean"] = float(values.mean())
                record[f"{metric}_ci_low"] = low
                record[f"{metric}_ci_high"] = high
            output.append(record)
    return output


def paired_comparisons(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Compare orderings after averaging correlated chips within source groups."""
    grouped: dict[tuple[str, float, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["method"]), float(row["retention"]), str(row["event_group"]))].append(row)
    output: list[dict[str, object]] = []
    for retention in RETENTION:
        for metric in ("sus", "detector_retention"):
            family: list[dict[str, object]] = []
            for baseline in ("vqvae_random", "vqvae_entropy"):
                events = sorted(
                    event for method, level, event in grouped
                    if method == "vqvae_utility" and level == retention
                    and (baseline, retention, event) in grouped
                )
                differences = np.asarray([
                    np.mean([float(row[metric]) for row in grouped[("vqvae_utility", retention, event)]])
                    - np.mean([float(row[metric]) for row in grouped[(baseline, retention, event)]])
                    for event in events
                ])
                if len(differences) < 2:
                    continue
                low, high = bootstrap_ci(differences, seed=20260922 + int(retention * 100))
                family.append({
                    "retention": retention,
                    "metric": metric,
                    "baseline": baseline,
                    "candidate": "vqvae_utility",
                    "n_events": len(events),
                    "mean_difference": float(differences.mean()),
                    "difference_ci_low": low,
                    "difference_ci_high": high,
                    "paired_t_p": float(stats.ttest_1samp(differences, 0).pvalue) if differences.std(ddof=1) else 1.0,
                    "wilcoxon_p": float(stats.wilcoxon(differences, zero_method="zsplit").pvalue),
                })
            for column in ("paired_t_p", "wilcoxon_p"):
                previous = 0.0
                for rank, record in enumerate(sorted(family, key=lambda item: float(item[column]))):
                    previous = max(previous, min(1.0, (len(family) - rank) * float(record[column])))
                    record[f"{column}_holm"] = previous
            output.extend(family)
    return output


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("No progressive-stream measurements")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
