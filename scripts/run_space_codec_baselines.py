"""Run JPEG2000 and CCSDS-style baselines on a local image benchmark.

The JPEG2000 baseline uses Pillow's JPEG2000 support. The CCSDS result here is
explicitly labelled as a **CCSDS-style wavelet proxy**, not a certified CCSDS
122/123 implementation. It provides a conservative research comparison until a
flight-standard external codec is integrated.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, features

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from metrics.semantic_utility import SemanticUtilityMetric
from semantic_ai.wildfire_detector import WildfireDetector


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class CodecResult:
    method: str
    reconstructed: Image.Image
    payload_bytes: int


def image_paths(input_dir: Path, limit: int | None) -> list[Path]:
    paths = sorted(path for path in input_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTS)
    return paths[:limit] if limit is not None else paths


def raw_rgb_bytes(image: Image.Image) -> int:
    return image.width * image.height * 3


def jpeg2000_roundtrip(image: Image.Image, rate: int) -> CodecResult:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG2000", quality_mode="rates", quality_layers=[rate])
    payload = buffer.getvalue()
    reconstructed = Image.open(io.BytesIO(payload)).convert("RGB")
    return CodecResult(method=f"jpeg2000_rate_{rate}", reconstructed=reconstructed, payload_bytes=len(payload))


def haar_forward_2d(channel: np.ndarray, levels: int) -> np.ndarray:
    coeff = channel.astype("float32").copy()
    h, w = coeff.shape
    for level in range(levels):
        hh = h >> level
        ww = w >> level
        if hh < 2 or ww < 2:
            break
        block = coeff[:hh, :ww]
        even_rows = block[0::2, :]
        odd_rows = block[1::2, :]
        low_rows = (even_rows + odd_rows) / 2.0
        high_rows = (even_rows - odd_rows) / 2.0
        even_cols = low_rows[:, 0::2]
        odd_cols = low_rows[:, 1::2]
        ll = (even_cols + odd_cols) / 2.0
        lh = (even_cols - odd_cols) / 2.0
        even_cols_h = high_rows[:, 0::2]
        odd_cols_h = high_rows[:, 1::2]
        hl = (even_cols_h + odd_cols_h) / 2.0
        hh_band = (even_cols_h - odd_cols_h) / 2.0
        rows = hh // 2
        cols = ww // 2
        coeff[:rows, :cols] = ll
        coeff[:rows, cols:ww] = lh
        coeff[rows:hh, :cols] = hl
        coeff[rows:hh, cols:ww] = hh_band
    return coeff


def haar_inverse_2d(coeff: np.ndarray, levels: int) -> np.ndarray:
    rec = coeff.astype("float32").copy()
    h, w = rec.shape
    max_level = 0
    for level in range(levels):
        if (h >> level) < 2 or (w >> level) < 2:
            break
        max_level = level
    for level in range(max_level, -1, -1):
        hh = h >> level
        ww = w >> level
        rows = hh // 2
        cols = ww // 2
        ll = rec[:rows, :cols]
        lh = rec[:rows, cols:ww]
        hl = rec[rows:hh, :cols]
        hh_band = rec[rows:hh, cols:ww]
        low_even = ll + lh
        low_odd = ll - lh
        high_even = hl + hh_band
        high_odd = hl - hh_band
        block = np.zeros((hh, ww), dtype="float32")
        block[0::2, 0::2] = low_even + high_even
        block[0::2, 1::2] = low_odd + high_odd
        block[1::2, 0::2] = low_even - high_even
        block[1::2, 1::2] = low_odd - high_odd
        rec[:hh, :ww] = block
    return rec


def ccsds_wavelet_proxy(image: Image.Image, quant_step: int, levels: int = 3) -> CodecResult:
    arr = np.asarray(image.convert("RGB")).astype("float32")
    rec_channels = []
    compressed_chunks = []
    for channel in range(3):
        coeff = haar_forward_2d(arr[..., channel], levels)
        quantized = np.round(coeff / quant_step).astype("int16")
        compressed_chunks.append(zlib.compress(quantized.tobytes(), level=9))
        recovered = haar_inverse_2d(quantized.astype("float32") * quant_step, levels)
        rec_channels.append(recovered)
    reconstructed = np.clip(np.stack(rec_channels, axis=-1), 0, 255).astype("uint8")
    payload_bytes = sum(len(chunk) for chunk in compressed_chunks) + 128
    return CodecResult(
        method=f"ccsds_wavelet_proxy_q{quant_step}",
        reconstructed=Image.fromarray(reconstructed),
        payload_bytes=payload_bytes,
    )


def psnr(original: Image.Image, reconstructed: Image.Image) -> float:
    a = np.asarray(original.convert("RGB")).astype("float32")
    b = np.asarray(reconstructed.convert("RGB").resize(original.size)).astype("float32")
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-12:
        return float("inf")
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def ssim(original: Image.Image, reconstructed: Image.Image) -> float:
    a = np.asarray(original.convert("L")).astype("float64")
    b = np.asarray(reconstructed.convert("L").resize(original.size)).astype("float64")
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    mu_a, mu_b = a.mean(), b.mean()
    sigma_a, sigma_b = a.var(), b.var()
    covariance = ((a - mu_a) * (b - mu_b)).mean()
    return float(((2 * mu_a * mu_b + c1) * (2 * covariance + c2)) / ((mu_a**2 + mu_b**2 + c1) * (sigma_a + sigma_b + c2)))


def mean_ci(values: list[float]) -> tuple[float, float, float, float]:
    arr = np.asarray(values, dtype="float64")
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    half = 1.96 * std / math.sqrt(max(arr.size, 1))
    return mean, std, mean - half, mean + half


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    metrics = ["psnr", "ssim", "semantic_utility_score", "detector_retention", "compression_ratio", "bandwidth_saved_percent"]
    methods = sorted({str(row["method"]) for row in rows})
    out = []
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        summary: dict[str, object] = {"method": method, "n_images": len(method_rows)}
        for metric in metrics:
            values = [float(row[metric]) for row in method_rows]
            mean, std, low, high = mean_ci(values)
            summary[f"{metric}_mean"] = round(mean, 6)
            summary[f"{metric}_std"] = round(std, 6)
            summary[f"{metric}_ci_low"] = round(low, 6)
            summary[f"{metric}_ci_high"] = round(high, 6)
        out.append(summary)
    return out


def markdown_report(path: Path, summary: list[dict[str, object]], n_images: int) -> None:
    lines = [
        "# JPEG2000 and CCSDS-Style Baseline Report",
        "",
        f"Images evaluated: **{n_images}**",
        "",
        "Important note: `ccsds_wavelet_proxy_*` is a wavelet/quantisation proxy inspired by CCSDS-style transform coding. It is **not** a certified CCSDS implementation.",
        "",
        "| Method | SUS | Detector Retention | PSNR | SSIM | Compression Ratio | Bandwidth Saved |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {method} | {sus:.2f} | {det:.3f} | {psnr:.2f} | {ssim:.3f} | {cr:.2f}x | {bw:.2f}% |".format(
                method=row["method"],
                sus=float(row["semantic_utility_score_mean"]),
                det=float(row["detector_retention_mean"]),
                psnr=float(row["psnr_mean"]),
                ssim=float(row["ssim_mean"]),
                cr=float(row["compression_ratio_mean"]),
                bw=float(row["bandwidth_saved_percent_mean"]),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- JPEG2000 provides a serious space-relevant conventional codec baseline.",
            "- The CCSDS-style proxy gives a first transform-coding reference but should be replaced with a certified CCSDS codec for ESA/DLR-grade claims.",
            "- These baselines should be reported alongside JPEG and VQ-VAE utility-aware token selection.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run JPEG2000 and CCSDS-style baselines.")
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/space_codec_baselines"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--jpeg2000-rates", type=int, nargs="*", default=[10, 20, 40])
    parser.add_argument("--ccsds-quant-steps", type=int, nargs="*", default=[12, 24, 48])
    args = parser.parse_args()

    if not features.check("jpg_2000"):
        raise SystemExit("Pillow was built without JPEG2000 support in this environment.")

    paths = image_paths(args.image_dir, args.limit)
    detector = WildfireDetector(save_visualizations=False)
    sus_metric = SemanticUtilityMetric()
    rows: list[dict[str, object]] = []

    for index, path in enumerate(paths):
        original = Image.open(path).convert("RGB")
        before = detector.detect_sentinel2(original)
        codecs: list[CodecResult] = []
        codecs.extend(jpeg2000_roundtrip(original, rate) for rate in args.jpeg2000_rates)
        codecs.extend(ccsds_wavelet_proxy(original, step) for step in args.ccsds_quant_steps)
        raw_bytes = raw_rgb_bytes(original)

        for codec in codecs:
            after = detector.detect_sentinel2(codec.reconstructed)
            sus, components = sus_metric.score_before_after(before, after)
            rows.append(
                {
                    "image": str(path),
                    "image_index": index,
                    "method": codec.method,
                    "payload_bytes": codec.payload_bytes,
                    "raw_rgb_bytes": raw_bytes,
                    "compression_ratio": raw_bytes / max(codec.payload_bytes, 1),
                    "bandwidth_saved_percent": 100.0 * (1.0 - codec.payload_bytes / max(raw_bytes, 1)),
                    "psnr": psnr(original, codec.reconstructed),
                    "ssim": ssim(original, codec.reconstructed),
                    "semantic_utility_score": sus,
                    "detector_retention": components.detector_retention,
                    "object_retention": components.object_retention,
                    "relevance_retention": components.relevance_retention,
                    "region_preservation": components.region_preservation,
                }
            )
        if (index + 1) % 50 == 0:
            print(f"processed={index + 1}/{len(paths)}")

    summary = aggregate(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "space_codec_baseline_rows.csv", rows)
    write_csv(args.output_dir / "space_codec_baseline_summary.csv", summary)
    markdown_report(args.output_dir / "space_codec_baseline_report.md", summary, len(paths))
    (args.output_dir / "space_codec_baseline_manifest.json").write_text(
        json.dumps({"image_dir": str(args.image_dir), "images": len(paths), "methods": sorted({row["method"] for row in rows})}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"images": len(paths), "summary": str(args.output_dir / "space_codec_baseline_summary.csv")}, indent=2))


if __name__ == "__main__":
    main()
