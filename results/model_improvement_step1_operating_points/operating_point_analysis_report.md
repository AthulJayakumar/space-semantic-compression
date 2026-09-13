# Model Improvement Step 1: Operating Point Analysis

## Purpose

Before changing the model architecture, this analysis tests whether the current utility-aware VQ-VAE system performs better at a less aggressive token-retention setting.

The earlier headline result used an aggressive 45-50% retention regime. That setting produced very high compression but lost too much detector-visible wildfire evidence. This report evaluates 10-100% utility-aware retention on the Sentinel-2/CEMS 100-scene benchmark.

## Utility-Aware Retention Sweep

| Retention | SUS | Detector Retention | Compression Ratio | Bandwidth Saved | PSNR | SSIM |
|---:|---:|---:|---:|---:|---:|---:|
| 10% | 47.58 | 0.369 | 387.09x | 99.74% | 11.77 | 0.272 |
| 20% | 60.09 | 0.493 | 261.13x | 99.61% | 12.92 | 0.417 |
| 30% | 68.60 | 0.599 | 202.57x | 99.50% | 14.02 | 0.521 |
| 40% | 76.66 | 0.702 | 168.29x | 99.40% | 15.10 | 0.597 |
| 50% | 82.54 | 0.783 | 145.36x | 99.31% | 16.15 | 0.653 |
| 60% | 85.73 | 0.837 | 128.95x | 99.22% | 17.17 | 0.697 |
| 70% | 88.60 | 0.878 | 116.85x | 99.14% | 18.15 | 0.733 |
| 80% | 90.49 | 0.907 | 107.77x | 99.07% | 19.14 | 0.766 |
| 90% | 91.07 | 0.918 | 101.28x | 99.01% | 20.14 | 0.797 |
| 100% | 92.81 | 0.947 | 99.08x | 98.98% | 21.98 | 0.842 |

## Recommended Practical Operating Point

Recommended point: **80% utility-aware token retention**.

At this point:

- SUS: **90.49**
- Detector retention: **0.907**
- Compression ratio: **107.77x**
- Bandwidth saved: **99.07%**
- PSNR: **19.14 dB**
- SSIM: **0.766**

This is a better research operating point than 45-50% retention because it preserves much more mission utility while still maintaining over 100x compression.

## Baseline Context

| Baseline | Dataset/Setting | SUS | Detector Retention | Compression Ratio | Bandwidth Saved |
|---|---|---:|---:|---:|---:|
| jpeg_quality_20 | Sentinel-2 100-scene | 97.10 | 0.985 | 27.86x | 96.15% |
| jpeg_quality_40 | Sentinel-2 100-scene | 98.32 | 0.994 | 16.32x | 93.54% |
| vqvae_full | Sentinel-2 100-scene | 92.81 | 0.947 | 99.08x | 98.98% |
| ccsds_wavelet_proxy_q24 | Sentinel-2 500-patch | 82.74 | 0.881 | 23.93x | 93.34% |
| jpeg2000_rate_20 | Sentinel-2 500-patch | 95.15 | 0.964 | 20.04x | 95.01% |

The comparisons are not all perfectly apples-to-apples: JPEG and VQ-VAE values are from the 100-scene Sentinel-2 benchmark, while JPEG2000/CCSDS-style values are from the newer 500-patch benchmark. They are still useful for research positioning.

## Interpretation

The current model does not need an architecture change as the first improvement. The first improvement is to stop presenting the 45-50% setting as the primary operating point.

The most defensible current story is:

> Utility-aware token transmission at 80% retention preserves high wildfire utility while still achieving substantially higher compression than JPEG/JPEG2000-style baselines.

This improves the model story immediately:

- 50% retention: SUS 82.54, detector retention 0.783, compression 145.36x.
- 80% retention: SUS 90.49, detector retention 0.907, compression 107.77x.
- 100% retention: SUS 92.81, detector retention 0.947, compression 99.08x.

The 80% point gives most of the utility benefit of full VQ-VAE while preserving stronger compression than the full-token setting.

## Next Improvement Step

The next model improvement should be **token scoring**, not a full model rewrite. The current token score should be extended with edge/detail preservation so wildfire boundaries, smoke texture, and burn-scar structure are less likely to be pruned.
