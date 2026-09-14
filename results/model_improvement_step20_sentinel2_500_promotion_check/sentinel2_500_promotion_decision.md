# Step 20: Sentinel-2 500-Patch VQ-VAE Promotion Check

## Purpose

This experiment tested whether the regularized mixed-domain VQ-VAE checkpoint should replace the original VQ-VAE as the default satellite compression model.

The comparison keeps the semantic token selector fixed and changes only the VQ-VAE checkpoint. This isolates the effect of improving the encoder/decoder itself.

## Experimental Setup

- Dataset: Sentinel-2 benchmark patches
- Number of images: 500
- Token retention: 80%
- Baseline checkpoint: `../checkpoints/vqvae_s16k8.pt`
- Candidate checkpoint: `models/checkpoints/vqvae_s16k8_mixed_regularized_finetuned.pt`
- LPIPS: skipped for runtime
- Output rows: 1,000 paired image-level evaluations

## Aggregate Results

| Model | Images | SUS | Detector Retention | PSNR | SSIM | Compression Ratio | Bandwidth Saved |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original VQ-VAE | 500 | 80.470 | 0.8642 | 21.305 | 0.8170 | 69.76 | 98.56% |
| Regularized mixed-domain VQ-VAE | 500 | 81.159 | 0.8403 | 22.631 | 0.8354 | 68.60 | 98.53% |

## Paired Statistical Results

| Metric | Mean Difference | Paired t-test p | Wilcoxon p | Bootstrap 95% CI |
|---|---:|---:|---:|---:|
| SUS | +0.6895 | 0.1741 | 0.00436 | [-0.2347, 1.6611] |
| Detector Retention | -0.0239 | 0.000105 | 0.00281 | [-0.0347, -0.0120] |
| PSNR | +1.3261 dB | 9.91e-67 | 2.50e-64 | [1.2015, 1.4605] |
| SSIM | +0.0184 | 1.15e-18 | 4.02e-52 | [0.0144, 0.0220] |
| Compression Ratio | -1.1653 | 3.93e-116 | 1.18e-73 | [-1.2356, -1.0947] |
| Bandwidth Saved | -0.0245 percentage points | 9.57e-120 | 7.20e-74 | [-0.0260, -0.0231] |

## Interpretation

The regularized mixed-domain VQ-VAE produces a clear and statistically strong improvement in conventional reconstruction quality:

- PSNR improves by approximately 1.33 dB.
- SSIM improves by approximately 0.018.

However, the mission-oriented metrics do not fully support promotion:

- SUS increases by 0.69 points, but the paired t-test is not significant and the bootstrap confidence interval includes zero.
- Detector retention decreases by approximately 2.39 percentage points, and this decrease is statistically significant.
- Compression ratio and bandwidth savings are slightly lower for the candidate checkpoint.

This means the improved checkpoint reconstructs Sentinel-2 imagery more cleanly, but it slightly weakens the detector-facing semantic signal that the project is designed to preserve.

## Decision

Do not promote the regularized mixed-domain VQ-VAE as the default checkpoint yet.

The original VQ-VAE remains the safest default for public demos and headline mission-utility experiments because it preserves detector retention better on the 500-patch Sentinel-2 benchmark.

The regularized mixed-domain checkpoint should remain in the repository as an experimental reconstruction-improved checkpoint. It is useful evidence that additional satellite-domain training can improve PSNR and SSIM, but further training or loss balancing is needed before it can replace the original model for semantic utility-aware wildfire communication.

## Recommended Next Step

Train a detector-retention-aware VQ-VAE variant, not only a reconstruction-regularized one. The next training objective should explicitly preserve mission utility by adding either detector-retention validation gating or a small semantic utility proxy term, while continuing to use the original checkpoint as a teacher.
