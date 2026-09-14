# Model Improvement Step 16: Satellite Fine-Tuned VQ-VAE

## Purpose

This step tests whether the compression model itself can be improved by
fine-tuning the VQ-VAE encoder/decoder on real satellite wildfire imagery.
Previous steps improved token selection only. This experiment adapts the
learned image-token reconstruction model while keeping the existing checkpoint
format and codebook size compatible with the backend.

## Fine-Tuning Setup

- Dataset: CEMS-HLS satellite burn-scar imagery
- Paired image/mask samples available: 439
- Training samples: 351
- Validation samples: 88
- Base checkpoint: `../checkpoints/vqvae_s16k8.pt`
- Output checkpoint: `models/checkpoints/vqvae_s16k8_cems_hls_finetuned.pt`
- Epochs: 3
- Batch size: 4
- Image size: 256
- Learning rate: 5e-5
- Codebook: frozen

The codebook was frozen to preserve token-space compatibility. This makes the
experiment a conservative encoder/decoder adaptation rather than a full
retraining of the learned token vocabulary.

## Training Result

| Epoch | Validation Reconstruction Loss | Validation PSNR |
|---:|---:|---:|
| 1 | 0.153432 | 19.840 |
| 2 | 0.150779 | 19.917 |
| 3 | **0.150529** | **19.953** |

The validation reconstruction loss decreased across all three epochs, showing
that the VQ-VAE can adapt to satellite wildfire imagery without changing the
public architecture.

## End-to-End Compression Benchmark

The fine-tuned checkpoint was evaluated on 80 held-out CEMS-HLS satellite
scenes with the same token-retention setting used in Step 15.

| VQ-VAE Checkpoint | Selector | SUS | Detector Retention | PSNR | SSIM | Compression Ratio | Bandwidth Saved |
|---|---|---:|---:|---:|---:|---:|---:|
| Original | Fixed mission utility | 97.166 | **0.9942** | 18.343 | 0.8164 | **554.45x** | **99.82%** |
| Fine-tuned | Fixed mission utility | **98.293** | 0.9824 | **18.736** | **0.8298** | 545.14x | 99.82% |
| Fine-tuned | 50% fixed + 50% AI | 98.253 | 0.9819 | **18.775** | 0.8293 | 545.36x | 99.82% |
| Fine-tuned | Pure mask-supervised AI | 96.437 | 0.9609 | 18.307 | 0.8027 | 538.41x | 99.81% |

## Interpretation

Fine-tuning the VQ-VAE improves reconstruction quality and mean Semantic
Utility Score on the CEMS-HLS benchmark:

- SUS improves from 97.166 to 98.293.
- PSNR improves from 18.343 dB to 18.736 dB.
- SSIM improves from 0.8164 to 0.8298.

However, detector retention decreases from 0.9942 to 0.9824. This means the
fine-tuned model reconstructs satellite imagery better overall, but it slightly
changes features used by the current wildfire detector.

## Decision

The fine-tuned checkpoint is promising, but should remain experimental until it
is evaluated on DFire, FLAME, and Sentinel-2/CEMS splits beyond CEMS-HLS.

The best current production-safe default remains:

```text
Original VQ-VAE + fixed mission-utility selector
```

The best research candidate for the next paper-style result is:

```text
Fine-tuned VQ-VAE + fixed mission-utility selector
```

because it improves SUS, PSNR, and SSIM on satellite wildfire imagery while
preserving the same compression architecture.

## Next Step

Run a cross-dataset benchmark:

1. Original VQ-VAE + fixed selector
2. Fine-tuned VQ-VAE + fixed selector
3. Fine-tuned VQ-VAE + 50% hybrid selector

on DFire, FLAME, and Sentinel-2/CEMS. The fine-tuned checkpoint should only be
promoted if it improves satellite results without harming non-satellite wildfire
validation too severely.
