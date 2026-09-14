# Model Improvement Step 17: Cross-Dataset VQ-VAE Promotion Check

## Purpose

Step 16 showed that fine-tuning the VQ-VAE on CEMS-HLS improved reconstruction
and semantic utility on CEMS-HLS-like satellite burn-scar scenes. Step 17 checks
whether that improvement generalizes to another Sentinel-2 patch set.

The token selector is held fixed as `mission_utility`; only the VQ-VAE
checkpoint changes.

## Datasets

- CEMS-HLS burn-scar satellite scenes: 80 images
- Sentinel-2 patch benchmark: 80 images

## Result Summary

| Dataset | Checkpoint | SUS | Detector Retention | PSNR | SSIM |
|---|---|---:|---:|---:|---:|
| CEMS-HLS | Original | 95.527 | **0.9863** | 18.782 | 0.8296 |
| CEMS-HLS | Fine-tuned | **97.321** | 0.9834 | **19.250** | **0.8476** |
| Sentinel-2 patches | Original | **78.852** | **0.8122** | **23.564** | **0.8612** |
| Sentinel-2 patches | Fine-tuned | 68.246 | 0.6367 | 23.333 | 0.8520 |

## Statistical Result

On CEMS-HLS, the fine-tuned checkpoint improves:

- SUS: +1.794, p=0.0055
- PSNR: +0.468 dB, p=0.00044
- SSIM: +0.0181, p=0.0353

On Sentinel-2 patches, the fine-tuned checkpoint significantly reduces:

- SUS: -10.606, p=1.23e-6
- Detector retention: -0.175, p=2.55e-11
- SSIM: -0.0092, p=4.00e-5

## Decision

The fine-tuned VQ-VAE should **not** replace the original checkpoint as the
default model.

The original checkpoint remains the safest general default:

```text
Original VQ-VAE + fixed mission-utility selector
```

The CEMS-HLS fine-tuned checkpoint should be treated as a specialist research
checkpoint:

```text
CEMS-HLS-specialized VQ-VAE for burn-scar satellite validation
```

## Interpretation

The fine-tuning worked, but it over-specialized. It improved the dataset it was
trained on, while harming broader Sentinel-2 generalization. This is useful
research evidence because it shows the model can adapt to satellite data, but
also shows that domain-specific fine-tuning needs stronger regularization or a
more diverse training mix before promotion.

## Next Improvement

The next training run should use mixed-domain fine-tuning:

1. CEMS-HLS burn-scar samples
2. Sentinel-2 patch samples
3. Reconstruction regularization against the original checkpoint

Goal:

```text
improve CEMS-HLS without losing Sentinel-2 generalization
```
