# Model Improvement Step 18: Mixed-Domain VQ-VAE Fine-Tuning

## Purpose

Step 17 showed that CEMS-HLS-only fine-tuning improved CEMS-HLS burn-scar scenes
but over-specialized and harmed broader Sentinel-2 patch performance. Step 18
tests mixed-domain fine-tuning using both CEMS-HLS and Sentinel-2 patches.

## Training Setup

- Base checkpoint: `../checkpoints/vqvae_s16k8.pt`
- Output checkpoint: `models/checkpoints/vqvae_s16k8_mixed_cems_sentinel_finetuned.pt`
- CEMS-HLS paired samples: 439
- Sentinel-2 patch samples: 439
- Train samples: 702
- Validation samples: 176
- Epochs: 3
- Batch size: 4
- Image size: 256
- Learning rate: 5e-5
- Codebook: frozen

## Training Result

| Epoch | Validation Reconstruction Loss | Validation PSNR |
|---:|---:|---:|
| 1 | 0.118349 | 22.547 |
| 2 | 0.118094 | 22.559 |
| 3 | **0.117453** | **22.592** |

## Cross-Dataset Result

| Dataset | Checkpoint | SUS | Detector Retention | PSNR | SSIM |
|---|---|---:|---:|---:|---:|
| CEMS-HLS | Original | 95.527 | **0.9863** | 18.782 | 0.8296 |
| CEMS-HLS | CEMS-only fine-tune | **97.321** | 0.9834 | **19.250** | **0.8476** |
| CEMS-HLS | Mixed-domain fine-tune | 96.422 | 0.9623 | 18.887 | 0.8196 |
| Sentinel-2 patches | Original | 78.852 | **0.8122** | 23.564 | 0.8612 |
| Sentinel-2 patches | CEMS-only fine-tune | 68.246 | 0.6367 | 23.333 | 0.8520 |
| Sentinel-2 patches | Mixed-domain fine-tune | **81.499** | 0.8038 | **25.255** | **0.8793** |

## Statistical Interpretation

Against the original checkpoint, the mixed-domain checkpoint improves
Sentinel-2 patches:

- SUS: +2.648, p=0.0156
- PSNR: +1.691 dB, p=9.04e-12
- SSIM: +0.0181, p=1.65e-7

However, on CEMS-HLS the mixed-domain checkpoint does not beat the CEMS-only
specialist checkpoint. It also reduces detector retention on CEMS-HLS relative
to the original checkpoint.

## Decision

The mixed-domain checkpoint is a better general satellite research candidate
than the CEMS-only checkpoint.

Recommended checkpoint roles:

```text
Default/demo-safe:
Original VQ-VAE + fixed mission selector

Burn-scar specialist:
CEMS-HLS fine-tuned VQ-VAE + fixed mission selector

General satellite research candidate:
Mixed-domain VQ-VAE + fixed mission selector
```

The mixed-domain checkpoint should not yet replace the original default because
detector retention is still slightly lower. It is, however, the best candidate
for the next research round because it improves broader Sentinel-2 utility and
reconstruction quality without the severe over-specialization seen in the
CEMS-only fine-tune.

## Next Step

Run detector-aware fine-tuning or distillation regularization:

```text
reconstruction loss
+ original-checkpoint reconstruction consistency
+ detector-retention preservation
```

The goal is to preserve the mixed-domain gains in PSNR, SSIM, and SUS while
recovering detector retention.
