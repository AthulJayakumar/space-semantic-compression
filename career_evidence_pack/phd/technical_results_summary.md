# Technical Results Summary

## Project

CompressAI / Space Semantic Compression

## Research Claim Under Test

Mission-aware semantic token prioritisation can preserve wildfire-relevant Earth Observation information under bandwidth constraints better than treating all visual content equally.

## Implemented System Components

- FastAPI backend
- Streamlit dashboard
- VQ-VAE encoder and decoder
- utility-aware token selector
- wildfire and satellite utility analysis
- satellite bandwidth simulation
- benchmark runners
- statistical analysis
- publication-style reports

## Datasets Used So Far

- CEMS-HLS wildfire and burn-scar satellite imagery
- Sentinel-2 benchmark patches
- synthetic and local validation images for smoke tests

## Model Variants Tested

| Variant | Purpose | Result |
|---|---|---|
| Original VQ-VAE | Conservative baseline | Best current default for detector retention |
| CEMS-only fine-tuned VQ-VAE | Improve wildfire satellite reconstruction | Strong on CEMS-HLS, poor generalisation to Sentinel-2 |
| Mixed-domain VQ-VAE | Improve cross-domain satellite reconstruction | Improved Sentinel-2 reconstruction but detector retention risk |
| Regularized mixed-domain VQ-VAE | Preserve original behavior while improving reconstruction | Best reconstruction candidate, not yet default |
| Fixed mission selector | Conservative token prioritisation | Strong and stable baseline |
| Mask-supervised AI selector | Learned selector from wildfire masks | Worse than fixed selector when used alone |
| Hybrid selector | Combined fixed and learned signal | Small gains in some settings, not yet headline result |

## Key Finding 1: More Training Helps Reconstruction But Can Hurt Utility

On the 500-patch Sentinel-2 benchmark:

| Model | SUS | Detector Retention | PSNR | SSIM |
|---|---:|---:|---:|---:|
| Original VQ-VAE | 80.470 | 0.8642 | 21.305 | 0.8170 |
| Regularized mixed-domain VQ-VAE | 81.159 | 0.8403 | 22.631 | 0.8354 |

The regularized model improves PSNR by 1.326 dB and SSIM by 0.0184, but detector retention drops by 0.0239. This supports the central research argument that visual reconstruction and mission utility must be evaluated separately.

## Key Finding 2: CEMS-Only Fine-Tuning Over-Specialises

The CEMS-only checkpoint improved CEMS-HLS performance but degraded Sentinel-2 detector retention substantially. This shows that wildfire-specific satellite fine-tuning can overfit to the dataset distribution and must be tested cross-dataset.

## Key Finding 3: Original Model Remains Best Conservative Default

The original VQ-VAE remains the safest default for demos and public headline experiments because it preserves detector retention better on the larger Sentinel-2 validation.

## Key Finding 4: The Next Model Improvement Must Be Utility-Aware

The next scientifically meaningful improvement is not simply longer training or more data. The next objective should explicitly preserve detector retention or semantic utility during training.

Possible next experimental direction:

- original checkpoint teacher consistency
- reconstruction loss
- detector-retention validation gate
- utility-preservation proxy term
- cross-dataset early stopping

## Research Interpretation

This project has moved beyond a software demo. It now has a defensible scientific result:

> Compression models that look better under PSNR and SSIM may still be worse for mission-specific semantic utility.

That is a strong argument for wildfire-centric semantic communication research.
