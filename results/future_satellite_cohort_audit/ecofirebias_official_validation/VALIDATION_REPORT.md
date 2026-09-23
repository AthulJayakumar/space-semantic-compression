# EcoFireBias official-validation comparison

**Status:** one frozen, previously unscored official-validation comparison; sealed test remains unscored. The earlier 18-event development gate failed and is not superseded.

60 event pairs (60 burn, 60 matched negative images), native 224 x 224, maximum 1,200 serialized bytes per image. Selection, checkpoints, methods, and quantized dNBR >85 proxy rule were frozen before model scoring. JPEG and JPEG2000 are distortion-optimized under the same ceiling; controls are byte-identical between checkpoint runs.

| Method | Burn SUS | Burn proxy Dice | Burn PSNR (dB) | Burn SSIM | Detector retention | Negative predicted-positive area | Mean serialized bytes |
|:--|--:|--:|--:|--:|--:|--:|--:|
| JPEG | 92.72 | 0.519 | 27.97 | 0.941 | 0.973 | 0.509 | 1116.7 |
| JPEG2000 RDO | 95.12 | 0.533 | 27.18 | 0.913 | 0.985 | 0.509 | 1177.5 |
| Base VQ-VAE | 70.85 | 0.286 | 19.87 | 0.644 | 0.728 | 0.225 | 1140.8 |
| Adapted VQ-VAE | 83.28 | 0.469 | 20.78 | 0.636 | 0.869 | 0.509 | 1155.4 |

## Event-paired adapted VQ-VAE differences

95% percentile bootstrap confidence intervals, 10,000 paired-event resamples; burn metrics use one burn image per event, negative area uses one negative image per event.

### Versus Base VQ-VAE

- sus: +12.429 [95% CI +8.736, +16.306].
- label_dice: +0.183 [95% CI +0.122, +0.245].
- psnr: +0.909 [95% CI +0.207, +1.652].
- ssim: -0.008 [95% CI -0.045, +0.023].
- detector_retention: +0.141 [95% CI +0.100, +0.181].
- predicted_positive_fraction: +0.284 [95% CI +0.202, +0.365].

### Versus JPEG2000 RDO

- sus: -11.846 [95% CI -15.136, -8.723].
- label_dice: -0.064 [95% CI -0.113, -0.018].
- psnr: -6.402 [95% CI -7.242, -5.634].
- ssim: -0.277 [95% CI -0.324, -0.234].
- detector_retention: -0.116 [95% CI -0.152, -0.083].
- predicted_positive_fraction: -0.000 [95% CI -0.060, +0.057].

## Interpretation and limits

This is a validation-split check, not a replacement for the predeclared failed development gate. All 60 validation events are distinct from selected train, internal-validation, label-calibration, reused-development and sealed-test events, but every selected validation event is in a country already represented in selected training. Event identifiers are patch-derived and do not guarantee scene or spatial-footprint independence. Labels are quantized dNBR spectral proxies, not manually verified burn-scar ground truth; original dNBR invalid pixels are not fully recoverable from the quantized export. Five images required grid reprojection; uncovered edge pixels were excluded. SUS and detector retention share a detector and are not independent evidence. Classical encoders choose highest original-image PSNR among predefined settings, using the original at the sender; selection overhead and encoder compute are not included in wire bytes. No LPIPS was run in this frozen comparison. No publication-grade superiority claim or sealed-test opening is justified.
