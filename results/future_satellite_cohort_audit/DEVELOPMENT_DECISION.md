# EcoFireBias training-development decision

**Status: no sealed test data scored.** This is an 18-event training-split development screen at 1,200 bytes per native 224x224 image.

| Candidate | Burn SUS | JPEG2000 burn SUS | Burn proxy Dice | JPEG2000 proxy Dice | Mean serialized bytes | Gate |
|:--|--:|--:|--:|--:|--:|:--|
| mixed_wire | 70.01 | 91.99 | 0.243 | 0.455 | 1124.2 | NO-GO |
| mask_aware | 64.42 | 91.99 | 0.162 | 0.455 | 1124.2 | NO-GO |

## Paired event differences versus JPEG2000

- mixed_wire sus: mean -21.983; 95% paired-event bootstrap CI [-31.291, -13.015].
- mixed_wire label_dice: mean -0.212; 95% paired-event bootstrap CI [-0.351, -0.059].
- mixed_wire predicted_positive_fraction: mean -0.193; 95% paired-event bootstrap CI [-0.382, +0.011].
- mask_aware sus: mean -27.574; 95% paired-event bootstrap CI [-36.904, -18.775].
- mask_aware label_dice: mean -0.293; 95% paired-event bootstrap CI [-0.429, -0.153].
- mask_aware predicted_positive_fraction: mean -0.278; 95% paired-event bootstrap CI [-0.472, -0.070].

## Descriptive quality on burn chips

| Method | PSNR (dB) | SSIM | Detector retention | Mean serialized bytes |
|:--|--:|--:|--:|--:|
| jpeg_rdo | 29.39 | 0.966 | 0.972 | 1124.4 |
| jpeg2000_rdo | 28.77 | 0.947 | 0.980 | 1169.0 |
| mixed_wire VQ-VAE | 19.26 | 0.640 | 0.726 | 1124.2 |
| mask_aware VQ-VAE | 18.49 | 0.644 | 0.634 | 1124.2 |

All five predeclared gate checks must pass. The dNBR mask is a quantized spectral proxy; some pixels with invalid original dNBR cannot be distinguished from low byte values. Two development pairs required UTM-grid reprojection, and uncovered edge pixels were excluded. This screen cannot support independent or publication-grade superiority claims.
