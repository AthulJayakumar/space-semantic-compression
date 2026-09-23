# EcoFireBias training-only adaptation

**Decision: no-go for sealed testing.** The 18-event development set was reused after a predeclared two-epoch fine-tune. Its results are exploratory.

| Method | Burn SUS | Burn dNBR-proxy Dice | Burn PSNR | Mean serialized bytes |
|:--|--:|--:|--:|--:|
| JPEG2000 RDO | 91.99 | 0.455 | 28.77 | 1169.0 |
| Mixed-wire base VQ-VAE | 70.01 | 0.243 | 19.26 | 1124.2 |
| Adapted VQ-VAE | 85.03 | 0.417 | 19.95 | 1134.3 |

## Paired adapted-minus-base changes

- sus: +15.020; 95% paired-event bootstrap CI [+7.911, +22.737].
- label_dice: +0.174; 95% paired-event bootstrap CI [+0.078, +0.275].
- psnr: +0.689; 95% paired-event bootstrap CI [-0.040, +1.458].
- ssim: -0.016; 95% paired-event bootstrap CI [-0.048, +0.017].
- detector_retention: +0.163; 95% paired-event bootstrap CI [+0.088, +0.242].
- predicted_positive_fraction: +0.161; 95% paired-event bootstrap CI [-0.011, +0.311].

The frozen gate still fails: sus_mean_advantage, proxy_dice_mean_advantage, sus_ci_positive, proxy_dice_ci_positive.

Training used 600 distinct EcoFireBias train events with a 480/120 event-disjoint internal split. No gate-development, label-calibration, or test event entered this training manifest. The detector and token selector were unchanged. The development set is not independent after repeated consultation, the dNBR mask is a quantized proxy, and no sealed test image was scored.
