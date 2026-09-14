# Step 19: Regularized Mixed-Domain VQ-VAE Decision

## Question

Can the original VQ-VAE be improved with additional satellite data without losing the detector-friendly behavior that supports wildfire semantic utility?

## Change Tested

A mixed-domain VQ-VAE fine-tune was trained on CEMS-HLS wildfire imagery plus Sentinel-2 patches. Unlike the previous mixed-domain checkpoint, this run added original-model consistency regularization using the original VQ-VAE as a frozen teacher.

The goal was to preserve the original checkpoint's semantic behavior while improving satellite-domain reconstruction.

## Training Configuration

- Base checkpoint: `../checkpoints/vqvae_s16k8.pt`
- Teacher checkpoint: `../checkpoints/vqvae_s16k8.pt`
- Output checkpoint: `models/checkpoints/vqvae_s16k8_mixed_regularized_finetuned.pt`
- Training data: 439 CEMS-HLS wildfire satellite images plus 439 Sentinel-2 patches
- Training samples: 702
- Validation samples: 176
- Epochs: 3
- Batch size: 4
- Image size: 256
- Learning rate: 5e-5
- Codebook frozen: yes
- Teacher consistency weight: 0.25

## Training Result

| Epoch | Validation Loss | Validation Reconstruction Loss | Validation PSNR |
|---:|---:|---:|---:|
| 1 | 0.129553 | 0.121552 | 22.343 |
| 2 | 0.127986 | 0.118509 | 22.538 |
| 3 | 0.127908 | 0.117727 | 22.591 |

The regularized model improved validation reconstruction quality during training while remaining anchored to the original checkpoint.

## Cross-Dataset Benchmark

All checkpoints were evaluated with the same mission-utility token selector at 80% token retention on 80 CEMS-HLS images and 80 Sentinel-2 patches.

| Dataset | Checkpoint | SUS | Detector Retention | PSNR | SSIM | Compression Ratio | Bandwidth Saved |
|---|---|---:|---:|---:|---:|---:|---:|
| CEMS-HLS | original | 95.527 | 0.9863 | 18.782 | 0.8296 | 555.72 | 99.82% |
| CEMS-HLS | CEMS-only fine-tuned | 97.321 | 0.9834 | 19.250 | 0.8476 | 547.00 | 99.82% |
| CEMS-HLS | mixed-domain fine-tuned | 96.422 | 0.9623 | 18.887 | 0.8196 | 542.37 | 99.81% |
| CEMS-HLS | mixed-domain regularized | 96.955 | 0.9853 | 18.906 | 0.8153 | 543.83 | 99.81% |
| Sentinel-2 | original | 78.852 | 0.8122 | 23.564 | 0.8612 | 63.95 | 98.43% |
| Sentinel-2 | CEMS-only fine-tuned | 68.246 | 0.6367 | 23.333 | 0.8520 | 63.42 | 98.42% |
| Sentinel-2 | mixed-domain fine-tuned | 81.499 | 0.8038 | 25.255 | 0.8793 | 62.71 | 98.40% |
| Sentinel-2 | mixed-domain regularized | 80.682 | 0.8077 | 25.371 | 0.8802 | 62.54 | 98.40% |

## Statistical Interpretation

Compared with the original checkpoint:

- On CEMS-HLS, the regularized mixed-domain checkpoint improved SUS by +1.428 points with paired t-test p = 0.0299.
- On CEMS-HLS, detector retention changed by only -0.0010 with p = 0.8642, meaning the loss was not statistically meaningful in this benchmark.
- On Sentinel-2, the regularized mixed-domain checkpoint improved PSNR by +1.807 dB and SSIM by +0.0190, both strongly significant.
- On Sentinel-2, detector retention changed by -0.0045 with p = 0.8035, again not statistically meaningful.
- Sentinel-2 SUS improved by +1.830 points, but this was not statistically significant at the 80-image sample size.

## Decision

The regularized mixed-domain checkpoint is the best research candidate so far for improving the original VQ-VAE.

It gives the strongest Sentinel-2 reconstruction quality, keeps detector retention close to the original checkpoint, and avoids the large detector-retention collapse observed in the CEMS-only checkpoint on Sentinel-2.

However, the original checkpoint should remain the conservative default for public demo runs until the regularized model is validated on a larger Sentinel-2 split and qualitative reconstructions are reviewed. For the PhD evidence package, the regularized mixed-domain model should be reported as a promising improved checkpoint rather than as a fully promoted replacement.

## Next Recommended Step

Run a larger Sentinel-2 validation split with the original and regularized mixed-domain checkpoints only. If the regularized checkpoint keeps detector retention within approximately one percentage point of the original while maintaining the PSNR and SSIM gains, it can become the default satellite VQ-VAE checkpoint.
