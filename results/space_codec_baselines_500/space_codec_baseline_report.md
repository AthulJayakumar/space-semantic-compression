# JPEG2000 and CCSDS-Style Baseline Report

Images evaluated: **500**

Important note: `ccsds_wavelet_proxy_*` is a wavelet/quantisation proxy inspired by CCSDS-style transform coding. It is **not** a certified CCSDS implementation.

| Method | SUS | Detector Retention | PSNR | SSIM | Compression Ratio | Bandwidth Saved |
|---|---:|---:|---:|---:|---:|---:|
| ccsds_wavelet_proxy_q24 | 82.74 | 0.881 | 25.35 | 0.934 | 23.93x | 93.34% |
| jpeg2000_rate_20 | 95.15 | 0.964 | 30.94 | 0.978 | 20.04x | 95.01% |

## Interpretation

- JPEG2000 provides a serious space-relevant conventional codec baseline.
- The CCSDS-style proxy gives a first transform-coding reference but should be replaced with a certified CCSDS codec for ESA/DLR-grade claims.
- These baselines should be reported alongside JPEG and VQ-VAE utility-aware token selection.
