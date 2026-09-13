# Testing and Validation Report

Generated: 2026-09-13 22:27:00

Repository: `space-semantic-compression`

## 1. Purpose

This report documents the current software testing and research validation status of the public Space Semantic Compression repository. The goal is to show what has been tested, what evidence is included, what claims are supported, and what still requires local datasets or checkpoints.

The project studies **semantic utility-aware compression for wildfire-centric Earth Observation**. The practical question is:

> Can a satellite transmit the most mission-important learned image tokens first when bandwidth is severely limited?

## 2. Software Test Results

The public repository was tested with Python 3.11 using the configured project dependencies.

| Test Area | Command | Result |
|---|---|---|
| Unit tests | `python -m pytest` | **Passed: 12/12** |
| Syntax/bytecode compilation | `python -m compileall ...` | **Passed** |
| Key module imports | backend, frontend, semantic AI, token pruning, metrics, communication, statistics | **Passed** |

The Streamlit import emits normal bare-mode warnings when imported outside `streamlit run`. These warnings are expected and do not indicate a failure.

## 3. What Was Tested

The automated test suite checks the following components:

- PSNR and SSIM image metrics
- semantic region analysis
- token keep-mask construction
- utility-aware token pruning
- Semantic Utility Score behaviour
- wildfire utility detector output shape and range
- satellite transmission simulation
- VQ-VAE round-trip interface behaviour

This means the core research components are testable and currently functioning at the unit level.

## 4. What Is Not Fully Re-run From The Public Repo Alone

The public GitHub repository does not include raw datasets or large model checkpoints. This is intentional. DFire, FLAME, Sentinel-2/CEMS imagery, and trained checkpoints are large and may have separate licensing or storage requirements.

Therefore, full end-to-end benchmark reruns require:

```text
datasets/dfire/
datasets/flame/
datasets/sentinel2/
checkpoints/vqvae_s16k8.pt
```

The repository still includes dataset loaders, benchmark scripts, summary result tables, and reports so that reviewers can inspect the method and evidence.

## 5. Included Dataset-Level Evidence

The included cross-dataset summary validates the utility-aware 45% token-retention operating point.

| Dataset | Images | Method | SUS | Detector Retention | Bandwidth Saved | Compression Ratio |
|---|---:|---|---:|---:|---:|---:|
| dfire | 50 | vqvae_utility_45 | 83.14 | 0.849 | 95.10% | 25.30x |
| flame | 18 | vqvae_utility_45 | 65.84 | 0.834 | 96.27% | 55.51x |
| sentinel2 | 100 | vqvae_utility_45 | 79.52 | 0.741 | 99.35% | 155.77x |

Interpretation:

- DFire shows strong wildfire utility preservation.
- FLAME shows lower SUS, indicating dataset sensitivity and the need for broader validation.
- Sentinel-2/CEMS shows the strongest satellite communication result, with very high bandwidth savings and compression ratio.

## 6. Sentinel-2 Baseline Comparison

The 100-scene Sentinel-2/CEMS benchmark compares conventional and learned-token methods.

| Method | SUS | Detector Retention | PSNR | SSIM | LPIPS | Compression Ratio | Bandwidth Saved |
|---|---:|---:|---:|---:|---:|---:|---:|
| jpeg_quality_20 | 97.10 | 0.985 | 25.90 | 0.952 | 0.324 | 27.86x | 96.15% |
| jpeg_quality_40 | 98.32 | 0.994 | 27.28 | 0.970 | 0.211 | 16.32x | 93.54% |
| vqvae_full | 92.81 | 0.947 | 21.98 | 0.842 | 0.670 | 99.08x | 98.98% |
| vqvae_random_45 | 70.91 | 0.673 | 14.90 | 0.438 | 0.714 | 152.83x | 99.34% |
| vqvae_entropy_45 | 74.92 | 0.694 | 14.70 | 0.473 | 0.715 | 153.94x | 99.34% |
| vqvae_utility_45 | 79.52 | 0.741 | 15.62 | 0.627 | 0.717 | 155.77x | 99.35% |

Interpretation:

- JPEG remains strong for general reconstruction quality and detector retention.
- Full VQ-VAE preserves more utility than pruned token modes but uses more tokens.
- Utility-aware token selection outperforms random and entropy-only token selection under the same 45% token-retention regime.
- Utility-aware compression is most defensible as an extreme-bandwidth, mission-aware transmission mode, not as a universal JPEG replacement.

## 7. Statistical Validation

Selected statistical comparisons from the included significance table:

| Comparison | Metric | Cohen's d | Bootstrap CI Low | Bootstrap CI High | Interpretation |
|---|---|---:|---:|---:|---|
| vqvae_random_45_vs_vqvae_utility_45 | semantic_utility_score | 0.480 | 5.168 | 12.074 | Utility-aware higher |
| vqvae_entropy_45_vs_vqvae_utility_45 | semantic_utility_score | 0.488 | 2.670 | 6.472 | Utility-aware higher |
| vqvae_random_45_vs_vqvae_utility_45 | detector_retention | 0.344 | 0.030 | 0.104 | Utility-aware higher |
| vqvae_entropy_45_vs_vqvae_utility_45 | detector_retention | 0.378 | 0.022 | 0.070 | Utility-aware higher |
| jpeg_quality_20_vs_vqvae_utility_45 | compression_ratio | 5.783 | 123.366 | 132.088 | Utility-aware higher |
| jpeg_quality_20_vs_vqvae_utility_45 | semantic_utility_score | -1.223 | -20.256 | -14.835 | Utility-aware lower |

Interpretation:

- Utility-aware token ranking improves SUS over random token selection.
- Utility-aware token ranking improves SUS over entropy-only token selection.
- JPEG has stronger SUS at moderate compression settings.
- Utility-aware token compression provides much higher compression ratio than JPEG Q20 in the tested Sentinel-2 benchmark.

## 8. Retention Sweep Evidence

The included Sentinel-2 retention file contains `1000` rows. This corresponds to 100 scenes evaluated across multiple token-retention settings.

This supports operating-point analysis: lower token retention improves communication savings, while higher retention improves utility and detector preservation.

## 9. Main Supported Research Claims

Based on the tests and included validation tables, the following claims are currently supported:

1. The public codebase imports, compiles, and passes unit tests.
2. The project has a reproducible modular architecture for semantic compression research.
3. Utility-aware token selection preserves more wildfire utility than random token selection in the Sentinel-2 benchmark.
4. Utility-aware token selection preserves more wildfire utility than entropy-only token selection in the Sentinel-2 benchmark.
5. The system demonstrates strong bandwidth savings on DFire, FLAME, and Sentinel-2/CEMS summary benchmarks.
6. JPEG remains a strong baseline and should be treated honestly in publications.
7. The strongest research framing is mission-utility preservation under extreme satellite communication constraints.

## 10. Current Limitations

The following limitations should be stated clearly in supervisor outreach and papers:

- Public repo does not include raw datasets or model checkpoint binaries.
- Full benchmark reproduction requires local dataset setup.
- Sentinel-2 validation is currently 100 scenes, not yet 500+ scenes.
- JPEG2000 and CCSDS-style baselines should be added for stronger ESA/DLR positioning.
- Real Jetson or flight-like hardware testing is still future work.
- FLAME sample size is small, so FLAME results should be treated as preliminary.

## 11. Recommended Next Validation Steps

Priority order:

1. Add a small permissively licensed sample image and a lightweight demo mode.
2. Add a GitHub Actions workflow for unit tests.
3. Add JPEG2000 baseline comparison.
4. Expand Sentinel-2 benchmark to 500+ scenes.
5. Add FIRMS and burn-scar label alignment details.
6. Run edge benchmarks on real Jetson-class hardware if available.
7. Record a 2-3 minute demo video for supervisors and job applications.

## 12. Conclusion

The repository is now clean enough for public review and passes core software checks. The included evidence is sufficient for PhD supervisor outreach and a preliminary research portfolio. For peer-reviewed publication, the next important step is not more README polishing, but broader benchmark reproduction with local datasets, stronger baselines, and hardware validation.
