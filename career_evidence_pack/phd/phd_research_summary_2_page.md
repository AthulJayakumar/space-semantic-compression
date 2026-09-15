# PhD Research Summary

## Proposed Title

Semantic Utility-Aware Compression for Wildfire-Centric Earth Observation Systems

## Research Problem

Earth Observation satellites generate more imagery than can always be transmitted efficiently under bandwidth, latency, power, and contact-window constraints. Conventional compression methods optimise for visual fidelity or distortion metrics such as PSNR and SSIM. In emergency monitoring missions, however, the most important question is not whether every pixel is reconstructed perfectly. The more important question is whether mission-critical information, such as wildfire activity, smoke, burn scars, or affected infrastructure, remains usable after compression and transmission.

This project studies semantic compression for wildfire-centric Earth Observation. It asks whether learned visual tokens can be prioritised according to mission utility so that bandwidth is spent first on information that matters for downstream wildfire interpretation.

## Core Hypothesis

Semantic utility-aware token prioritisation preserves wildfire-relevant Earth Observation information more effectively than conventional distortion-optimised compression under equivalent bandwidth constraints.

## Research Questions

1. Can wildfire-relevant regions in Earth Observation imagery be converted into a quantitative semantic utility signal?
2. Can learned VQ-VAE tokens be ranked and transmitted according to mission utility?
3. Does utility-aware token retention preserve detector-facing wildfire information better than random, entropy-only, or conventional compression baselines?
4. What trade-off exists between visual reconstruction quality and mission utility?
5. Can the method remain practical under satellite communication constraints?

## Methodology

The platform uses a modular semantic communication pipeline:

1. Input Earth Observation or wildfire image
2. Semantic utility analysis
3. VQ-VAE token encoding
4. Utility-aware token ranking
5. Bandwidth-constrained token retention
6. Reconstruction
7. Mission utility and image-quality evaluation

The current system evaluates:

- Semantic Utility Score
- detector retention
- PSNR
- SSIM
- LPIPS where practical
- compression ratio
- bandwidth saved
- statistical significance
- bootstrap confidence intervals

## Current Experimental Evidence

The strongest current Earth Observation result is a 500-patch Sentinel-2 benchmark comparing the original VQ-VAE with a regularized mixed-domain VQ-VAE.

| Model | SUS | Detector Retention | PSNR | SSIM | Compression Ratio | Bandwidth Saved |
|---|---:|---:|---:|---:|---:|---:|
| Original VQ-VAE | 80.470 | 0.8642 | 21.305 | 0.8170 | 69.76 | 98.56% |
| Regularized mixed-domain VQ-VAE | 81.159 | 0.8403 | 22.631 | 0.8354 | 68.60 | 98.53% |

The results show that satellite-domain fine-tuning improves PSNR and SSIM, but also causes a statistically significant detector-retention drop. This is a useful research finding: visual reconstruction and mission utility are not identical objectives. The outcome supports the need for utility-aware evaluation rather than relying only on conventional image-quality metrics.

## Main Contribution

The project contributes a research-grade prototype and evaluation framework for semantic compression in Earth Observation:

- a wildfire-centric semantic utility metric
- utility-aware token prioritisation
- satellite communication-oriented compression evaluation
- Sentinel-2 validation with statistical analysis
- evidence that reconstruction gains may conflict with detector retention
- a modular platform suitable for future onboard AI experiments

## Research Gap

Existing image compression research mainly optimises distortion, perceptual quality, or bitrate. Existing Earth Observation systems often prioritise acquisition, onboard detection, or downstream analytics. Few systems systematically evaluate mission-utility-aware semantic compression using learned token representations, downstream detector retention, and satellite communication constraints within one experimental framework.

## Why This Is PhD-Worthy

This project has a clear scientific tension: improving visual reconstruction does not automatically improve mission utility. That creates a strong PhD research direction around utility-preserving representation learning, semantic communication, and evaluation methods for Earth Observation.

The project is suitable for research groups in:

- Earth Observation
- onboard AI
- remote sensing
- neural compression
- semantic communication
- disaster monitoring
- satellite edge computing

## Four-Year PhD Direction

Year 1:
Formalise utility metrics, improve datasets, validate wildfire-focused semantic utility estimation.

Year 2:
Develop detector-retention-aware token selection and representation learning.

Year 3:
Evaluate under realistic satellite communication constraints and extend to multi-hazard Earth Observation validation.

Year 4:
Optimise for onboard deployment, publish final framework, and prepare thesis contributions.

## Target Venues

- IEEE Transactions on Geoscience and Remote Sensing
- ISPRS Journal of Photogrammetry and Remote Sensing
- IEEE JSTARS
- IEEE IGARSS
- IEEE ICIP workshops
- Remote Sensing journal
- ESA Phi-Lab or onboard AI workshops
