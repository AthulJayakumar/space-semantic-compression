# PhD Research Proposal

## Semantic Utility-Aware Adaptive Communication for Wildfire-Centric Earth Observation Systems Under Resource Constraints

**Research Area:** Artificial Intelligence, Earth Observation, Semantic Communication, Learned Image Compression, Satellite Communications  
**Primary Use Case:** Wildfire detection and wildfire response support  
**Secondary Validation Use Cases:** Flood detection and ship detection  
**Research Platform:** CompressAI  
**Prepared:** 2026-06-01  

---

## Abstract

Earth observation systems are increasingly used for climate monitoring, wildfire detection, disaster response, and environmental protection. At the same time, many satellite and edge observation platforms remain constrained by limited downlink bandwidth, intermittent contact windows, high latency, packet loss, restricted onboard computation, and limited energy availability. These constraints are particularly relevant during wildfire events, where timely transmission of mission-relevant information can support early warning, situational awareness, and emergency response.

This PhD proposal investigates **semantic utility-aware adaptive communication** for resource-constrained Earth observation systems, with wildfire detection as the primary research use case. The proposed framework studies how learned image tokens can be prioritized and transmitted according to wildfire-relevant semantic utility rather than general pixel-level distortion alone. The system combines VQ-VAE-based discrete image tokenization, wildfire utility estimation, semantic utility scoring, adaptive token selection, satellite communication simulation, and energy-aware evaluation.

The central hypothesis is that **semantic utility-aware token prioritization can preserve downstream wildfire-relevant Earth observation utility more effectively than conventional distortion-optimized compression under equivalent bandwidth constraints**.

The project builds on an existing CompressAI prototype that already implements VQ-VAE tokenization, Semantic Utility Score (SUS), utility-aware token pruning, satellite transmission simulation, energy-aware objective tracking, benchmarking, FastAPI services, and a Streamlit dashboard. The proposed PhD will transform this operational prototype into a rigorous research programme through dataset-level evaluation, statistical validation, trained wildfire utility models, retention-sweep experiments, ablation studies, and edge-deployment analysis.

The project is deliberately focused on wildfire detection as the primary mission. Flood detection and ship detection are included only as secondary validation cases to test whether the framework can generalize to related Earth observation tasks without broadening the central research scope.

---

## 1. Introduction and Motivation

Wildfires are increasing in frequency, intensity, and societal impact due to climate change, prolonged drought, land-use change, and extreme weather conditions. Rapid wildfire monitoring is important for:

- early detection,
- emergency response,
- evacuation planning,
- fire spread monitoring,
- burned-area assessment,
- environmental protection,
- climate resilience,
- protection of infrastructure and human settlements.

Satellite and airborne Earth observation systems provide wide-area coverage that is valuable for wildfire monitoring. However, image acquisition capacity often exceeds communication capacity. This is especially true for small satellites, CubeSats, remote sensing constellations, and edge-deployed observation systems operating with limited downlink windows and power budgets.

Conventional image compression methods reduce data volume, but they generally do not distinguish between mission-critical and mission-irrelevant regions. For example, an image region containing active fire or smoke may be more important than homogeneous terrain, cloud background, or visually textured but irrelevant areas. A compression system optimized only for pixel distortion may allocate bits to visually complex regions while failing to preserve the information most relevant to wildfire response.

This proposal therefore studies a mission-aware approach to Earth observation communication. The aim is not to replace all conventional compression methods, but to investigate whether wildfire-relevant semantic information can be preserved more efficiently under resource constraints by using learned token representations and semantic utility-aware token selection.

---

## 2. Problem Statement

The research problem addressed in this PhD is:

> How can Earth observation imagery be adaptively compressed and transmitted so that wildfire-relevant semantic information is preserved under bandwidth, latency, compute, and energy constraints?

The problem arises from the interaction of three limitations in current systems.

First, traditional codecs such as JPEG, JPEG2000, and CCSDS image compression standards are primarily signal- or distortion-oriented. They are mature and operationally important, but they do not explicitly optimize wildfire detection utility.

Second, learned image compression methods have improved rate-distortion performance, yet many remain task-agnostic. They are typically optimized to reconstruct images, not to preserve mission-specific downstream information.

Third, semantic communication research has often remained abstract or focused on generic communication tasks. There remains a need for practical Earth observation systems that integrate learned visual tokens, satellite communication constraints, energy-aware evaluation, and downstream mission utility.

This PhD addresses these limitations through a wildfire-centric semantic communication framework.

---

## 3. Research Hypothesis

The primary hypothesis is:

> **Semantic utility-aware token prioritization preserves wildfire-relevant Earth observation information more effectively than conventional distortion-optimized compression under equivalent bandwidth constraints.**

This hypothesis will be evaluated through controlled comparisons between:

- JPEG baselines,
- full-token VQ-VAE transmission,
- non-utility-aware learned compression,
- semantic utility-aware VQ-VAE token transmission,
- energy-aware adaptive token selection.

---

## 4. Research Questions

### RQ1
How can wildfire-relevant semantic utility be estimated from Earth observation imagery in a computationally efficient way suitable for resource-constrained platforms?

### RQ2
How can discrete learned image tokens be prioritized using semantic utility, token entropy, bandwidth cost, and energy cost?

### RQ3
Does utility-aware token transmission preserve wildfire detection utility better than classical and learned compression baselines at comparable bitrate?

### RQ4
What tradeoffs arise between rate, distortion, semantic utility, transmission latency, and energy consumption?

### RQ5
How sensitive is the proposed Semantic Utility Score to its weighting assumptions, and can these assumptions be validated through ablation and sensitivity analysis?

### RQ6
To what extent can the proposed framework generalize to secondary Earth observation missions such as flood detection and ship detection?

### RQ7
Can semantic utility-aware compression be executed efficiently on edge or satellite-class hardware?

### 4.1 Research Question Traceability

| Research Question | Methodology | Evaluation Strategy | Measurable Outcome |
|---|---|---|---|
| RQ1 | Wildfire utility estimation using detector confidence, relevance maps, and utility maps | Compare utility maps against wildfire labels, active-fire references, and downstream detector retention | Utility-map quality, detector retention, SUS correlation with downstream wildfire performance |
| RQ2 | VQ-VAE tokenization and utility-aware token scoring | Token retention sweeps and comparison with random, entropy-only, and saliency-only token selection | SUS, PSNR, SSIM, LPIPS, compression ratio, bandwidth saved |
| RQ3 | Utility-aware semantic token transmission | Matched-bitrate comparison against JPEG, full-token VQ-VAE, and non-utility-aware learned baselines | Downstream wildfire detection retention, SUS, statistical significance |
| RQ4 | Rate-distortion-energy-utility objective | Multi-objective analysis across retention levels and link budgets | Rate-utility curves, energy-utility curves, objective values |
| RQ5 | SUS sensitivity and ablation experiments | Vary SUS weights and remove individual SUS components | Sensitivity curves, ablation tables, correlation with downstream task metrics |
| RQ6 | Secondary validation on flood and ship tasks | Reuse the same pipeline with mission-specific utility maps | Generalization performance relative to wildfire experiments, reported separately |
| RQ7 | Edge deployment profiling | CPU, CUDA, ONNX, and Jetson-class experiments where hardware is available | Latency, memory, throughput, energy estimate, feasibility limits |

---

## 5. Research Scope

The PhD is centered on one primary mission:

```text
Primary mission: wildfire_detection
```

Wildfire detection is selected because it has clear societal relevance, climate relevance, and operational importance for Earth observation. The mission connects directly to disaster response, environmental monitoring, and autonomous satellite-based alerting.

Two secondary missions are included only for validation:

```text
Secondary validation missions:
- flood_detection
- ship_detection
```

These secondary missions will be used to test whether the framework is adaptable, but they will not become independent research directions. The algorithmic development, evaluation design, thesis narrative, and publication strategy remain wildfire-centric.

---

## 6. Literature Review

### 6.1 Traditional Image Compression

Traditional image compression methods remain essential for visual data transmission. JPEG uses block-based discrete cosine transforms and quantization. It is efficient, widely supported, and still a relevant baseline. JPEG2000 uses wavelet coding and supports progressive transmission, making it more suitable for some remote sensing and archival applications. CCSDS image compression standards are directly relevant to space missions because they address operational requirements for spacecraft image data compression.

However, these methods are primarily designed around signal reconstruction and rate-distortion tradeoffs. They generally do not account for whether an image region is relevant to a downstream Earth observation task. In wildfire monitoring, this can be limiting because the most visually complex regions are not always the most operationally important, and the most important regions may occupy only a small part of an image.

### 6.2 Learned Image Compression

Learned image compression has advanced through autoencoders, variational image compression, entropy models, hyperpriors, autoregressive priors, and neural codecs. Ballé et al. demonstrated end-to-end optimized image compression and variational approaches to learned compression. Minnen et al. introduced joint autoregressive and hierarchical priors, improving entropy modeling. Cheng et al. advanced learned compression with discretized Gaussian mixture likelihoods and attention modules. The CompressAI framework has provided a reproducible implementation environment for neural compression research.

Most learned compression methods optimize rate-distortion objectives. Although these methods may outperform classical codecs in some conditions, they do not usually optimize task-specific Earth observation utility. For wildfire monitoring, a rate-distortion objective alone may not preserve fire, smoke, or burned-area evidence optimally.

### 6.3 VQ-VAE and Discrete Visual Tokens

The VQ-VAE architecture introduced by van den Oord et al. learns discrete latent representations through vector quantization. Discrete tokens are useful for semantic communication because they can be ranked, pruned, packetized, entropy-analyzed, progressively transmitted, and potentially modeled by sequence models.

For this project, VQ-VAE tokenization provides a practical bridge between learned compression and communication systems. Instead of transmitting pixels or continuous latent tensors, the system transmits a spatial grid of discrete visual tokens. This makes adaptive token selection feasible under bandwidth constraints.

### 6.4 Semantic Communications

Semantic communication aims to transmit meaning or task-relevant information rather than exact bit-level reconstructions. Recent work on task-oriented communication, AI-native communication systems, and semantic communication for future 6G networks suggests that communication systems can be optimized for downstream task success rather than symbol fidelity alone.

However, much of the semantic communication literature remains theoretical, simulation-based, or focused on generic classification tasks. There is still limited work connecting semantic communication to practical Earth observation workflows with learned visual tokens, satellite communication constraints, and downstream geospatial analytics.

### 6.5 Earth Observation AI

Deep learning is widely used in Earth observation for land-cover classification, object detection, disaster monitoring, burned-area assessment, wildfire detection, flood mapping, ship detection, and infrastructure monitoring. Datasets such as EuroSAT, SpaceNet, and DeepGlobe have supported research in remote sensing classification and segmentation.

Wildfire monitoring uses satellite products such as MODIS active fire products, VIIRS active fire data, FIRMS, Sentinel-2 imagery, and burned-area products. These sources support active fire detection, smoke monitoring, and post-fire assessment. However, many Earth observation AI pipelines assume that imagery is already available on the ground, rather than considering how limited satellite downlink affects what imagery can be transmitted.

### 6.6 Edge AI for Space Systems

Edge AI for space systems investigates onboard inference, CubeSat autonomy, embedded visual processing, and resource-constrained satellite intelligence. These systems are constrained by compute, energy, memory, thermal design, and communication windows. Onboard AI can reduce downlink load by filtering or prioritizing data, but it must be efficient and robust.

This PhD is positioned at the intersection of edge AI, learned compression, and semantic satellite communication.

### 6.7 Strengthened Research Gap

The current literature lacks systems that simultaneously integrate:

- learned discrete token compression,
- wildfire-centric semantic utility estimation,
- adaptive token transmission,
- energy-aware communication,
- satellite communication constraints,
- downstream Earth observation evaluation.

Classical codecs provide mature compression but lack mission awareness. Learned compression provides powerful rate-distortion optimization but generally lacks explicit wildfire utility modeling. Semantic communication provides the theoretical motivation for meaning-oriented transmission but often lacks complete Earth observation implementations. Edge AI for space systems motivates onboard intelligence but does not by itself define how semantic image tokens should be selected for transmission.

This gap motivates a focused research programme on semantic utility-aware adaptive communication for wildfire-centric Earth observation.

---

## 7. Datasets

### 7.1 Primary Datasets

#### Sentinel-2 Wildfire Imagery

Sentinel-2 provides multispectral Earth observation imagery with spatial resolutions suitable for land-cover, vegetation, burn scar, and wildfire-related analysis. Sentinel-2 imagery will support wildfire utility estimation, burned-area context, and evaluation of compression effects on wildfire-relevant regions.

**Purpose:** primary image source for wildfire-centric semantic utility experiments.  
**Relevance:** high-resolution multispectral Earth observation imagery relevant to fire monitoring and post-fire assessment.  
**Use in this PhD:** utility-map generation, retention-sweep evaluation, downstream wildfire-region preservation.

#### MODIS Active Fire Products

MODIS active fire products provide global active fire detections and thermal anomaly information. They are widely used in environmental monitoring and fire research.

**Purpose:** support active fire labels and validation of wildfire relevance.  
**Relevance:** established satellite fire product with global coverage.  
**Use in this PhD:** wildfire event localization, label support, comparison with semantic utility maps.

#### FIRMS Wildfire Data

NASA FIRMS distributes near-real-time active fire data derived from satellite sensors such as MODIS and VIIRS.

**Purpose:** event-level wildfire reference data.  
**Relevance:** operationally used for fire monitoring and response.  
**Use in this PhD:** aligning Earth observation images with active fire detections and evaluating mission relevance.

### 7.2 Secondary Validation Datasets

#### EuroSAT

EuroSAT is a Sentinel-2 land-use and land-cover classification dataset. It provides a controlled benchmark for remote sensing representation learning.

**Purpose:** generalization testing beyond wildfire scenes.  
**Relevance:** standard Earth observation classification dataset.  
**Use in this PhD:** testing whether learned token compression preserves broader land-cover semantics.

#### SpaceNet

SpaceNet provides high-resolution satellite imagery and labels for tasks such as building footprint extraction and road network detection.

**Purpose:** secondary validation on infrastructure-heavy Earth observation scenes.  
**Relevance:** supports evaluation of object and infrastructure preservation.  
**Use in this PhD:** testing whether semantic token selection preserves structured human-made features.

#### DeepGlobe

DeepGlobe includes remote sensing datasets for land-cover classification, road extraction, and building detection.

**Purpose:** segmentation-oriented validation.  
**Relevance:** widely used for remote sensing semantic segmentation.  
**Use in this PhD:** testing semantic preservation under segmentation-like downstream tasks.

### 7.3 Dataset Roles

The primary wildfire datasets support:

- wildfire utility estimation,
- downstream wildfire evaluation,
- event-level validation,
- semantic utility scoring.

The secondary datasets support:

- generalization testing,
- non-wildfire Earth observation validation,
- analysis of method robustness across scene types.

---

## 8. Research Novelty

### Novelty 1: Semantic Utility Score

The Semantic Utility Score evaluates how much mission-relevant information is preserved after compression. Unlike PSNR or SSIM, it is intended to measure wildfire-relevant preservation rather than general image similarity.

### Novelty 2: Utility-Aware Token Prioritization

The project introduces adaptive token selection based on utility, entropy, and cost:

```text
token_score = alpha * utility + beta * entropy - gamma * cost
```

This differs from standard learned compression because token transmission is guided by wildfire mission relevance.

### Novelty 3: Joint Rate-Distortion-Energy-Utility Optimization

The evaluation framework uses:

```text
L = lambda1 * Rate + lambda2 * Distortion + lambda3 * Energy - lambda4 * Utility
```

This explicitly includes communication efficiency, image quality, energy use, and mission utility.

### Novelty 4: Mission-Aware Earth Observation Communication

The framework grounds semantic communication in a concrete wildfire Earth observation task rather than a generic semantic transmission benchmark.

---

## 9. Scientific Justification of SUS Weights

The initial SUS formulation is:

```text
SUS =
0.4 * detector_retention
+ 0.3 * object_retention
+ 0.2 * relevance_retention
+ 0.1 * region_preservation
```

Each component is normalized to the interval `[0, 1]`, and the weights sum to `1.0`, so SUS is mathematically bounded between `0` and `100` after scaling. These weights are initial research assumptions, not fixed scientific truths. They reflect the intuition that wildfire detection depends most strongly on retaining high-confidence detector evidence and preserving fire/smoke regions, while contextual relevance and spatial region preservation also contribute.

The weights will be validated through:

- sensitivity analysis,
- ablation studies,
- downstream wildfire detector evaluation,
- statistical comparisons,
- dataset-level experiments.

If experiments show that alternative weightings better predict downstream wildfire performance, the thesis will revise the weighting scheme accordingly.

The rate-distortion-energy-utility objective will also use normalized components so that no single term dominates purely because of units. Rate will be represented by relative payload size, distortion by normalized reconstruction error or one minus a bounded quality metric, energy by normalized estimated joules per image, and utility by SUS divided by `100`.

---

## 10. Architecture and Methodology

### 10.1 Architecture Figure

```text
Wildfire Image
      |
      v
Wildfire Utility Detector
      |
      v
Semantic Utility Map
      |
      v
VQ-VAE Tokenization
      |
      v
Utility-Aware Token Ranking
      |
      v
Adaptive Transmission
      |
      v
Ground Reconstruction
      |
      v
Wildfire Detection Evaluation
```

### 10.2 Stage Explanation

**Wildfire Image:** input Earth observation image containing possible fire, smoke, burned area, or surrounding context.

**Wildfire Utility Detector:** estimates wildfire-relevant evidence using detector confidence, region relevance, or lightweight utility estimation.

**Semantic Utility Map:** assigns each spatial region a normalized utility value from `0` to `1`.

**VQ-VAE Tokenization:** converts the image into a discrete token grid suitable for packetized transmission.

**Utility-Aware Token Ranking:** ranks each token according to semantic utility, token entropy, and bandwidth cost.

**Adaptive Transmission:** transmits only a subset of tokens under bandwidth or energy constraints.

**Ground Reconstruction:** reconstructs the image from retained tokens.

**Wildfire Detection Evaluation:** evaluates whether wildfire-relevant information is preserved after compression and reconstruction.

---

## 11. Success Criteria

The following are target benchmarks rather than guaranteed outcomes:

- SUS in the range of `75-80` or higher at moderate token retention levels, depending on dataset difficulty and label quality;
- more than `80%` bandwidth reduction relative to original image payloads under at least one moderate-retention operating point;
- at least `85-90%` wildfire detection retention in downstream detector evaluation where reliable wildfire labels are available;
- statistically significant improvement over non-utility-aware learned baselines under matched bandwidth constraints for at least one primary wildfire dataset;
- near-real-time inference on a GPU workstation and documented latency/memory feasibility on at least one edge-relevant platform or emulator;
- clear rate-utility tradeoff curves across token retention levels;
- reproducible dataset-level results with confidence intervals.

These criteria will guide evaluation but may be revised if dataset characteristics, label noise, sensor resolution, or hardware access make a target unrealistic. The thesis will report both successful and unsuccessful operating regimes.

---

## 12. Preliminary Pilot Results

The current results are **preliminary pilot results** from one validation image:

```text
data/photos/val/img_000005.jpg
```

They demonstrate feasibility only. They should not be interpreted as dataset-level conclusions.

### 12.1 Single Compression Run

| Metric | Result |
|---|---:|
| Original size | `100.1055 KB` |
| Compressed semantic payload | `8.7578 KB` |
| Compression ratio | `11.4304x` |
| Bandwidth saved | `91.25%` |
| PSNR | `22.1203 dB` |
| SSIM | `0.9400` |
| LPIPS | `0.501685` |
| Semantic Utility Score | `77.6579` |
| Semantic fidelity | `64.41%` |
| Tokens transmitted | `7,373 / 16,384` |
| Token entropy | `3.5399 bits` |
| Inference latency | `1670.7928 ms` |
| Transmission time saved | `0.1390 s` |
| Total energy estimate | `20.053101 J` |

### 12.2 Pilot Benchmark Results

| Method | Compression | Bandwidth Saved | PSNR | SSIM | LPIPS | SUS |
|---|---:|---:|---:|---:|---:|---:|
| JPEG Q20 | `3.6420x` | `72.54%` | `34.0599` | `0.9964` | `0.184744` | `86.4023` |
| JPEG Q40 | `2.1799x` | `54.13%` | `37.3114` | `0.9982` | `0.086540` | `86.7196` |
| JPEG Q80 | `0.9475x` | `0.00%` | `47.0541` | `0.9999` | `0.010872` | `87.4354` |
| VQ-VAE full tokens | `7.4551x` | `86.59%` | `27.0335` | `0.9837` | `0.386368` | `100.0000` |
| Semantic VQ-VAE 45% | `11.4304x` | `91.25%` | `22.1203` | `0.9400` | `0.501685` | `77.6579` |
| Semantic VQ-VAE 25% | `17.7472x` | `94.37%` | `15.2083` | `0.6724` | `0.576639` | `60.7547` |

The pilot results suggest that utility-aware token pruning can increase compression and bandwidth savings, but it also reduces distortion-based quality metrics and semantic utility as retention decreases. This tradeoff is central to the proposed research.

---

## 13. Research Challenges and Mitigation Strategies

### 13.1 Utility Estimation Uncertainty

Wildfire utility maps may be noisy, especially under smoke, cloud cover, mixed terrain, or low resolution.

**Mitigation:** compare multiple utility estimators, validate against active fire products, and quantify uncertainty.

### 13.2 Detector Bias and Generalization

Wildfire detectors may generalize poorly across regions, seasons, sensors, or atmospheric conditions.

**Mitigation:** evaluate across multiple datasets and perform geographic and temporal splits where possible.

### 13.3 Tradeoff Between Utility and Visual Fidelity

Preserving semantic utility may reduce PSNR or perceptual quality.

**Mitigation:** report both distortion metrics and mission metrics; avoid using one metric as the sole criterion.

### 13.4 Dataset Variability

Wildfire datasets vary in resolution, labeling quality, spectral bands, and temporal coverage.

**Mitigation:** define dataset-specific protocols and report results separately before aggregation.

### 13.5 Edge Deployment Constraints

Onboard systems may have limited compute, memory, power, and thermal capacity.

**Mitigation:** evaluate CPU, GPU, ONNX, and Jetson-class deployment options; measure latency and memory explicitly.

### 13.6 Satellite Communication Assumptions

Transmission simulations may simplify real satellite link dynamics.

**Mitigation:** clearly state link assumptions, conduct sensitivity analysis, and align simulations with realistic downlink parameters where possible.

---

## 14. Experimental Plan

### Stage 1: Wildfire Dataset Evaluation

Evaluate utility-aware compression on Sentinel-2 wildfire imagery, MODIS active fire products, and FIRMS-linked wildfire data.

### Stage 2: Token Retention Sweep

Evaluate token retention levels:

```text
10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90%, 100%
```

Generate PSNR, SSIM, LPIPS, SUS, bandwidth, rate-utility, and energy-utility curves.

### Stage 3: Baseline Comparison

Compare against:

- JPEG,
- JPEG2000 where available,
- CCSDS-style compression where available,
- full-token VQ-VAE,
- non-utility-aware VQ-VAE,
- trained autoencoder,
- variational autoencoder,
- lightweight CNN compression model.

### Stage 4: Ablation Studies

Evaluate:

- without utility map,
- without detector confidence,
- without entropy term,
- without energy term,
- without adaptive transmission.

### Stage 5: Statistical Validation

Report mean, median, standard deviation, confidence intervals, paired t-tests, Wilcoxon signed-rank tests, and effect sizes.

### Stage 6: Secondary Mission Validation

Use flood detection and ship detection to test adaptability while preserving wildfire as the core thesis scope.

### Stage 7: Edge Deployment Evaluation

Evaluate CPU, CUDA, ONNX, and Jetson-class execution with latency, memory, FPS, and energy estimates.

---

## 15. Feasibility and Risk Management

The project is feasible because the CompressAI prototype is already operational. It includes:

- VQ-VAE tokenization,
- semantic utility scoring,
- utility-aware pruning,
- LPIPS, PSNR, and SSIM evaluation,
- JPEG baseline evaluation,
- satellite transmission simulation,
- energy modeling,
- benchmark export,
- API and dashboard support.

The PhD therefore extends an existing research platform rather than beginning from an empty implementation base.

| Risk | Mitigation |
|---|---|
| Wildfire detector quality is insufficient | Fine-tune or replace detectors using wildfire datasets |
| SUS weights are subjective | Conduct sensitivity analysis and ablation studies |
| JPEG outperforms learned models in PSNR | Evaluate utility and downstream detection under matched bitrate, not PSNR alone |
| Dataset access is incomplete | Use multiple public datasets and document availability constraints |
| Edge inference is too slow | Use ONNX, quantization, TensorRT, and smaller utility models |
| Scope becomes too broad | Keep wildfire as the central mission and use other missions only for validation |

---

## 16. Expected Contributions

Each contribution is designed to be measurable, publishable, and linked to thesis chapters and experiments.

1. **A wildfire-centric semantic utility framework for Earth observation compression.**  
   Measured through wildfire utility maps, downstream detector retention, and SUS.  
   Thesis link: Chapter 3.  
   Publication link: Paper 1.

2. **A Semantic Utility Score for evaluating mission-aware compression.**  
   Validated through sensitivity analysis, ablations, and correlation with downstream wildfire detection.  
   Thesis link: Chapter 3.  
   Publication link: Paper 1.

3. **A utility-aware token prioritization algorithm for learned discrete visual tokens.**  
   Evaluated through retention sweeps and comparison with non-utility-aware token selection.  
   Thesis link: Chapter 4.  
   Publication link: Paper 2.

4. **A rate-distortion-energy-utility evaluation framework.**  
   Measured through objective components, energy estimates, and rate-utility tradeoff curves.  
   Thesis link: Chapter 5.  
   Publication link: Paper 3.

5. **A dataset-level benchmark for wildfire semantic compression.**  
   Evaluated with mean, median, standard deviation, confidence intervals, and statistical tests.  
   Thesis link: Chapter 6.  
   Publication link: Papers 1 and 2.

6. **An edge-deployment analysis for semantic compression on resource-constrained hardware.**  
   Measured through latency, memory, FPS, and energy estimates on edge-relevant platforms.  
   Thesis link: Chapter 7.  
   Publication link: Paper 4.

---

## 17. Expected Publications

### Paper 1
**Semantic Utility Score for Wildfire-Centric Earth Observation Compression**  
Potential venues: IEEE Access, Remote Sensing, ICIP.

### Paper 2
**Utility-Aware Token Prioritization for Semantic Communication in Wildfire Earth Observation**  
Potential venues: IEEE TGRS, ICIP, Remote Sensing.

### Paper 3
**Energy-Aware Adaptive Communication for Resource-Constrained Earth Observation Systems**  
Potential venues: IEEE Aerospace, IEEE Access, ESA Earth Observation workshops.

### Paper 4
**Edge Deployment of Semantic Compression on Satellite-Class Hardware**  
Potential venues: IEEE Aerospace, embedded AI workshops, space systems venues.

---

## 18. Institutional Alignment

### 18.1 University of Luxembourg SnT

The project aligns with research themes in satellite communications, non-terrestrial networks, AI-native communication, integrated terrestrial-non-terrestrial systems, and future 6G space networks. It provides a concrete Earth observation use case for semantic communication under resource constraints.

### 18.2 ESA Earth Observation Priorities

The wildfire focus aligns with climate resilience, disaster response, environmental monitoring, sustainable satellite communications, and autonomous Earth observation. The project studies how onboard or near-edge AI can reduce communication load while preserving mission-relevant information.

### 18.3 TU Delft Space and Edge AI Research

The project aligns with CubeSat deployment, onboard inference, real-time edge AI, autonomous Earth observation systems, and resource-constrained embedded intelligence. The planned edge deployment experiments provide a direct bridge between algorithmic research and satellite-class implementation constraints.

---

## 19. Thesis Structure

### Chapter 1: Introduction

Motivation, problem statement, research questions, and contributions.

### Chapter 2: Literature Review

Traditional compression, learned compression, VQ-VAE, semantic communication, Earth observation AI, and edge AI for space systems.

### Chapter 3: Semantic Utility Framework

Wildfire utility estimation, SUS definition, sensitivity analysis, and validation.

### Chapter 4: Utility-Aware Token Prioritization

VQ-VAE tokenization, token scoring, adaptive pruning, and bandwidth-constrained selection.

### Chapter 5: Rate-Distortion-Energy-Utility Optimization

Formal objective, energy modeling, transmission simulation, and tradeoff analysis.

### Chapter 6: Earth Observation Evaluation

Wildfire dataset experiments, baseline comparisons, retention sweeps, ablations, and statistical validation.

### Chapter 7: Edge Deployment

CPU, GPU, ONNX, Jetson-class deployment, latency, memory, FPS, and energy analysis.

### Chapter 8: Conclusions

Findings, limitations, contributions, and future research.

---

## 20. Four-Year Work Plan

### Year 1

- Complete detailed literature review.
- Curate wildfire datasets.
- Refine wildfire utility estimation.
- Validate preliminary SUS formulation.
- Produce first workshop or short conference submission.

### Year 2

- Develop utility-aware token prioritization.
- Run retention sweeps and ablation studies.
- Compare against JPEG and learned compression baselines.
- Submit Paper 1 or Paper 2.

### Year 3

- Develop energy-aware objective and satellite communication analysis.
- Conduct dataset-level statistical validation.
- Add secondary validation on flood and ship detection.
- Submit journal or conference paper.

### Year 4

- Complete edge deployment evaluation.
- Finalize statistical analysis and thesis integration.
- Submit final journal paper.
- Write and defend dissertation.

---

## 21. Ethical and Societal Considerations

The research is motivated by wildfire response, climate resilience, and environmental monitoring. It may contribute to faster disaster awareness and more efficient satellite communication. However, Earth observation technologies can also raise concerns related to surveillance and dual-use applications.

The project will therefore emphasize:

- environmental and disaster-response applications,
- transparent reporting of limitations,
- responsible dataset use,
- avoidance of unsupported operational claims,
- clear separation between research evaluation and deployment readiness.

---

## 22. Supervisor-Review Notes Integrated Into Proposal

From the perspective of a potential Luxembourg SnT supervisor, the proposal must clearly connect semantic communication with satellite and non-terrestrial network research. This has been addressed through the communication-constrained framing, adaptive transmission formulation, and future 6G/NTN alignment.

From the perspective of a TU Delft supervisor, the proposal must be feasible on resource-constrained hardware. This has been addressed through the edge deployment plan, ONNX path, latency metrics, and Jetson-class evaluation plan.

From the perspective of an ESA Earth Observation reviewer, the proposal must show societal relevance, climate relevance, and realistic Earth observation methodology. This has been addressed by focusing the research on wildfire detection, climate resilience, disaster response, and Sentinel/MODIS/FIRMS-linked evaluation.

Remaining assumptions that require validation during the PhD include SUS weighting, detector generalization, dataset representativeness, and realism of satellite communication simulations. These are explicitly included as research challenges rather than hidden assumptions.

---

## 23. Conclusion

This proposal defines a focused PhD research programme on semantic utility-aware adaptive communication for wildfire-centric Earth observation systems under resource constraints. The work addresses a specific and relevant research gap: current compression and communication systems do not jointly integrate learned discrete token compression, wildfire utility estimation, adaptive token transmission, energy-aware communication, satellite constraints, and downstream Earth observation evaluation.

The project is feasible because it extends an existing CompressAI prototype rather than starting from scratch. At the same time, the scientific work remains substantial: dataset-level validation, utility metric design, adaptive token selection, formal objective evaluation, statistical testing, and edge deployment analysis are all required before publication-level conclusions can be made.

The expected outcome is a rigorous, publishable, and practically relevant framework for AI-native Earth observation communication, with wildfire detection as the primary use case and broader relevance to future autonomous satellite systems.

---

## References

[1] G. K. Wallace, "The JPEG Still Picture Compression Standard," *Communications of the ACM*, vol. 34, no. 4, pp. 30-44, 1991.

[2] D. S. Taubman and M. W. Marcellin, *JPEG2000: Image Compression Fundamentals, Standards and Practice*. Boston, MA, USA: Springer, 2002.

[3] Consultative Committee for Space Data Systems, *Image Data Compression*, CCSDS 122.0-B-2, Blue Book, 2017.

[4] J. Ballé, V. Laparra, and E. P. Simoncelli, "End-to-End Optimized Image Compression," in *Proc. International Conference on Learning Representations (ICLR)*, 2017.

[5] J. Ballé, D. Minnen, S. Singh, S. J. Hwang, and N. Johnston, "Variational Image Compression with a Scale Hyperprior," in *Proc. International Conference on Learning Representations (ICLR)*, 2018.

[6] D. Minnen, J. Ballé, and G. D. Toderici, "Joint Autoregressive and Hierarchical Priors for Learned Image Compression," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2018.

[7] Z. Cheng, H. Sun, M. Takeuchi, and J. Katto, "Learned Image Compression with Discretized Gaussian Mixture Likelihoods and Attention Modules," in *Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2020.

[8] L. Theis, W. Shi, A. Cunningham, and F. Huszár, "Lossy Image Compression with Compressive Autoencoders," in *Proc. International Conference on Learning Representations (ICLR)*, 2017.

[9] G. Toderici et al., "Full Resolution Image Compression with Recurrent Neural Networks," in *Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2017.

[10] E. Agustsson et al., "Generative Adversarial Networks for Extreme Learned Image Compression," in *Proc. IEEE International Conference on Computer Vision (ICCV)*, 2019.

[11] J. Bégaint, F. Racapé, S. Feltman, and A. Pushparaja, "CompressAI: A PyTorch Library and Evaluation Platform for End-to-End Compression Research," arXiv:2011.03029, 2020.

[12] A. van den Oord, O. Vinyals, and K. Kavukcuoglu, "Neural Discrete Representation Learning," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2017.

[13] C. E. Shannon, "A Mathematical Theory of Communication," *Bell System Technical Journal*, vol. 27, pp. 379-423 and 623-656, 1948.

[14] W. Weaver, "Recent Contributions to the Mathematical Theory of Communication," in *The Mathematical Theory of Communication*. Urbana, IL, USA: University of Illinois Press, 1949.

[15] E. C. Strinati and S. Barbarossa, "6G Networks: Beyond Shannon Towards Semantic and Goal-Oriented Communications," *Computer Networks*, vol. 190, 2021.

[16] D. Gündüz, Z. Qin, I. E. Aguerri, H. S. Dhillon, Z. Yang, A. Yener, K. K. Wong, and C. B. Chae, "Beyond Transmitting Bits: Context, Semantics, and Task-Oriented Communications," *IEEE Journal on Selected Areas in Communications*, vol. 41, no. 1, pp. 5-41, 2023.

[17] H. Xie, Z. Qin, G. Y. Li, and B. H. Juang, "Deep Learning Enabled Semantic Communication Systems," *IEEE Transactions on Signal Processing*, vol. 69, pp. 2663-2675, 2021.

[18] Q. Lan, D. Wen, Z. Zhang, Q. Zeng, X. Chen, P. Popovski, and K. Huang, "What Is Semantic Communication? A View on Conveying Meaning in the Era of Machine Intelligence," *Journal of Communications and Information Networks*, vol. 6, no. 4, pp. 336-371, 2021.

[19] P. Popovski et al., "A Perspective on Time Toward Wireless 6G," *Proceedings of the IEEE*, vol. 110, no. 8, pp. 1116-1146, 2022.

[20] P. Helber, B. Bischke, A. Dengel, and D. Borth, "EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification," *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing*, vol. 12, no. 7, pp. 2217-2226, 2019.

[21] A. Van Etten, D. Lindenbaum, and T. M. Bacastow, "SpaceNet: A Remote Sensing Dataset and Challenge Series," arXiv:1807.01232, 2018.

[22] I. Demir et al., "DeepGlobe 2018: A Challenge to Parse the Earth Through Satellite Images," in *Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops*, 2018.

[23] X. X. Zhu et al., "Deep Learning in Remote Sensing: A Comprehensive Review and List of Resources," *IEEE Geoscience and Remote Sensing Magazine*, vol. 5, no. 4, pp. 8-36, 2017.

[24] L. Ma, Y. Liu, X. Zhang, Y. Ye, G. Yin, and B. A. Johnson, "Deep Learning in Remote Sensing Applications: A Meta-Analysis and Review," *ISPRS Journal of Photogrammetry and Remote Sensing*, vol. 152, pp. 166-177, 2019.

[25] D. Marmanis, M. Datcu, T. Esch, and U. Stilla, "Deep Learning Earth Observation Classification Using ImageNet Pretrained Networks," *IEEE Geoscience and Remote Sensing Letters*, vol. 13, no. 1, pp. 105-109, 2016.

[26] L. Bruzzone and D. F. Prieto, "Automatic Analysis of the Difference Image for Unsupervised Change Detection," *IEEE Transactions on Geoscience and Remote Sensing*, vol. 38, no. 3, pp. 1171-1182, 2000.

[27] C. O. Justice et al., "The MODIS Fire Products," *Remote Sensing of Environment*, vol. 83, no. 1-2, pp. 244-262, 2002.

[28] L. Giglio, W. Schroeder, and C. O. Justice, "The Collection 6 MODIS Active Fire Detection Algorithm and Fire Products," *Remote Sensing of Environment*, vol. 178, pp. 31-41, 2016.

[29] W. Schroeder, P. Oliva, L. Giglio, and I. A. Csiszar, "The New VIIRS 375 m Active Fire Detection Data Product: Algorithm Description and Initial Assessment," *Remote Sensing of Environment*, vol. 143, pp. 85-96, 2014.

[30] L. Giglio, L. Boschetti, D. P. Roy, M. L. Humber, and C. O. Justice, "The Collection 6 MODIS Burned Area Mapping Algorithm and Product," *Remote Sensing of Environment*, vol. 217, pp. 72-85, 2018.

[31] L. Giglio, W. Schroeder, J. V. Hall, and C. O. Justice, "MODIS Collection 6 Active Fire Product User's Guide," NASA, 2015.

[32] S. Roteta, A. Bastarrika, M. Padilla, T. Storm, and E. Chuvieco, "Development of a Sentinel-2 Burned Area Algorithm: Generation of a Small Fire Database for Sub-Saharan Africa," *Remote Sensing of Environment*, vol. 222, pp. 1-17, 2019.

[33] E. Chuvieco, M. P. Martín, and A. Palacios, "Assessment of Different Spectral Indices in the Red-Near-Infrared Spectral Domain for Burned Land Discrimination," *International Journal of Remote Sensing*, vol. 23, no. 23, pp. 5103-5110, 2002.

[34] D. P. Roy, L. Boschetti, C. O. Justice, and J. Ju, "The Collection 5 MODIS Burned Area Product: Global Evaluation by Comparison with the MODIS Active Fire Product," *Remote Sensing of Environment*, vol. 112, no. 9, pp. 3690-3707, 2008.

[35] P. Warden and D. Situnayake, *TinyML: Machine Learning with TensorFlow Lite on Arduino and Ultra-Low-Power Microcontrollers*. Sebastopol, CA, USA: O'Reilly Media, 2019.

[36] S. Mittal, "A Survey on Optimized Implementation of Deep Learning Models on the NVIDIA Jetson Platform," *Journal of Systems Architecture*, vol. 97, pp. 428-442, 2019.

[37] G. Giuffrida et al., "The Phi-Sat-1 Mission: The First On-Board Deep Neural Network Demonstrator for Satellite Earth Observation," *IEEE Transactions on Geoscience and Remote Sensing*, vol. 60, pp. 1-14, 2022.

[38] M. Esposito et al., "Artificial Intelligence for Space Applications: A Survey," *IEEE Access*, vol. 10, pp. 113650-113683, 2022.

[39] J. Bouwmeester and J. Guo, "Survey of Worldwide Pico- and Nanosatellite Missions, Distributions and Subsystem Technology," *Acta Astronautica*, vol. 67, no. 7-8, pp. 854-862, 2010.

[40] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, "You Only Look Once: Unified, Real-Time Object Detection," in *Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2016.
