# Semantic Utility-Aware Communication for Wildfire-Centric Earth Observation

**PhD research proposal | Athul Jayakumar | September 2026**
**Field:** Earth Observation, learned image compression, semantic communication and edge AI
**Status:** application draft supported by preliminary validation, not a claim of demonstrated superiority

## Abstract

Earth Observation systems face a mismatch between growing image collection and finite downlink opportunities. For wildfire monitoring, transmitted information is valuable not only when a reconstructed image looks faithful, but when evidence of burning, smoke and damaged terrain remains usable without creating misleading detections. This project asks whether discrete learned image tokens can be selected according to wildfire relevance so that mission utility is preserved at severe, explicitly accounted byte budgets. The hypothesis is falsifiable: utility-aware selection should improve independent wildfire-label retention over strong conventional codecs at comparable transmitted size, without an unacceptable increase in false alarms or onboard cost.

A working VQ-VAE prototype already enables token selection, serialisation, reconstruction and event-paired comparison. In one frozen Sentinel-2 validation check of 60 post-fire/negative event pairs at a 1,200-byte payload ceiling, adaptation raised Semantic Utility Score (SUS) by 12.43 points and dNBR-proxy Dice by 0.183 over a base VQ-VAE. However, JPEG2000 remained ahead of the adapted model by 11.85 SUS points and 0.064 Dice, and the adapted detector's predicted-positive area increased on negative images. These findings establish feasibility and a precise unsolved problem, not proof of the hypothesis. The PhD will validate mission utility against independent geospatial labels, investigate token prioritisation within fixed data partitions, and measure the benefit-cost boundary for satellite-like communication. Wildfire remains the primary mission throughout.

## 1. Scientific problem and gap

JPEG2000 and the CCSDS image-compression family provide mature, rate-controlled approaches relevant to satellite imaging [1,2]. Learned codecs optimise rate-distortion trade-offs [3,4], and VQ-VAE provides discrete representations that can be scheduled as tokens [5]. The gap addressed here is narrower than a claim that semantic compression is new: **for wildfire Earth Observation, it is not yet established whether utility-ranked learned tokens outperform strong codecs at an auditable byte budget when utility is checked against independent labels and false-positive behaviour.** This is a combined measurement, representation and systems question. Detector-derived utility alone risks rewarding preservation of a detector's own errors. Apparent rate gains can also disappear when token indices, masks, headers and metadata are counted.

Sentinel-2 supplies high-resolution multispectral observations, but an RGB token codec is not automatically a multispectral or flight-ready compressor [6]. FIRMS active-fire detections offer useful external context, but point detections are not pixel-level burn-scar truth [7]. The proposal therefore distinguishes active-fire points, independently sourced burn-area polygons, spectral proxies and model-generated maps. It does not conflate them.

**Proposed contributions.** (i) An auditable wildfire utility evaluation combining SUS components with independent-label overlap and negative-scene specificity; (ii) a controlled test of utility-aware versus random/entropy token selection and JPEG/JPEG2000 at matched transmitted sizes; (iii) an event-, footprint- and geography-screened Earth Observation benchmark; and (iv) a measured boundary between downlink benefit and onboard compute/memory cost. Each contribution remains publishable even if the semantic method loses, provided the comparison and negative findings are reproducible.

## 2. Research questions and decision criteria

| Question | Method | Primary measurable outcome |
| --- | --- | --- |
| **RQ1: Measurement.** Does SUS track independent wildfire evidence, or mainly detector self-consistency? | Lock SUS definition; compare its components with independent burn-area masks, positive and negative scenes, and calibration/error analysis. | Event-level association with mask Dice/IoU; negative-scene predicted-positive area; uncertainty intervals. |
| **RQ2: Representation.** At a comparable payload, can utility-ranked VQ-VAE tokens retain more wildfire evidence than random/entropy selection and JPEG/JPEG2000? | Paired per-scene byte-budget curves; count every transmitted token, selection mask and header; freeze operating points before testing. | Paired differences in independent-mask Dice **and** SUS, with 95% event-clustered intervals; actual bytes and quality metrics. |
| **RQ3: Generalisation.** Does any gain survive a new wildfire event, scene footprint and geography? | Train/develop on separate sources; reserve an independently labelled external cohort for one final test after a predeclared development gate. | External paired effect estimates by event and geography; fire-positive and fire-negative stratification. |
| **RQ4: Systems.** When does utility justify onboard processing and limited-contact transmission? | Profile encoding, selection, decoding, peak memory and energy on specified hardware; simulate contact windows with stated link assumptions. | Measured latency, memory and payload; simulated downlink time explicitly separated from hardware measurements. |

The central hypothesis is that utility-aware token prioritisation can preserve **independently verified** wildfire information better than distortion-oriented codecs at equal transmitted size in at least a defined severe-rate regime, without worsening negative-scene false alarms beyond a predeclared tolerance. The null outcome is also meaningful: if strong codecs dominate after independent-label and byte accounting, the thesis will characterise why and where semantic selection fails. Any superiority claim requires both utility and independent-label endpoints; SUS alone cannot establish it.

## 3. Methodology

**Pipeline.** Sentinel-2 or wildfire imagery -> frozen wildfire utility estimator -> utility map -> VQ-VAE encoder -> discrete tokens -> utility-aware ranking under a payload budget -> transmitted bitstream -> decoder -> reconstructed image -> detector and independent-label evaluation. A common RGB pathway provides a controlled starting point; any multispectral extension will be labelled as a separate condition, not silently pooled with RGB results.

**Utility measurement.** The current score is retained, not redefined: SUS = 100 x (0.4D + 0.3O + 0.2R + 0.1P), where D, O, R and P denote detector-confidence, object-count, relevance-mass and important-region retention relative to the original. Each component and its denominator will be reported. Empty-target and zero-denominator cases will follow a fixed, documented rule and be analysed separately. Ratios can exceed one when reconstruction creates detections; reporting must therefore retain raw components and describe any clipping before interpreting SUS as a 0-100 score. SUS shares a detector with detector-retention and is **not** an independent ground truth. Primary external checks will be mask Dice/IoU, false-positive area on negatives and, where labels permit, event-level detection sensitivity.

**Data and separation.** Existing DFire and FLAME experiments establish prototype operation but not satellite generalisation. The first-year priority is to obtain licensed Sentinel-2 imagery paired with independently sourced burned-area annotations and negatives. Every sample will retain source-scene identifier, acquisition date, event identifier, geospatial footprint, label provenance, licence and preprocessing history. Split blocking will be by wildfire event and source scene, with geographic separation audited at footprint level. Cloud, smoke, season and biome will be recorded for stratified analysis. FIRMS points may support event matching, not replace polygon masks. The present EcoFireBias quantised dNBR labels are **spectral proxies**, not verified ground truth.

**Comparison protocol.** Fix preprocessing, spatial support, colour path and data partitions before model selection. Compare JPEG, JPEG2000, full VQ-VAE, random-token, entropy-token and utility-token modes, plus CCSDS where an implementation and appropriate input modality are available. Use a series of feasible maximum byte budgets and, for a narrower rate-matched comparison, match *actual* serialised payload bytes within a prespecified tolerance. Count codec headers, token addresses, masks and side information. Document that the current JPEG/JPEG2000 encoder search uses the original image to choose highest-PSNR feasible settings; its search time is excluded from the present payload figures and will be included in systems profiling. Report PSNR, SSIM and LPIPS as complementary reconstruction measures, not substitutes for mission utility.

**Statistical plan.** The unit of analysis is the wildfire event, not an image patch. Report per-method distributions and paired event-clustered bootstrap 95% intervals, plus a prespecified paired test with multiplicity control for the principal comparisons. Show effect sizes and confidence intervals rather than relying on p-values. Predeclare the primary byte range and the joint SUS-plus-independent-mask endpoint on development data; evaluate the reserved test only after that gate is met. Report negative-scene area and detector calibration even when the main utility score improves. Do not repeatedly tune on the same validation cohort.

## 4. Preliminary evidence and honest interpretation

The most informative completed comparison used the **EcoFireBias official validation split**, 60 event pairs (120 native 224 x 224 images), with one burn and one matched negative image per event. Selection, methods and quantised dNBR-byte-above-85 proxy-mask rule were frozen before scoring. Every method obeyed the same **maximum 1,200-byte serialised payload per image**, not exactly the same byte count. The adapted checkpoint drew on a separate 600-event source-training selection (480 gradient-training, 120 internal-validation events).

| Method | Burn SUS /100 | Burn proxy Dice | PSNR dB | SSIM | Detector retention | Negative predicted-positive area | Mean payload bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| JPEG | 92.72 | 0.519 | 27.97 | 0.941 | 0.973 | 0.509 | 1,116.7 |
| JPEG2000 RDO | **95.12** | **0.533** | 27.18 | 0.913 | **0.985** | 0.509 | 1,177.5 |
| Base VQ-VAE | 70.85 | 0.286 | 19.87 | 0.644 | 0.728 | 0.225 | 1,140.8 |
| Adapted VQ-VAE | 83.28 | 0.469 | 20.78 | 0.636 | 0.869 | 0.509 | 1,155.4 |

Across 10,000 event-paired bootstrap resamples, adapted minus base VQ-VAE was **+12.43 SUS** (95% CI +8.74 to +16.31) and **+0.183 proxy Dice** (+0.122 to +0.245). Adapted minus JPEG2000 was **-11.85 SUS** (-15.14 to -8.72) and **-0.064 Dice** (-0.113 to -0.018). Negative predicted-positive area rose **+0.284** over base (+0.202 to +0.365), a specificity warning; that area is not a calibrated false-positive rate. These are preliminary validation observations, not independent confirmation. LPIPS and onboard energy were **not** measured in this comparison.

An earlier frozen 18-event joint development gate **failed** against JPEG2000 and was subsequently consulted again; it cannot serve as independent evidence. A 37-pair EcoFireBias test cohort remains **sealed and unscored**. Official-validation events have distinct patch-level event IDs from selected training, but selected validation countries occur in training and patch IDs do not prove non-overlapping source footprints. An older 100-tile CEMS/ImpactMesh screen has reconstructable event-code overlap for 52 tiles with known checkpoint inputs and incomplete ancestor lineage. DFire/FLAME and that CEMS screen are exploratory platform checks, not the principal satellite result. Evidence source: frozen EcoFireBias official-validation report and September 2026 submission reconciliation, both in the project repository.

**Interpretation:** the model can be trained and evaluated reproducibly, and adaptation improves its own baseline. The present evidence **does not support superiority over JPEG2000**. That deficit, the proxy-label limitation and the negative-scene behaviour make the proposed PhD question concrete and falsifiable.

## 5. Four-year research plan and outputs

| Period | Work and go/no-go milestone | Intended output |
| --- | --- | --- |
| Year 1 | Acquire a licensed, independently annotated Sentinel-2 cohort; verify event, scene and footprint separation; audit positive/negative labels; freeze byte and joint utility criteria. | Dataset/protocol paper or reproducible benchmark note; validated endpoint definition. |
| Year 2 | Within training/development partitions, test utility-ranked token selection and bounded representation adaptation against random, entropy, JPEG and JPEG2000; quantify false-alarm trade-off. | Methods paper only if a reproducible contribution emerges; otherwise rigorous negative comparison. |
| Year 3 | Measure encoder, selector, bitstream and decoder on a specified edge platform; model contact-window scenarios with explicit radio assumptions; include CCSDS when technically comparable. | Systems/evaluation paper with measured and simulated quantities separated. |
| Year 4 | Apply the predeclared gate to a final untouched external cohort, replicate effects across geography and wildfire conditions, release code/metadata permitted by licence, and write thesis. | External-validation paper or transparent null result; thesis and reproducibility package. |

The sequence is deliberately gated. No promise of a 500-scene cohort, flight demonstration, patent, accepted paper or particular performance margin is made before data access and results justify it. Flood and ship imagery, if ever used, are secondary checks only; wildfire remains the research mission.

## 6. Risks, feasibility and fit

The biggest scientific risk is that JPEG2000 continues to win. This is not treated as a marketing obstacle: it creates a publishable boundary analysis of rate, utility and detector bias. Data leakage is addressed through complete scene/footprint lineage and independent source documentation; any inherited checkpoint with unresolved lineage is excluded from a clean final test. Proxy-label error is addressed by independent annotations and uncertainty stratification. A detector may inflate SUS or trigger false alarms; independent masks, negative scenes and component-level reporting prevent a single detector-driven number from deciding success. Hardware claims require specified hardware and repeatable profiling, while link and energy projections remain simulations until measured.

The applicant's completed AI MSc and existing open research prototype provide practical preparation for a four-year project. The work fits groups combining EO, trustworthy vision and communication-constrained edge systems, but the proposal is intentionally research-led: a supervisor can shape dataset access and hardware facilities without changing the central hypothesis. The strongest application claim is not a proven product advantage; it is a carefully instrumented, tractable scientific question with a working baseline and an honestly identified gap.

## References

[1] A. Skodras, C. Christopoulos and T. Ebrahimi, "The JPEG 2000 still image compression standard," *IEEE Signal Processing Magazine*, 18(5), 36-58, 2001. https://doi.org/10.1109/79.952804

[2] CCSDS, *Image Data Compression*, CCSDS 122.0-B-2, 2017. https://ccsds.org/Pubs/122x0b2e1.pdf

[3] J. Balle, V. Laparra and E. P. Simoncelli, "End-to-End Optimized Image Compression," ICLR, 2017. https://arxiv.org/abs/1611.01704

[4] J. Balle et al., "Variational Image Compression with a Scale Hyperprior," ICLR, 2018. https://arxiv.org/abs/1802.01436

[5] A. van den Oord, O. Vinyals and K. Kavukcuoglu, "Neural Discrete Representation Learning," NeurIPS, 2017. https://arxiv.org/abs/1711.00937

[6] ESA, "Sentinel-2: Facts and figures," mission documentation. https://www.esa.int/Applications/Observing_the_Earth/Copernicus/Sentinel-2/Facts_and_figures

[7] NASA FIRMS, "Active Fire Data," product documentation. https://firms.modaps.eosdis.nasa.gov/content/active_fire/
