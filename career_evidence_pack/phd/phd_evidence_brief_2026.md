# Preliminary PhD Research Evidence Brief

**Applicant:** Athul Jayakumar
**Proposed topic:** Semantic Utility-Aware Compression for Wildfire-Centric Earth Observation
**Evidence status:** Preliminary, September 2026. This is an evidence annex, not a claim of method superiority.

## Research Problem and Hypothesis

Satellite downlink constraints create a scientific question beyond ordinary image fidelity: which information should be transmitted first when a complete image cannot be sent? This project tests whether learned image tokens ranked by wildfire relevance can preserve mission-relevant evidence more efficiently than distortion-optimised codecs at a fixed communication budget. Wildfire is the primary mission; other hazards are not part of the present validation claim. The central hypothesis remains **unconfirmed**: the strongest current JPEG2000 comparison favours JPEG2000.

## Existing System and Evaluation

Post-fire Sentinel-2 RGB chips enter a wildfire utility estimator and a VQ-VAE encoder. The current utility-aware method ranks discrete latent tokens, serialises retained tokens with their mask, reconstructs at the receiver, and measures both image quality and wildfire-relevant retention. Baselines are selected under the same **1,200-byte maximum transmitted payload per native 224 x 224 image**, counting actual serialised payload bytes. JPEG and JPEG2000 choose the highest original-image PSNR among predefined feasible encoder settings. This is a common byte ceiling, **not exactly matched transmitted byte counts**; encoder search costs are excluded.

The current evidence uses the EcoFireBias official validation split: **60 separate event pairs, 120 images** (one burn and one nearby negative chip per event), sampled across six continents before model scoring. The adapted VQ-VAE used a distinct 600-event source-training selection, split into **480 gradient-training and 120 internal-validation events**. Dataset metadata, source RGB, quantised dNBR proxy masks, and image-mask alignment were checked before this one-time validation. Five images needed reprojection; uncovered edge pixels were excluded. The reference mask is a **quantised spectral proxy** (`dNBR byte > 85`), not manually verified burn-scar truth.

**Endpoints.** SUS is a 0-100 weighted combination of detector-confidence, object, relevance, and important-region retention. It is a model-dependent mission-utility summary, not ground truth. Proxy Dice measures overlap between detector-positive reconstruction regions and a separate dNBR-derived mask. Negative predicted-positive area measures how much of a nominally negative chip the detector flags after reconstruction; it is not a calibrated false-positive rate. PSNR, SSIM, and actual serialised bytes keep the analysis anchored to visual and communication performance. None of these measures alone establishes operational wildfire benefit.

## Main Preliminary Result

| Method | Burn SUS / 100 | Burn proxy Dice | PSNR (dB) | SSIM | Detector retention | Negative predicted-positive area | Mean payload (bytes) |
|:--|--:|--:|--:|--:|--:|--:|--:|
| JPEG | 92.72 | 0.519 | 27.97 | 0.941 | 0.973 | 0.509 | 1,116.7 |
| JPEG2000 RDO | **95.12** | **0.533** | 27.18 | 0.913 | **0.985** | 0.509 | 1,177.5 |
| Base VQ-VAE | 70.85 | 0.286 | 19.87 | 0.644 | 0.728 | 0.225 | 1,140.8 |
| Adapted VQ-VAE | 83.28 | 0.469 | 20.78 | 0.636 | 0.869 | 0.509 | 1,155.4 |

Event-paired adaptation gains over the base VQ-VAE were **+12.43 SUS points** (95% bootstrap CI +8.74 to +16.31) and **+0.183 proxy Dice** (+0.122 to +0.245). However, adapted VQ-VAE remained **11.85 SUS points below JPEG2000** (CI -15.14 to -8.72) and **0.064 Dice below** (CI -0.113 to -0.018). Its predicted-positive area on negative chips increased by **0.284** versus the base (CI +0.202 to +0.365), an important specificity concern. Intervals use 10,000 event-paired bootstrap resamples. SUS and detector retention share a detector and are not independent endpoints; proxy Dice provides a different, though imperfect, label source.

## Scientific Interpretation

The observed training response is useful preliminary evidence that representation adaptation changes wildfire-relevant retention. It **does not demonstrate** that the semantic method beats classical compression. A separately frozen 18-event development gate also failed against JPEG2000; its repeated consultation prevents treating it as confirmation. The 37-pair geographically screened EcoFireBias test cohort remains sealed and unscored. The official-validation events have distinct identifiers from selected training and development events, but **all selected validation countries are represented in training**, and patch-level event IDs do not prove distinct satellite scene footprints. The result is therefore appropriate for a PhD *research problem and feasibility case*, not for a publication claim of superiority.

## Proposed PhD Questions

1. **Measurement:** How should wildfire utility be validated against independent geospatial annotations, beyond detector-derived SUS? Outcome: event-level agreement, false-positive area, calibration, and uncertainty on geographically separated scenes.
2. **Representation and selection:** Under a matched serialised-byte budget, can learned tokens preserve wildfire-relevant regions better than JPEG2000 while controlling negative-scene false alarms? Outcome: paired SUS **and** independent-mask Dice differences, with confidence intervals and a predeclared advancement rule.
3. **Operational relevance:** When, if ever, do utility gains justify onboard compute, latency, memory, and downlink cost? Outcome: measured end-to-end bitrate and device profiling, distinguished from simulated link or energy estimates.

## First-Year Validation Priorities

Secure a licensed satellite cohort with verified independent fire-perimeter or burn-scar annotations; establish exact scene/event and geographic separation from all known checkpoint training data; fix preprocessing, payload accounting, endpoints, and success thresholds **before** evaluating it. Preserve the current failed gate and sealed cohort. Then run one paired JPEG/JPEG2000-versus-token study across byte budgets, report both positive and negative scenes, and publish negative results if the baselines remain stronger. Only after validity is established should model or selector changes be assessed on a separate development source.

## Feasible Four-Year Trajectory

**Year 1:** resolve data provenance and licensing, obtain an independently annotated cohort, validate the utility endpoint, and preregister split and byte-budget rules. **Year 2:** investigate representation and token-ranking changes only within training/development partitions; quantify gains and false-alarm costs against classical codecs. **Year 3:** profile inference and serialisation on a specified edge device and evaluate contact-window constraints, separating measured device results from simulations. **Year 4:** run a final untouched external replication if the predefined advancement criteria are met, publish results including failures, and consolidate the thesis. The project remains valuable if semantic compression loses: identifying the operating conditions and endpoint definitions under which it fails is a defensible research contribution.

## Reproducibility and Scope

Frozen selection, run hashes, per-image rows, and paired statistics are recorded in `results/future_satellite_cohort_audit/ecofirebias_official_validation/`. The earlier gate decision is in `results/future_satellite_cohort_audit/DEVELOPMENT_DECISION.md`. The dataset is [EcoFireBias / wildfire_global](https://huggingface.co/datasets/moritzrengert1/wildfire_global), revision `39f331e50458fba3669837d69fc2fdddbb1d1d69`. This brief supersedes older positive-sounding evidence summaries, but does **not** alter the existing research proposal. Its abstract and preliminary-evidence claims should be reconciled with this result before submission.
