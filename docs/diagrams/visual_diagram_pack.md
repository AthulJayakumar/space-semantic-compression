# CompressAI Visual Diagram Pack

This page collects the main block diagrams, architecture diagrams, working diagrams, and result graphs for the Space Semantic Compression project.

Generated image files are saved in:

`figures/project_diagrams/`

## 1. System Architecture

Use this diagram to explain the software platform.

```mermaid
flowchart LR
    UI[Streamlit Mission Dashboard] --> API[FastAPI Backend]
    API --> SEM[Semantic Utility Service]
    API --> ENC[Encoder Service]
    ENC --> VQVAE[VQ-VAE Checkpoints]
    VQVAE --> TOK[Token Prioritisation]
    TOK --> DEC[Decoder Service]
    DEC --> MET[Metrics Service]
    SEM --> TOK
    MET --> REP[Reports, Figures, Evidence Pack]
```

Image file:

`figures/project_diagrams/01_system_architecture.png`

![CompressAI system architecture](../../figures/project_diagrams/01_system_architecture.png)

## 2. Working Pipeline

Use this diagram in the PhD proposal methodology section.

```mermaid
flowchart TD
    A[Sentinel-2 or wildfire image] --> B[Wildfire utility detector]
    B --> C[Semantic utility map]
    C --> D[VQ-VAE token encoder]
    D --> E[Utility-aware token ranking]
    E --> F[Bandwidth-constrained token transmission]
    F --> G[Decoder reconstruction]
    G --> H[SUS and detector-retention evaluation]
```

Image file:

`figures/project_diagrams/02_working_pipeline.png`

![Semantic utility-aware compression working pipeline](../../figures/project_diagrams/02_working_pipeline.png)

## 3. Token Transmission Working Diagram

Use this when explaining the core invention or research concept.

```mermaid
flowchart LR
    IMG[Image tile] --> TOKENS[Learned token grid]
    UTIL[Mission utility map] --> RANK[Token priority score]
    TOKENS --> RANK
    RANK --> KEEP[Retain high-utility tokens]
    RANK --> DROP[Prune low-utility tokens]
    KEEP --> DOWNLINK[Satellite downlink payload]
    DOWNLINK --> RECON[Reconstruction and downstream detector]
```

Image file:

`figures/project_diagrams/03_token_transmission_working_diagram.png`

![Token-based semantic transmission working diagram](../../figures/project_diagrams/03_token_transmission_working_diagram.png)

## 4. Research Benchmark Pipeline

Use this for research reports and supervisor discussions.

```mermaid
flowchart LR
    DATA[Datasets] --> MODELS[Model variants]
    MODELS --> RUNS[Compression runs]
    RUNS --> METRICS[Metric computation]
    METRICS --> STATS[Statistical analysis]
    STATS --> OUTPUTS[CSV, reports, figures]
```

Image file:

`figures/project_diagrams/04_experiment_validation_pipeline.png`

![Research benchmark and validation pipeline](../../figures/project_diagrams/04_experiment_validation_pipeline.png)

## 5. Career Evidence Positioning

Use this to explain how the same project supports patent strategy, PhD applications, Innovator Founder evidence, and Global Talent evidence.

```mermaid
mindmap
  root((Semantic Utility-Aware Satellite Compression))
    Patent/IP
      Technical process
      Token prioritisation
      Adaptive downlink
    PhD
      Research hypothesis
      Sentinel-2 evidence
      Publications
    Innovator Founder
      Innovative
      Viable
      Scalable
    Global Talent
      Open-source evidence
      Expert validation
      Research outputs
```

Image file:

`figures/project_diagrams/05_career_evidence_positioning.png`

![Patent PhD visa evidence positioning](../../figures/project_diagrams/05_career_evidence_positioning.png)

## 6. Result Graphs

These graphs are generated from the 500-patch Sentinel-2 benchmark.

| Figure | Purpose |
|---|---|
| `06_sentinel2_metric_comparison.png` | Compare original and regularized VQ-VAE across SUS, detector retention, PSNR, SSIM, compression ratio, and bandwidth saved |
| `07_paired_effects_ci.png` | Show paired mean effects with bootstrap 95% confidence intervals |
| `08_model_decision_tradeoff.png` | Explain why the regularized model is not promoted as default yet |

### Sentinel-2 Metric Comparison

![Sentinel-2 metric comparison](../../figures/project_diagrams/06_sentinel2_metric_comparison.png)

### Paired Effects With Confidence Intervals

![Paired effects with confidence intervals](../../figures/project_diagrams/07_paired_effects_ci.png)

### Model Promotion Trade-Off

![Model promotion trade-off](../../figures/project_diagrams/08_model_decision_tradeoff.png)

## Regeneration Command

```powershell
C:\Conda\envs\img2word\python.exe scripts\generate_project_diagrams.py
```
