# Model Card: VQ-VAE EcoFireBias Adapted 2026-09

## Identity

| Field | Value |
| --- | --- |
| Model ID | `vqvae-ecofirebias-adapted-2026-09` |
| Project release | `0.2.0` |
| Architecture | VQ-VAE |
| Codebook | 8,192 entries |
| Code dimension | 256 |
| Spatial stride | 16 |
| Expected file | `models/checkpoints/vqvae_ecofirebias_train_adapted.pt` |
| File size | 28,890,327 bytes |
| SHA-256 | `668cf5bda5c69b196d70306dbc6ed576c2675d372ffcfe2605e53a40286011f6` |
| Release status | Experimental; no-go for sealed-test scoring |

The checkpoint binary is excluded from Git. The API verifies the expected SHA-256 hash before loading the default release model.

## Intended Research Use

The model encodes 224 x 224 RGB wildfire or Sentinel-2-derived images into discrete latent tokens for controlled token-selection, serialisation and reconstruction experiments. It supports research on whether selected semantic tokens preserve wildfire-relevant evidence under severe payload constraints.

It is suitable for:

- reproducing the documented VQ-VAE inference pathway;
- comparing base and adapted representations on authorised development data;
- testing payload accounting and reconstruction from transmitted tokens only;
- generating demonstrations clearly labelled as research prototypes.

It is not approved for operational wildfire detection, emergency decisions, spacecraft deployment, navigation, safety-critical use or claims of superiority over JPEG2000.

## Training Provenance

The adapted checkpoint starts from `vqvae_mixed_wire_finetuned.pt`, whose SHA-256 is recorded in the frozen training decision. Adaptation used 600 distinct EcoFireBias source-training event groups: 480 for gradient training and 120 for internal validation. The fine-tune used two epochs, batch size 4, 224-pixel inputs, learning rate `1e-5`, seed 1234 and a frozen codebook. Gate-development, label-calibration and sealed-test event identifiers were excluded from the training manifest.

The checkpoint stores the original architecture arguments and a `fine_tuning` metadata object. The public provenance records are:

- `results/future_satellite_cohort_audit/ecofirebias_training_expansion/TRAINING_PLAN_FROZEN.json`
- `results/future_satellite_cohort_audit/ecofirebias_training_expansion/ADAPTATION_DECISION.json`

The frozen decision records the SHA-256 digest of the full local training manifest. The path-bearing manifest is not published because it contains machine-specific source paths; dataset acquisition and manifest-generation scripts are included.

## Evaluation Summary

On the frozen EcoFireBias official-validation comparison of 60 burn/negative event pairs at a maximum 1,200 serialised bytes per image:

| Model or codec | Burn SUS | dNBR-proxy Dice | PSNR (dB) | SSIM | Mean bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| JPEG2000 RDO | 95.12 | 0.533 | 27.18 | 0.913 | 1,177.5 |
| Base VQ-VAE | 70.85 | 0.286 | 19.87 | 0.644 | 1,140.8 |
| Adapted VQ-VAE | 83.28 | 0.469 | 20.78 | 0.636 | 1,155.4 |

Adaptation improved the base VQ-VAE by 12.43 SUS points and 0.183 proxy Dice, with event-paired 95% bootstrap intervals that excluded zero. The adapted model remained 11.85 SUS points and 0.064 Dice below JPEG2000. Its predicted-positive area on negative images rose by 0.284 relative to the base VQ-VAE.

This was a common maximum-byte ceiling, not an exact equality of transmitted bytes. JPEG/JPEG2000 chose the highest original-image PSNR among predefined feasible settings. Encoder search cost was not included in payload bytes.

## Limitations and Bias

- Labels are quantised dNBR spectral proxies, not manually verified burn-scar masks.
- All selected validation countries occur in the selected training data.
- Patch event IDs do not prove source-scene or geospatial-footprint independence.
- Five validation images required mask-grid reprojection.
- SUS and detector retention share a detector and are not independent evidence.
- Negative predicted-positive area is not a calibrated false-positive rate.
- LPIPS, device energy and real onboard performance were not measured in the frozen comparison.
- The prior 18-event advancement gate failed; its later reuse makes subsequent use exploratory.
- The 37-pair EcoFireBias test cohort is sealed and has not been scored.

## Ethical and Operational Considerations

Reconstructed imagery may omit important areas or create detector responses that were not present in the source. Results must not be used to trigger emergency action. Any operational study requires domain review, calibrated uncertainty, verified geospatial labels, negative-scene controls and a conventional full-image fallback.

## Correct Citation Language

Use: "The adapted VQ-VAE improved over its base checkpoint on a frozen validation cohort but remained below JPEG2000 at the tested 1,200-byte ceiling."

Do not use: "The model outperforms JPEG2000" or "the model has been independently validated for satellite wildfire operations."
