# Research Status - Release 0.2.0

**Frozen on:** 24 September 2026
**Primary mission:** wildfire-centric Earth Observation
**Central hypothesis:** unconfirmed

## Current Evidence Tier

The repository contains three different evidence tiers:

1. **Current frozen evidence:** the 60-event-pair EcoFireBias official-validation comparison at a 1,200-byte ceiling. This is the principal result used in the PhD proposal.
2. **Development and exploratory evidence:** HLS, DFire, FLAME, CEMS/ImpactMesh, selector and decoder screens. These demonstrate implementation history but do not establish independent superiority.
3. **Sealed evidence:** the 37-pair EcoFireBias test cohort. It remains unscored because the predeclared advancement gate was not met.

## Decision

The adapted VQ-VAE is the correct release checkpoint because it is the latest model with frozen training provenance and a one-time official-validation report. It is **not** the winning method. JPEG2000 is the strongest tested comparator on the principal SUS and dNBR-proxy Dice endpoints.

The release checkpoint is therefore labelled:

```text
experimental_no_go_for_sealed_test
```

No further model selection should use the scored 18-event development or 60-event official-validation cohorts. Improvement requires a separate training/development source and a genuinely untouched, independently labelled satellite cohort.

## Application-Safe Research Claim

> A reproducible semantic-transmission prototype has been developed and evaluated. Training adaptation significantly improved the learned baseline on a frozen Sentinel-2 validation cohort, but JPEG2000 remained stronger at the tested maximum payload. The proposed PhD will determine whether independent-label utility-aware token transmission can close this gap under event-, scene- and geography-separated validation while controlling negative-scene false alarms and onboard cost.

## Files of Record

- `MODEL_CARD.md`
- `results/future_satellite_cohort_audit/ecofirebias_official_validation/VALIDATION_PLAN_FROZEN.json`
- `results/future_satellite_cohort_audit/ecofirebias_official_validation/VALIDATION_REPORT.json`
- `results/future_satellite_cohort_audit/ecofirebias_official_validation/VALIDATION_REPORT.md`
- `results/future_satellite_cohort_audit/DEVELOPMENT_DECISION.md`
- `career_evidence_pack/phd/submission_reconciliation_2026.md`
- `reports/phd_proposal_submission_2026.md`

## Scientifically Valid Next Step

Acquire and audit an external Sentinel-2 cohort with independent masks, positive and negative scenes, source-scene identifiers, acquisition metadata and geospatial footprints. Freeze split membership, exact payload comparison, the joint SUS-plus-independent-mask endpoint and the negative-scene tolerance before scoring. Do not open the existing sealed test merely because the current gate failed.
