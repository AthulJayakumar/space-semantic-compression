# PhD Application Pack - Current Version

**Applicant:** Athul Jayakumar
**Prepared:** September 2026

## Send these

1. **Research proposal:** [phd_proposal_submission_2026.pdf](phd_proposal_submission_2026.pdf). This is the current five-page proposal. The [editable Markdown](phd_proposal_submission_2026.md) is its source.
2. **Preliminary evidence annex, when requested:** [phd_preliminary_evidence_brief_2026.pdf](phd_preliminary_evidence_brief_2026.pdf). It gives the frozen numbers and important limitations in two pages.
3. **Academic CV, degree record and references:** prepare separately for each call. The existing job-application CV is not a substitute for a research CV. Ask referees early; check each programme's permitted referee types and deadlines rather than assuming an employer reference is accepted.

## Repository release used by this application

The application documents correspond to research release `0.2.0` and model ID `vqvae-ecofirebias-adapted-2026-09`. Reviewers can verify the version in `research_release.json`, read the limitations in `MODEL_CARD.md`, and reproduce the internal consistency checks with `python scripts/verify_research_release.py`. The checkpoint is deliberately labelled experimental and no-go for sealed-test scoring.

Do **not** submit `final_supervisor_ready_phd_proposal.pdf` or older CEMS-based evidence summaries as the current proposal. Their preliminary-result claims predate the September 2026 frozen comparison and lineage audit.

## Core message for supervisors

The applicant has a functioning semantic compression prototype and a measurable research problem. On a 60-event-pair Sentinel-2 validation comparison, adaptation improved a base VQ-VAE, but JPEG2000 remained stronger at the tested 1,200-byte ceiling. The proposal investigates whether independent labels, rigorous byte accounting, event/footprint separation and better learned token selection can establish a genuine utility advantage, or define the regimes in which conventional compression wins. The hypothesis is **unconfirmed**.

## Before each submission

- Match the proposal to a real supervisor's research facilities and current funded project without changing the scientific claim.
- Confirm the programme's page limit, funding eligibility, application deadline and required referee format from its official call.
- Use an academic CV focused on the MSc project, methods, open-source work, reproducibility and research outputs; do not label an unaccepted manuscript as a publication.
- Keep any outreach email shorter than the proposal and attach the current proposal plus evidence annex only when appropriate.
- Do not describe the sealed EcoFireBias test cohort as evaluated, the quantised dNBR mask as ground truth, or the present model as superior to JPEG2000.

No proposal can guarantee admission. The current pack supports an honest, technically credible application; references, fit, funding and a supervisor's capacity remain decisive external factors.
