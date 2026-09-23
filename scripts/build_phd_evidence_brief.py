"""Build a two-page supervisor evidence annex from frozen validation results."""

from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
    Spacer, Table, TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/future_satellite_cohort_audit/ecofirebias_official_validation/VALIDATION_REPORT.json"
OUTPUT = ROOT / "reports/phd_preliminary_evidence_brief_2026.pdf"
INK = colors.HexColor("#16324D")
TEAL = colors.HexColor("#176A70")
MUTED = colors.HexColor("#54616B")
RULE = colors.HexColor("#D5DDE0")


def styles() -> dict[str, ParagraphStyle]:
    return {
        "title": ParagraphStyle("Title", fontName="Helvetica-Bold", fontSize=18, leading=21,
                                textColor=INK, spaceAfter=7),
        "subtitle": ParagraphStyle("Subtitle", fontName="Helvetica", fontSize=9, leading=13,
                                   textColor=MUTED, spaceAfter=5),
        "section": ParagraphStyle("Section", fontName="Helvetica-Bold", fontSize=10.5, leading=13,
                                  textColor=INK, spaceBefore=12, spaceAfter=5),
        "body": ParagraphStyle("Body", fontName="Helvetica", fontSize=9.2, leading=13.1,
                               textColor=INK, spaceAfter=7),
        "small": ParagraphStyle("Small", fontName="Helvetica", fontSize=8.1, leading=11.1,
                                textColor=MUTED, spaceAfter=5),
        "callout": ParagraphStyle("Callout", fontName="Helvetica-Bold", fontSize=9.5, leading=14,
                                  textColor=TEAL, spaceAfter=8),
        "thead": ParagraphStyle("TH", fontName="Helvetica-Bold", fontSize=8, leading=10,
                                textColor=colors.white, alignment=TA_CENTER),
        "td": ParagraphStyle("TD", fontName="Helvetica", fontSize=8.1, leading=10.2,
                             textColor=INK, alignment=TA_CENTER),
        "tdleft": ParagraphStyle("TDL", fontName="Helvetica", fontSize=8.1, leading=10.2,
                                 textColor=INK, alignment=TA_LEFT),
    }


def p(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def footer(canvas, doc) -> None:
    canvas.saveState()
    width, _ = A4
    canvas.setStrokeColor(RULE)
    canvas.line(45, 43, width - 45, 43)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(45, 31, "Athul Jayakumar | Preliminary research evidence | September 2026")
    canvas.drawRightString(width - 45, 31, f"{doc.page}")
    canvas.restoreState()


def build() -> Path:
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    if data["events"] != 60 or data["sealed_test_scored"] or data["byte_ceiling"] != 1200:
        raise ValueError("Frozen evidence status differs from the brief")
    summaries = data["summaries"]
    diffs = data["paired_adapted_minus"]
    style = styles()
    story = []

    story.extend([
        p("Preliminary PhD Research Evidence", style["title"]),
        p("Semantic Utility-Aware Compression for Wildfire-Centric Earth Observation", style["subtitle"]),
        p("Applicant: Athul Jayakumar  |  Evidence annex, not a claim of method superiority", style["small"]),
        HRFlowable(width="100%", thickness=1, color=TEAL, spaceAfter=7),
        p("Research problem", style["section"]),
        p("Under restricted satellite downlink, which image information should be sent first? This project asks whether wildfire-aware ranking of learned image tokens can retain mission-relevant evidence more efficiently than distortion-optimised compression. Wildfire remains the primary mission. The central hypothesis is <b>not yet confirmed</b>: JPEG2000 leads in the current matched-ceiling study.", style["body"]),
        p("System and pre-scoring design", style["section"]),
        p("Post-fire Sentinel-2 RGB is analysed for wildfire utility, encoded as VQ-VAE tokens, ranked, serialised with a retention mask, transmitted and reconstructed. The one-time official-validation comparison covers <b>60 event pairs / 120 images</b>, with one burn and one nearby negative chip per event. Adaptation used a separate 600-event source-training selection (480 gradient-training, 120 internal validation). All methods were constrained to <b>at most 1,200 actual serialised bytes per native 224 x 224 image</b>; JPEG/JPEG2000 settings were selected for original-image PSNR under that ceiling. Payloads are under a common cap, not identical in size.", style["body"]),
        p("What was measured", style["section"]),
        p("SUS is a model-dependent 0-100 combination of detector confidence, object, relevance and important-region retention. Proxy Dice compares predicted regions with a separate dNBR-derived mask; negative predicted-positive area indicates false-alarm burden on nominally negative chips, but is not a calibrated false-positive rate. PSNR, SSIM and actual serialised bytes measure visual and communication performance. No single endpoint establishes operational wildfire benefit.", style["body"]),
        p("Main result on 60 burn chips", style["section"]),
    ])

    headers = ["Method", "SUS /100", "Proxy<br/>Dice", "PSNR<br/>(dB)", "SSIM", "Mean<br/>bytes"]
    rows = [[p(h, style["thead"]) for h in headers]]
    for name in ("JPEG", "JPEG2000 RDO", "Base VQ-VAE", "Adapted VQ-VAE"):
        s = summaries[name]
        values = [name, f"{s['burn_sus']:.2f}", f"{s['burn_proxy_dice']:.3f}",
                  f"{s['burn_psnr_db']:.2f}", f"{s['burn_ssim']:.3f}",
                  f"{s['mean_serialized_bytes']:,.1f}"]
        rows.append([p(value, style["tdleft"] if col == 0 else style["td"]) for col, value in enumerate(values)])
    table = Table(rows, colWidths=[119, 65, 70, 65, 65, 77], repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), INK),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4F7F8")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LINEBELOW", (0, -1), (-1, -1), 0.7, RULE),
    ]))
    story.extend([table, Spacer(1, 5)])
    adapted_base = diffs["Base VQ-VAE"]
    adapted_jp2 = diffs["JPEG2000 RDO"]
    story.extend([
        p("Paired differences (adapted VQ-VAE minus comparator)", style["section"]),
        p(f"Versus base: <b>+{adapted_base['sus']['mean_difference']:.2f} SUS</b> "
          f"(95% event-bootstrap CI +{adapted_base['sus']['ci_low']:.2f} to +{adapted_base['sus']['ci_high']:.2f}) "
          f"and <b>+{adapted_base['label_dice']['mean_difference']:.3f} proxy Dice</b> "
          f"(+{adapted_base['label_dice']['ci_low']:.3f} to +{adapted_base['label_dice']['ci_high']:.3f}).", style["body"]),
        p(f"Versus JPEG2000: <b>{adapted_jp2['sus']['mean_difference']:.2f} SUS</b> "
          f"(CI {adapted_jp2['sus']['ci_low']:.2f} to {adapted_jp2['sus']['ci_high']:.2f}) "
          f"and <b>{adapted_jp2['label_dice']['mean_difference']:.3f} proxy Dice</b> "
          f"({adapted_jp2['label_dice']['ci_low']:.3f} to {adapted_jp2['label_dice']['ci_high']:.3f}). "
          "Ten thousand event-paired bootstrap resamples were used.", style["body"]),
        PageBreak(),
        p("Interpretation and research programme", style["title"]),
        HRFlowable(width="100%", thickness=1, color=TEAL, spaceAfter=7),
        p("What the result does and does not establish", style["section"]),
        p("Training adaptation changes wildfire-relevant retention and improves the VQ-VAE relative to its base checkpoint. It does <b>not</b> show an advantage over JPEG2000. Negative-chip predicted-positive area rose from "
          f"{summaries['Base VQ-VAE']['negative_predicted_positive_fraction']:.3f} to "
          f"{summaries['Adapted VQ-VAE']['negative_predicted_positive_fraction']:.3f}; "
          f"the paired increase is +{adapted_base['predicted_positive_fraction']['mean_difference']:.3f} "
          f"(CI +{adapted_base['predicted_positive_fraction']['ci_low']:.3f} to "
          f"+{adapted_base['predicted_positive_fraction']['ci_high']:.3f}). "
          "The separately frozen 18-event development gate failed, and the 37-pair screened test cohort remains sealed and unscored.", style["body"]),
        p("Validity limits", style["section"]),
        p("Validation event IDs are distinct from selected training and development IDs, but every selected validation country occurs in training; patch-level IDs do not prove scene-footprint separation. The burn mask is a quantised dNBR spectral proxy, not a manually checked burn-scar boundary. Five images required reprojection with uncovered edges excluded. SUS and detector retention share a detector, so they are not independent endpoints. No LPIPS or measured onboard energy result is claimed here.", style["body"]),
        p("Three focused PhD questions", style["section"]),
        p("<b>1. Measurement:</b> Can utility estimates be validated against independent geospatial fire annotations, with calibrated uncertainty and negative-scene specificity?", style["body"]),
        p("<b>2. Representation:</b> At matched serialised byte budgets, can learned token prioritisation improve both SUS and independent-mask Dice over JPEG2000 without increasing false alarms?", style["body"]),
        p("<b>3. Mission feasibility:</b> When do any measured utility gains justify onboard latency, memory, compute and downlink cost?", style["body"]),
        p("First-year validation sequence", style["section"]),
        p("Secure a licensed independently annotated Sentinel-2 cohort; verify exact event, footprint and geographic separation from known training; fix preprocessing, byte accounting, endpoints and thresholds before scoring; run a paired codec comparison across byte budgets with both positive and negative scenes. Report negative findings if the conventional baselines remain stronger. Keep the old failed gate and sealed cohort unchanged.", style["body"]),
        p("Bounded four-year trajectory", style["section"]),
        p("<b>Year 1:</b> provenance, independent labels and frozen evaluation protocol. "
          "<b>Year 2:</b> representation and token-ranking studies confined to training/development data, with false-alarm checks. "
          "<b>Year 3:</b> measured edge-device inference and serialisation costs alongside clearly labelled link simulations. "
          "<b>Year 4:</b> one untouched external replication after predeclared advancement, papers including negative findings, and thesis synthesis. "
          "If JPEG2000 remains stronger, identifying the conditions and measurement limits behind that failure is still a valid research outcome.", style["body"]),
        p("Reproducibility note", style["section"]),
        p("Frozen selection, hashes, per-image rows and paired statistics: "
          "results/future_satellite_cohort_audit/ecofirebias_official_validation/. "
          "Source dataset: EcoFireBias / wildfire_global, Hugging Face revision "
          "39f331e50458fba3669837d69fc2fdddbb1d1d69. "
          "This annex supersedes older positive-sounding evidence summaries; the existing proposal was not edited.", style["small"]),
    ])
    doc = SimpleDocTemplate(str(OUTPUT), pagesize=A4, rightMargin=45, leftMargin=45,
                            topMargin=42, bottomMargin=56, title="Preliminary PhD Research Evidence",
                            author="Athul Jayakumar")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return OUTPUT


if __name__ == "__main__":
    print(build())
