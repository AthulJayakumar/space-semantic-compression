"""Generate Priority 1 PDF packages for PhD, patent, and visa use.

Outputs:
- application_packages/phd_supervisor_package.pdf
- application_packages/founder_visa_package.pdf
- application_packages/private/patent_attorney_brief_private.pdf

The private patent brief is written under an ignored folder and should not be
published before patent-attorney review.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "application_packages"
PRIVATE_OUT = OUT / "private"
FIG = ROOT / "figures/project_diagrams"


BLUE = colors.HexColor("#2E86AB")
GREEN = colors.HexColor("#41A368")
ORANGE = colors.HexColor("#F28E2B")
RED = colors.HexColor("#C44536")
DARK = colors.HexColor("#263238")
LIGHT = colors.HexColor("#F6F8FB")
MID = colors.HexColor("#D8DEE8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    PRIVATE_OUT.mkdir(parents=True, exist_ok=True)
    build_phd_package(OUT / "phd_supervisor_package.pdf")
    build_founder_visa_package(OUT / "founder_visa_package.pdf")
    build_patent_package(PRIVATE_OUT / "patent_attorney_brief_private.pdf")
    print(OUT)


def styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "Title",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=24,
            leading=30,
            textColor=DARK,
            alignment=TA_CENTER,
            spaceAfter=14,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=12,
            leading=16,
            textColor=colors.HexColor("#455A64"),
            alignment=TA_CENTER,
            spaceAfter=20,
        ),
        "h1": ParagraphStyle(
            "Heading1",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=19,
            textColor=BLUE,
            spaceBefore=10,
            spaceAfter=8,
        ),
        "h2": ParagraphStyle(
            "Heading2",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=15,
            textColor=DARK,
            spaceBefore=8,
            spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=13,
            textColor=DARK,
            alignment=TA_LEFT,
            spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "Small",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#546E7A"),
        ),
        "bullet": ParagraphStyle(
            "Bullet",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.2,
            leading=12.5,
            leftIndent=12,
            firstLineIndent=-8,
            textColor=DARK,
            spaceAfter=4,
        ),
        "callout": ParagraphStyle(
            "Callout",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=13,
            textColor=DARK,
            backColor=colors.HexColor("#EDF6F9"),
            borderColor=colors.HexColor("#B7DDE8"),
            borderWidth=0.6,
            borderPadding=7,
            spaceBefore=7,
            spaceAfter=8,
        ),
    }


def header_footer(canvas, doc, title: str, confidential: bool = False) -> None:
    canvas.saveState()
    width, height = A4
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#607D8B"))
    label = "PRIVATE - DO NOT PUBLISH" if confidential else title
    canvas.drawString(doc.leftMargin, height - 1.05 * cm, label)
    canvas.drawRightString(width - doc.rightMargin, 0.8 * cm, f"Page {doc.page}")
    canvas.setStrokeColor(colors.HexColor("#D7DEE8"))
    canvas.line(doc.leftMargin, height - 1.22 * cm, width - doc.rightMargin, height - 1.22 * cm)
    canvas.restoreState()


def doc_template(path: Path, title: str, confidential: bool = False) -> SimpleDocTemplate:
    return SimpleDocTemplate(
        str(path),
        pagesize=A4,
        rightMargin=1.55 * cm,
        leftMargin=1.55 * cm,
        topMargin=1.65 * cm,
        bottomMargin=1.3 * cm,
        title=title,
        author="Athul Jayakumar",
    )


def build_pdf(path: Path, title: str, story: list, confidential: bool = False) -> None:
    doc = doc_template(path, title, confidential)
    doc.build(
        story,
        onFirstPage=lambda canvas, document: header_footer(canvas, document, title, confidential),
        onLaterPages=lambda canvas, document: header_footer(canvas, document, title, confidential),
    )


def p(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def bullets(items: Iterable[str], style: ParagraphStyle) -> list[Paragraph]:
    return [p(f"- {item}", style) for item in items]


def image(path: Path, width_cm: float) -> Image:
    img = Image(str(path))
    scale = (width_cm * cm) / img.imageWidth
    img.drawWidth = img.imageWidth * scale
    img.drawHeight = img.imageHeight * scale
    return img


def table(data: list[list[str]], widths: list[float] | None = None) -> Table:
    col_widths = [w * cm for w in widths] if widths else None
    tbl = Table([[Paragraph(str(cell), styles()["small"]) for cell in row] for row in data], colWidths=col_widths, hAlign="LEFT")
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EAF3F7")),
                ("TEXTCOLOR", (0, 0), (-1, 0), DARK),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CFD8DC")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#FAFBFC")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return tbl


def title_page(style: dict[str, ParagraphStyle], title: str, subtitle: str, tag: str, confidential: bool = False) -> list:
    story: list = [
        Spacer(1, 1.2 * cm),
        p(title, style["title"]),
        p(subtitle, style["subtitle"]),
        Spacer(1, 0.5 * cm),
        image(FIG / "01_system_architecture.png", 15.5),
        Spacer(1, 0.4 * cm),
        p(tag, style["callout"]),
    ]
    if confidential:
        story.append(p("Private document for patent-attorney review. Do not publish, upload, or send without legal guidance.", style["callout"]))
    story.append(PageBreak())
    return story


def build_phd_package(path: Path) -> None:
    s = styles()
    story = title_page(
        s,
        "PhD Supervisor Package",
        "Semantic Utility-Aware Compression for Wildfire-Centric Earth Observation Systems",
        "Purpose: provide a concise research package for supervisors in Earth Observation, onboard AI, semantic communication, and neural compression.",
    )

    story += [
        p("1. Research Summary", s["h1"]),
        p(
            "This project investigates whether satellite imagery can be compressed and transmitted according to mission relevance rather than pixel-level distortion alone. The current wildfire-focused system ranks learned visual tokens by semantic utility and evaluates whether wildfire-relevant information survives compression and reconstruction.",
            s["body"],
        ),
        p("Core hypothesis", s["h2"]),
        p(
            "Semantic utility-aware token prioritisation preserves wildfire-relevant Earth Observation information more effectively than conventional distortion-optimised compression under equivalent bandwidth constraints.",
            s["callout"],
        ),
        p("Research gap", s["h2"]),
        p(
            "Existing image compression methods mainly optimise distortion, perceptual quality, or bitrate. Earth Observation workflows often evaluate downstream analytics separately. This project connects utility estimation, learned token compression, satellite communication constraints, and detector-retention evaluation in one experimental platform.",
            s["body"],
        ),
        p("2. Methodology Pipeline", s["h1"]),
        image(FIG / "02_working_pipeline.png", 16.0),
        Spacer(1, 0.25 * cm),
        p("The pipeline converts wildfire and Sentinel-2 imagery into learned VQ-VAE tokens, ranks those tokens according to semantic mission utility, transmits only the selected token subset, reconstructs imagery, and evaluates both reconstruction quality and semantic utility retention.", s["body"]),
        PageBreak(),
        p("3. Current Experimental Evidence", s["h1"]),
        table(
            [
                ["Model", "SUS", "Detector retention", "PSNR", "SSIM", "Compression ratio", "Bandwidth saved"],
                ["Original VQ-VAE", "80.470", "0.8642", "21.305", "0.8170", "69.76x", "98.56%"],
                ["Regularized mixed-domain VQ-VAE", "81.159", "0.8403", "22.631", "0.8354", "68.60x", "98.53%"],
            ],
            [4.2, 1.7, 2.5, 1.7, 1.7, 2.3, 2.4],
        ),
        Spacer(1, 0.25 * cm),
        image(FIG / "06_sentinel2_metric_comparison.png", 16.4),
        p(
            "Interpretation: satellite-domain fine-tuning improves PSNR and SSIM, but detector retention drops. This is scientifically useful because it shows that visual reconstruction quality and mission utility are not identical objectives.",
            s["callout"],
        ),
        PageBreak(),
        p("4. Statistical Result and Model Decision", s["h1"]),
        image(FIG / "07_paired_effects_ci.png", 15.2),
        Spacer(1, 0.25 * cm),
        image(FIG / "08_model_decision_tradeoff.png", 12.0),
        p(
            "Decision: the original VQ-VAE remains the conservative default for mission-utility experiments because it preserves detector retention better on the 500-patch Sentinel-2 benchmark. The regularized model is retained as an experimental reconstruction-improved candidate.",
            s["body"],
        ),
        p("5. Proposed PhD Direction", s["h1"]),
        table(
            [
                ["Year", "Research focus", "Expected output"],
                ["1", "Formalise wildfire utility metrics, improve datasets, validate detector-retention evaluation", "Workshop paper and reproducible benchmark"],
                ["2", "Develop utility-preserving token selection and detector-retention-aware representation learning", "Conference paper and model comparison"],
                ["3", "Evaluate under realistic satellite communication constraints and cross-dataset EO settings", "Journal paper and satellite simulation study"],
                ["4", "Optimise for onboard deployment and finalise thesis framework", "Thesis, open-source release, final journal submission"],
            ],
            [1.4, 8.2, 6.4],
        ),
        p("6. Supervisor Fit", s["h1"]),
        *bullets(
            [
                "Earth Observation and remote sensing: wildfire-centric Sentinel-2 validation and mission-oriented evaluation.",
                "Onboard AI and edge computing: token pruning and compression under bandwidth constraints.",
                "Semantic communication: transmission of task-relevant information rather than all pixels.",
                "Neural compression: VQ-VAE tokenisation and reconstruction-quality versus utility trade-offs.",
            ],
            s["bullet"],
        ),
        p("Best next outreach asset: send this package with the GitHub repository, the two-page research summary, and a short demo video.", s["callout"]),
    ]
    build_pdf(path, "PhD Supervisor Package", story)


def build_founder_visa_package(path: Path) -> None:
    s = styles()
    story = title_page(
        s,
        "Founder and Visa Evidence Package",
        "Onboard AI Software for Mission-Aware Satellite Downlink Optimisation",
        "Purpose: frame the same validated research platform for Innovator Founder, Global Talent, incubator, and supervisor-facing evidence.",
    )
    story += [
        p("1. Business Proposition", s["h1"]),
        p(
            "The proposed product is an onboard AI software layer that reduces satellite downlink bandwidth while preserving mission-critical Earth Observation information. The initial application is wildfire monitoring, where preserving fire, smoke, burn-scar, and affected-terrain evidence matters more than visually perfect reconstruction of every background pixel.",
            s["body"],
        ),
        p("One-line positioning", s["h2"]),
        p("Mission-aware semantic communication software for resource-constrained Earth Observation satellites.", s["callout"]),
        p("2. Innovative, Viable, Scalable", s["h1"]),
        table(
            [
                ["Criterion", "Evidence"],
                ["Innovative", "Utility-aware learned token transmission for satellite imagery rather than generic pixel compression."],
                ["Viable", "Working API, dashboard, benchmark framework, Sentinel-2 evidence, statistical reports, and reproducible diagrams."],
                ["Scalable", "Applicable to CubeSats, EO analytics, wildfire monitoring, flood validation, maritime monitoring, and defence/civil protection use cases."],
            ],
            [3.0, 13.0],
        ),
        p("3. Platform Architecture", s["h1"]),
        image(FIG / "01_system_architecture.png", 16.0),
        PageBreak(),
        p("4. Current Technical Evidence", s["h1"]),
        table(
            [
                ["Metric", "Original VQ-VAE", "Regularized mixed VQ-VAE", "Interpretation"],
                ["SUS", "80.470", "81.159", "Candidate slightly higher, but not enough alone for promotion"],
                ["Detector retention", "0.8642", "0.8403", "Original safer for mission-utility default"],
                ["PSNR", "21.305", "22.631", "Candidate clearly improves visual reconstruction"],
                ["SSIM", "0.8170", "0.8354", "Candidate clearly improves structural similarity"],
                ["Bandwidth saved", "98.56%", "98.53%", "Both demonstrate strong communication savings"],
            ],
            [3.2, 3.0, 3.5, 6.3],
        ),
        Spacer(1, 0.25 * cm),
        image(FIG / "03_token_transmission_working_diagram.png", 15.2),
        p("5. Global Talent Evidence Mapping", s["h1"]),
        table(
            [
                ["Evidence type", "Current status", "Next action"],
                ["Original technical work", "Working semantic compression platform", "Prepare short public technical video"],
                ["Research output", "Reports, figures, Sentinel-2 validation", "Submit arXiv or workshop preprint"],
                ["Open-source contribution", "GitHub repository available", "Polish README and examples"],
                ["External validation", "Not yet enough", "Collect supervisor and incubator feedback"],
                ["Commercial promise", "Founder business case drafted", "Contact incubators and collect letters of interest"],
            ],
            [3.8, 5.7, 6.5],
        ),
        PageBreak(),
        p("6. Career Evidence Positioning", s["h1"]),
        image(FIG / "05_career_evidence_positioning.png", 15.5),
        p("7. Immediate 60-Day Plan", s["h1"]),
        *bullets(
            [
                "Create a 2 to 3 minute demo video showing image input, utility map, token selection, reconstruction, and metrics.",
                "Send the PhD package to 20 targeted supervisors in EO, semantic communication, onboard AI, and neural compression.",
                "Approach ESA BIC, Satellite Applications Catapult, and space-tech incubators with the founder package.",
                "Book patent-attorney consultation before publishing claim-level technical mechanisms.",
                "Prepare one arXiv or workshop-style preprint around mission-utility-aware semantic compression.",
            ],
            s["bullet"],
        ),
        p("This package is evidence framing, not immigration advice. It is designed to support discussions with endorsing bodies, supervisors, incubators, and legal professionals.", s["callout"]),
    ]
    build_pdf(path, "Founder and Visa Evidence Package", story)


def build_patent_package(path: Path) -> None:
    s = styles()
    story = title_page(
        s,
        "Private Patent Attorney Brief",
        "Semantic Utility-Aware Token Transmission for Resource-Constrained Earth Observation Satellites",
        "Purpose: provide a private technical disclosure for legal review before any public claim-level publication.",
        confidential=True,
    )
    story += [
        p("1. Confidentiality Notice", s["h1"]),
        p(
            "This document is private and should not be uploaded to public GitHub, arXiv, websites, pitch decks, or supervisor emails before patent-attorney review. Use NDAs for detailed commercial discussions.",
            s["callout"],
        ),
        p("2. Problem Solved", s["h1"]),
        p(
            "Earth Observation satellites collect large image volumes but face downlink, latency, contact-window, and power constraints. Existing compression approaches usually optimise bitrate, distortion, or perceptual quality. For wildfire monitoring, the technical need is to preserve mission-relevant information under communication constraints.",
            s["body"],
        ),
        p("3. Core Technical Concept", s["h1"]),
        p(
            "The system encodes satellite imagery into learned visual tokens, estimates mission-specific utility over the image, maps utility to token-level importance, ranks tokens according to utility and communication constraints, transmits a selected token subset, and evaluates reconstruction with downstream mission utility metrics.",
            s["body"],
        ),
        image(FIG / "03_token_transmission_working_diagram.png", 15.4),
        PageBreak(),
        p("4. Example Embodiment", s["h1"]),
        *bullets(
            [
                "Receive a satellite image or image tile.",
                "Generate a mission utility map using wildfire, smoke, burn-scar, saliency, or detector-derived relevance.",
                "Encode the image into learned visual tokens using a neural encoder.",
                "Map the utility estimate onto the token grid.",
                "Compute token priority using mission utility and optional entropy, cost, or reconstruction terms.",
                "Select a token subset under bandwidth, latency, or downlink constraints.",
                "Transmit selected tokens and metadata.",
                "Reconstruct imagery or downstream products from transmitted tokens.",
                "Evaluate detector retention and semantic utility before and after reconstruction.",
            ],
            s["bullet"],
        ),
        p("5. Patent-Relevant Claim Directions For Attorney Review", s["h1"]),
        table(
            [
                ["Direction", "Private drafting note"],
                ["Mission-utility token transmission", "Computer-implemented method for assigning mission utility to learned image tokens and transmitting a selected subset."],
                ["Onboard satellite processing", "Satellite payload or edge system configured to prioritise tokens before downlink."],
                ["Detector-retention preservation", "Compression process evaluated or controlled by downstream detector retention."],
                ["Adaptive downlink", "Token retention adjusted according to communication budget and mission priority."],
                ["Wildfire EO embodiment", "Wildfire, smoke, and burn-scar relevance as an example dependent embodiment."],
            ],
            [4.4, 11.4],
        ),
        p("6. Current Technical Evidence", s["h1"]),
        table(
            [
                ["Model", "SUS", "Detector retention", "PSNR", "SSIM", "Bandwidth saved"],
                ["Original VQ-VAE", "80.470", "0.8642", "21.305", "0.8170", "98.56%"],
                ["Regularized mixed-domain VQ-VAE", "81.159", "0.8403", "22.631", "0.8354", "98.53%"],
            ],
            [4.3, 2.0, 3.0, 2.0, 2.0, 2.4],
        ),
        p(
            "The strongest patent-relevant insight is that reconstruction quality and detector-facing mission utility can diverge. This supports a technical argument for mission-utility-aware compression rather than generic image compression.",
            s["callout"],
        ),
        PageBreak(),
        p("7. Prior-Art Areas To Search", s["h1"]),
        *bullets(
            [
                "semantic communication for image transmission",
                "task-aware image compression and machine-vision-oriented compression",
                "neural image compression and learned image token pruning",
                "remote sensing image compression and onboard satellite compression",
                "JPEG2000 region-of-interest coding",
                "CCSDS image data compression standards",
                "adaptive satellite downlink and onboard image prioritisation",
                "wildfire or disaster-monitoring compression systems",
            ],
            s["bullet"],
        ),
        p("8. Questions For Patent Attorney", s["h1"]),
        *bullets(
            [
                "Is the invention best framed as a satellite communication process rather than software alone?",
                "Does existing public GitHub material limit broad claim scope?",
                "Should the independent claim focus on learned token mapping, detector retention, adaptive downlink, or the full pipeline?",
                "Should wildfire be in the independent claim or only in dependent claims and examples?",
                "Should detector-retention-aware training be protected in a later filing if developed further?",
                "Should the first filing route be UK, EPO, or PCT?",
            ],
            s["bullet"],
        ),
        p("9. Public Disclosure Caution", s["h1"]),
        p(
            "Avoid public disclosure of detailed claim language, unreleased training objectives, exact technical variants, or commercial implementation details until legal advice is received. Public materials should describe high-level research results and avoid claim-level mechanism disclosure.",
            s["callout"],
        ),
    ]
    build_pdf(path, "Private Patent Attorney Brief", story, confidential=True)


if __name__ == "__main__":
    main()
