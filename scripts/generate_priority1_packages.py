"""Generate polished Priority 1 application PDF packages.

This script creates three substantial documents:

1. Public PhD supervisor package
2. Public founder / visa evidence package
3. Private patent attorney brief

The private patent brief is written to application_packages/private/, which is
ignored by Git. Do not publish that file before legal review.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "application_packages"
PRIVATE_OUT = OUT / "private"
FIG = ROOT / "figures/project_diagrams"

BLUE = colors.HexColor("#2E86AB")
DARK = colors.HexColor("#263238")
MUTED = colors.HexColor("#546E7A")
LIGHT_BLUE = colors.HexColor("#EAF3F7")
LIGHT_GREEN = colors.HexColor("#EEF7F1")
LIGHT_ORANGE = colors.HexColor("#FFF4E6")
GRID = colors.HexColor("#CFD8DC")


SENTINEL_RESULTS = [
    ["Metric", "Original VQ-VAE", "Regularized mixed-domain VQ-VAE", "Interpretation"],
    ["SUS", "80.470", "81.159", "Candidate slightly higher, but not significant enough alone for promotion."],
    ["Detector retention", "0.8642", "0.8403", "Original is safer for wildfire mission-utility preservation."],
    ["PSNR", "21.305", "22.631", "Candidate improves visual reconstruction by +1.326 dB."],
    ["SSIM", "0.8170", "0.8354", "Candidate improves structural similarity."],
    ["Compression ratio", "69.76x", "68.60x", "Original remains slightly stronger for compression ratio."],
    ["Bandwidth saved", "98.56%", "98.53%", "Both preserve very high communication savings."],
]

PAIRED_STATS = [
    ["Metric", "Mean difference", "Paired t-test p", "Bootstrap 95% CI", "Decision"],
    ["SUS", "+0.6895", "0.1741", "[-0.2347, 1.6611]", "Not conclusive by paired t-test."],
    ["Detector retention", "-0.0239", "0.000105", "[-0.0347, -0.0120]", "Significant drop; do not promote candidate as default."],
    ["PSNR", "+1.3261 dB", "9.91e-67", "[1.2015, 1.4605]", "Strong reconstruction gain."],
    ["SSIM", "+0.0184", "1.15e-18", "[0.0144, 0.0220]", "Strong structural gain."],
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    PRIVATE_OUT.mkdir(parents=True, exist_ok=True)
    build_phd_package(OUT / "phd_supervisor_package.pdf")
    build_founder_visa_package(OUT / "founder_visa_package.pdf")
    build_patent_package(PRIVATE_OUT / "patent_attorney_brief_private.pdf")
    write_readme(OUT / "README.md")
    print(OUT)


def make_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "cover_title": ParagraphStyle(
            "cover_title",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=25,
            leading=31,
            textColor=DARK,
            alignment=TA_CENTER,
            spaceAfter=12,
        ),
        "cover_subtitle": ParagraphStyle(
            "cover_subtitle",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=12,
            leading=16,
            textColor=MUTED,
            alignment=TA_CENTER,
            spaceAfter=18,
        ),
        "h1": ParagraphStyle(
            "h1",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=15.5,
            leading=19,
            textColor=BLUE,
            spaceBefore=9,
            spaceAfter=7,
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=15,
            textColor=DARK,
            spaceBefore=7,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "body",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.4,
            leading=12.8,
            textColor=DARK,
            alignment=TA_LEFT,
            spaceAfter=5.5,
        ),
        "small": ParagraphStyle(
            "small",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.7,
            leading=9.6,
            textColor=MUTED,
        ),
        "bullet": ParagraphStyle(
            "bullet",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9,
            leading=12.4,
            leftIndent=13,
            firstLineIndent=-8,
            textColor=DARK,
            spaceAfter=3.7,
        ),
        "callout": ParagraphStyle(
            "callout",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=9.5,
            leading=12.6,
            textColor=DARK,
            backColor=LIGHT_BLUE,
            borderColor=colors.HexColor("#B7DDE8"),
            borderWidth=0.6,
            borderPadding=7,
            spaceBefore=6,
            spaceAfter=8,
        ),
        "warning": ParagraphStyle(
            "warning",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=9.5,
            leading=12.6,
            textColor=DARK,
            backColor=LIGHT_ORANGE,
            borderColor=colors.HexColor("#E0B86E"),
            borderWidth=0.6,
            borderPadding=7,
            spaceBefore=6,
            spaceAfter=8,
        ),
    }


def para(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def bullet_list(items: Iterable[str], style: ParagraphStyle) -> list[Paragraph]:
    return [para(f"- {item}", style) for item in items]


def fig(path: Path, width_cm: float) -> Image:
    image = Image(str(path))
    scale = (width_cm * cm) / image.imageWidth
    image.drawWidth = image.imageWidth * scale
    image.drawHeight = image.imageHeight * scale
    return image


def make_table(data: list[list[str]], widths: list[float] | None = None, header_color=LIGHT_BLUE) -> Table:
    styles = make_styles()
    col_widths = [w * cm for w in widths] if widths else None
    converted = [[Paragraph(str(cell), styles["small"]) for cell in row] for row in data]
    table = Table(converted, colWidths=col_widths, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), header_color),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.35, GRID),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#FAFBFC")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def header_footer(canvas, doc, title: str, confidential: bool = False) -> None:
    canvas.saveState()
    width, height = A4
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    label = "PRIVATE - DO NOT PUBLISH" if confidential else title
    canvas.drawString(doc.leftMargin, height - 1.05 * cm, label)
    canvas.drawRightString(width - doc.rightMargin, 0.8 * cm, f"Page {doc.page}")
    canvas.setStrokeColor(colors.HexColor("#D7DEE8"))
    canvas.line(doc.leftMargin, height - 1.22 * cm, width - doc.rightMargin, height - 1.22 * cm)
    canvas.restoreState()


def build_pdf(path: Path, title: str, story: list, confidential: bool = False) -> None:
    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        rightMargin=1.45 * cm,
        leftMargin=1.45 * cm,
        topMargin=1.65 * cm,
        bottomMargin=1.25 * cm,
        title=title,
        author="Athul Jayakumar",
    )
    doc.build(
        story,
        onFirstPage=lambda canvas, document: header_footer(canvas, document, title, confidential),
        onLaterPages=lambda canvas, document: header_footer(canvas, document, title, confidential),
    )


def cover(styles: dict[str, ParagraphStyle], title: str, subtitle: str, purpose: str, confidential: bool = False) -> list:
    elements: list = [
        Spacer(1, 0.9 * cm),
        para(title, styles["cover_title"]),
        para(subtitle, styles["cover_subtitle"]),
        fig(FIG / "01_system_architecture.png", 15.8),
        Spacer(1, 0.35 * cm),
        para(purpose, styles["callout"]),
    ]
    if confidential:
        elements.append(para("Private attorney-facing material. Do not publish, upload, or circulate without legal review.", styles["warning"]))
    elements.append(PageBreak())
    return elements


def build_phd_package(path: Path) -> None:
    s = make_styles()
    story = cover(
        s,
        "PhD Supervisor Package",
        "Semantic Utility-Aware Compression for Wildfire-Centric Earth Observation Systems",
        "Purpose: a supervisor-ready research package with problem framing, methodology, architecture, experiments, results, limitations, and a 4-year PhD plan.",
    )
    story += [
        para("1. Executive Summary", s["h1"]),
        para(
            "This project investigates semantic communication for Earth Observation satellites. The central idea is to transmit learned image tokens according to wildfire mission utility instead of compressing every pixel equally. The current system uses a VQ-VAE encoder/decoder, semantic utility estimation, utility-aware token ranking, satellite communication metrics, and detector-retention evaluation.",
            s["body"],
        ),
        para(
            "The key scientific finding so far is that better visual reconstruction does not automatically mean better mission utility. In the 500-patch Sentinel-2 benchmark, the regularized mixed-domain VQ-VAE improves PSNR and SSIM, but it reduces detector retention. This motivates a PhD research programme on utility-preserving representation learning and mission-aware semantic compression.",
            s["callout"],
        ),
        para("Core research hypothesis", s["h2"]),
        para(
            "Semantic utility-aware token prioritisation preserves wildfire-relevant Earth Observation information more effectively than conventional distortion-optimised compression under equivalent bandwidth constraints.",
            s["body"],
        ),
        para("Supervisor relevance", s["h2"]),
        *bullet_list(
            [
                "Earth Observation and remote sensing: Sentinel-2 wildfire-relevant validation and satellite imagery constraints.",
                "Onboard AI: token selection under constrained communication budgets.",
                "Semantic communication: transmission of task-relevant information rather than all pixels.",
                "Neural compression: VQ-VAE tokenisation, reconstruction, and detector-retention trade-offs.",
            ],
            s["bullet"],
        ),
        para("2. Research Gap and Questions", s["h1"]),
        para(
            "Most image compression research optimises distortion, perceptual quality, or bitrate. Earth Observation systems often evaluate downstream interpretation separately from compression. This project addresses the gap between compression quality and mission value by directly measuring whether wildfire-relevant information survives compression and reconstruction.",
            s["body"],
        ),
        make_table(
            [
                ["Research question", "Methodology", "Measurable outcome"],
                ["RQ1: Can wildfire relevance be represented as a semantic utility signal?", "Utility maps from wildfire, smoke, burn-scar, saliency, and detector cues.", "Utility-map coverage, relevance retention, region preservation."],
                ["RQ2: Can VQ-VAE tokens be prioritised by mission utility?", "Token-grid mapping, utility-aware ranking, token pruning.", "SUS, detector retention, token count, compression ratio."],
                ["RQ3: Does mission utility diverge from visual quality?", "Compare original and fine-tuned VQ-VAE checkpoints.", "PSNR/SSIM versus SUS/detector-retention trade-off."],
                ["RQ4: Is this practical for satellite communication?", "Payload estimation and bandwidth-saving analysis.", "Bandwidth saved, compression ratio, transmission reduction."],
            ],
            [4.1, 6.0, 6.0],
        ),
        PageBreak(),
        para("3. System Architecture", s["h1"]),
        fig(FIG / "01_system_architecture.png", 16.2),
        para(
            "The implementation is modular: the frontend supports mission-control-style interaction, FastAPI exposes compression and analysis endpoints, service layers isolate semantic analysis and VQ-VAE encoding/decoding, and the benchmark layer produces CSV, JSON, figures, and statistical reports.",
            s["body"],
        ),
        make_table(
            [
                ["Module", "Role in research platform"],
                ["Semantic Utility Service", "Detects or estimates wildfire-relevant regions and produces mission utility maps."],
                ["Encoder / Decoder Services", "Load VQ-VAE checkpoints and perform tokenisation and reconstruction."],
                ["Token Prioritisation", "Ranks tokens using mission utility, entropy, and communication cost."],
                ["Metrics and Transmission Services", "Compute SUS, detector retention, PSNR, SSIM, compression ratio, and bandwidth savings."],
                ["Benchmark and Statistics", "Runs paired experiments, bootstrap confidence intervals, t-tests, Wilcoxon tests, and result tables."],
            ],
            [4.4, 11.6],
        ),
        para("4. Working Methodology", s["h1"]),
        fig(FIG / "02_working_pipeline.png", 16.0),
        para(
            "The pipeline receives a Sentinel-2 or wildfire image, estimates semantic utility, encodes the image into learned tokens, ranks tokens according to mission value, simulates bandwidth-constrained transmission, reconstructs the image, and evaluates semantic utility retention.",
            s["body"],
        ),
        PageBreak(),
        para("5. Token Transmission Mechanism", s["h1"]),
        fig(FIG / "03_token_transmission_working_diagram.png", 15.3),
        para(
            "The system treats the encoded image as a token grid. Important wildfire-relevant regions receive higher priority, while low-utility background regions can be pruned more aggressively. This creates an experimental semantic communication system rather than a standard distortion-only compression pipeline.",
            s["body"],
        ),
        para("6. Benchmark and Validation Workflow", s["h1"]),
        fig(FIG / "04_experiment_validation_pipeline.png", 15.5),
        make_table(
            [
                ["Benchmark element", "Current implementation"],
                ["Datasets", "CEMS-HLS wildfire imagery and Sentinel-2 benchmark patches."],
                ["Model variants", "Original VQ-VAE, CEMS-only fine-tune, mixed-domain fine-tune, regularized mixed-domain fine-tune."],
                ["Metrics", "SUS, detector retention, PSNR, SSIM, compression ratio, bandwidth saved, inference latency."],
                ["Statistics", "Paired t-test, Wilcoxon signed-rank, bootstrap confidence intervals."],
                ["Decision rule", "Do not promote a checkpoint if detector retention drops significantly, even if PSNR/SSIM improve."],
            ],
            [4.0, 12.0],
        ),
        PageBreak(),
        para("7. Sentinel-2 500-Patch Evidence", s["h1"]),
        make_table(SENTINEL_RESULTS, [3.2, 3.0, 4.0, 6.0]),
        Spacer(1, 0.25 * cm),
        fig(FIG / "06_sentinel2_metric_comparison.png", 16.1),
        para(
            "This figure is the most important current result for supervisors. It shows why the project is not merely an engineering demo: mission utility and conventional image quality disagree. The candidate model looks better under PSNR and SSIM, but the original model is safer for detector retention.",
            s["callout"],
        ),
        PageBreak(),
        para("8. Statistical Interpretation", s["h1"]),
        fig(FIG / "07_paired_effects_ci.png", 14.4),
        Spacer(1, 0.2 * cm),
        make_table(PAIRED_STATS, [3.0, 3.0, 3.0, 3.6, 4.0]),
        para("9. Model Promotion Decision", s["h1"]),
        fig(FIG / "08_model_decision_tradeoff.png", 10.8),
        para(
            "Decision: the regularized mixed-domain checkpoint should not replace the original checkpoint yet. It is a strong reconstruction-improved candidate, but the detector-retention drop is statistically significant. This makes the original checkpoint the safer public default for mission-utility experiments.",
            s["warning"],
        ),
        PageBreak(),
        para("10. Contributions and Publication Potential", s["h1"]),
        make_table(
            [
                ["Contribution", "Why it is publishable"],
                ["Semantic Utility Score and detector-retention evaluation", "Moves evaluation beyond PSNR/SSIM toward mission-specific EO usefulness."],
                ["Utility-aware token prioritisation", "Demonstrates semantic communication using learned image-token representations."],
                ["Satellite communication framing", "Connects compression to bandwidth, downlink, and resource-constrained operation."],
                ["Cross-model checkpoint evidence", "Shows that satellite fine-tuning can improve visual quality while harming mission utility."],
                ["Reproducible platform", "Supports further ablation, benchmark, and onboard AI experiments."],
            ],
            [5.1, 10.8],
        ),
        para("11. Limitations", s["h1"]),
        *bullet_list(
            [
                "Current detector and utility maps are research-grade and require further validation against expert wildfire labels.",
                "The regularized VQ-VAE is not yet the default because detector retention drops on the larger Sentinel-2 test.",
                "JPEG/JPEG2000 remain strong visual baselines and should be positioned honestly.",
                "More geographically diverse Sentinel-2 wildfire scenes are needed for journal-level claims.",
                "Onboard deployment has been simulated; hardware validation on Jetson or satellite-class compute remains future work.",
            ],
            s["bullet"],
        ),
        para("12. Four-Year PhD Plan", s["h1"]),
        make_table(
            [
                ["Year", "Research focus", "Expected outputs"],
                ["1", "Formalise wildfire utility metrics, strengthen datasets, validate semantic utility maps.", "Workshop paper, reproducible benchmark, supervisor-aligned proposal."],
                ["2", "Develop utility-preserving token selection and detector-retention-aware representation learning.", "Conference paper, open-source model comparison."],
                ["3", "Evaluate realistic satellite communication constraints and cross-dataset Earth Observation generalisation.", "Journal paper, mission-scale bandwidth analysis."],
                ["4", "Optimise for onboard deployment and finalise thesis contributions.", "Thesis, final journal paper, documented platform release."],
            ],
            [1.4, 7.5, 7.2],
        ),
        para(
            "Best use: send this PDF with a short email, GitHub link, and a 2-page summary. It is designed to help a supervisor understand the problem, the evidence, and the PhD direction quickly.",
            s["callout"],
        ),
    ]
    build_pdf(path, "PhD Supervisor Package", story)


def build_founder_visa_package(path: Path) -> None:
    s = make_styles()
    story = cover(
        s,
        "Founder and Visa Evidence Package",
        "Onboard AI Software for Mission-Aware Satellite Downlink Optimisation",
        "Purpose: a structured evidence package for Innovator Founder preparation, Global Talent evidence planning, incubator outreach, and space-tech discussions.",
    )
    story += [
        para("1. Executive Business Summary", s["h1"]),
        para(
            "The proposed venture develops onboard AI software that reduces satellite downlink bandwidth while preserving mission-critical Earth Observation information. The first mission focus is wildfire monitoring, where preserving fire, smoke, burn-scar, and affected-terrain evidence matters more than reconstructing every background pixel perfectly.",
            s["body"],
        ),
        para("Positioning statement", s["h2"]),
        para("Mission-aware semantic communication software for resource-constrained Earth Observation satellites.", s["callout"]),
        para("2. Problem and Customer Pain", s["h1"]),
        *bullet_list(
            [
                "Small satellites and CubeSats face limited downlink capacity, power budgets, and contact windows.",
                "Emergency monitoring needs rapid transmission of useful information, not necessarily full-resolution imagery.",
                "Conventional compression optimises visual fidelity, but mission teams care about whether operational information survives.",
                "Wildfire monitoring requires prioritising active fire, smoke, burn scars, and affected terrain.",
            ],
            s["bullet"],
        ),
        para("3. Product Concept", s["h1"]),
        fig(FIG / "03_token_transmission_working_diagram.png", 15.0),
        para(
            "The product is a software layer that sits between onboard sensing and downlink. It analyses image utility, encodes imagery into learned tokens, prioritises mission-relevant tokens, and transmits a smaller payload that preserves operational value.",
            s["body"],
        ),
        PageBreak(),
        para("4. Innovative, Viable, Scalable", s["h1"]),
        make_table(
            [
                ["Criterion", "Evidence"],
                ["Innovative", "The system is not generic compression; it prioritises learned satellite-image tokens according to mission utility and communication constraints."],
                ["Viable", "The repository contains a working backend, frontend, VQ-VAE pipeline, benchmark framework, Sentinel-2 validation, diagrams, and reproducible reports."],
                ["Scalable", "The method can extend from wildfire to flood, maritime, infrastructure, defence, and civil protection missions."],
                ["Founder credibility", "The project has real code, test coverage, benchmark evidence, and structured application packages."],
            ],
            [3.1, 12.9],
            header_color=LIGHT_GREEN,
        ),
        para("5. Platform Architecture", s["h1"]),
        fig(FIG / "01_system_architecture.png", 16.2),
        para("6. Evidence From Sentinel-2 Validation", s["h1"]),
        make_table(SENTINEL_RESULTS, [3.2, 3.0, 4.0, 6.0]),
        PageBreak(),
        para("7. Communication Efficiency Result", s["h1"]),
        fig(FIG / "06_sentinel2_metric_comparison.png", 16.0),
        para(
            "The key commercial result is not that the system beats every existing codec. The stronger claim is that it provides a mission-aware framework for deciding what information to preserve when communication is constrained. Current benchmark evidence shows very high bandwidth saving while exposing a measurable trade-off between reconstruction quality and detector utility.",
            s["callout"],
        ),
        para("8. Current Product Readiness", s["h1"]),
        make_table(
            [
                ["Area", "Current status", "What is still needed"],
                ["Prototype", "Working API, dashboard, encoder/decoder, token selector, benchmark scripts.", "Short demo video and hosted demo walkthrough."],
                ["Evidence", "Sentinel-2 benchmark, statistical results, diagrams, reports.", "More external dataset validation and third-party feedback."],
                ["IP", "Private invention disclosure and attorney brief drafted.", "Patent attorney review before public claim-level disclosure."],
                ["Commercial", "Founder/visa package prepared.", "Customer interviews, letters of interest, incubator feedback."],
                ["Deployment", "Docker and modular services exist.", "Jetson/onboard-class hardware validation."],
            ],
            [3.0, 6.1, 6.8],
        ),
        PageBreak(),
        para("9. Global Talent Evidence Mapping", s["h1"]),
        make_table(
            [
                ["Evidence category", "Current evidence", "Next action"],
                ["Innovation", "Original semantic satellite compression prototype.", "Publish a technical preprint and record demo."],
                ["Technical contribution", "GitHub repository, modular architecture, benchmark framework.", "Collect expert reviews from EO and compression researchers."],
                ["Research promise", "PhD proposal, statistical benchmark, diagrams, supervisor package.", "Begin supervisor outreach and workshop submission."],
                ["Commercial promise", "Founder package, IP brief, space-sector use case.", "Collect customer discovery notes and letters of interest."],
                ["External validation", "Not yet enough.", "Request written feedback from supervisors, incubators, and patent attorney."],
            ],
            [3.5, 6.5, 5.7],
        ),
        para("10. Founder Route Positioning", s["h1"]),
        fig(FIG / "05_career_evidence_positioning.png", 15.4),
        para("11. Target Organisations", s["h1"]),
        make_table(
            [
                ["Category", "Targets"],
                ["Space incubators", "ESA BIC, Satellite Applications Catapult, UK space accelerators, Luxembourg space ecosystem."],
                ["Research groups", "University of Luxembourg SnT, TU Delft, DLR, ESA Phi-Lab aligned groups, UK EO and onboard AI labs."],
                ["Early users", "CubeSat teams, EO analytics companies, wildfire monitoring teams, environmental monitoring groups."],
                ["Advisors", "Patent attorney, EO scientist, satellite systems engineer, startup mentor."],
            ],
            [3.4, 12.3],
        ),
        para("12. 60-Day Action Plan", s["h1"]),
        *bullet_list(
            [
                "Record a 2 to 3 minute demo video showing image input, utility heatmap, token selection, reconstruction, and metrics.",
                "Send supervisor package to 20 targeted PhD supervisors.",
                "Contact 5 to 8 incubators or space programmes with the founder package.",
                "Book patent attorney consultation using the private brief.",
                "Prepare a preprint or workshop submission using the current Sentinel-2 evidence.",
                "Collect written feedback and letters of interest for visa or founder evidence.",
            ],
            s["bullet"],
        ),
        para("This package is evidence framing and not immigration advice. It is intended to support conversations with endorsing bodies, incubators, supervisors, and legal professionals.", s["warning"]),
    ]
    build_pdf(path, "Founder and Visa Evidence Package", story)


def build_patent_package(path: Path) -> None:
    s = make_styles()
    story = cover(
        s,
        "Private Patent Attorney Brief",
        "Semantic Utility-Aware Token Transmission for Resource-Constrained Earth Observation Satellites",
        "Purpose: private technical material for a patent attorney to assess novelty, claim scope, prior-art risk, and filing strategy.",
        confidential=True,
    )
    story += [
        para("1. Confidentiality Notice", s["h1"]),
        para(
            "This document is private and should not be uploaded to GitHub, arXiv, websites, pitch decks, or supervisor emails before patent-attorney review. Use an NDA when sharing technical detail with commercial partners.",
            s["warning"],
        ),
        para("2. Technical Problem", s["h1"]),
        para(
            "Earth Observation satellites collect large image volumes but face downlink, latency, contact-window, and power constraints. Existing compression approaches usually optimise bitrate, distortion, or perceptual quality. For wildfire monitoring, the technical need is to preserve mission-relevant information under communication constraints.",
            s["body"],
        ),
        para("3. Proposed Technical Solution", s["h1"]),
        para(
            "The system encodes satellite imagery into learned visual tokens, estimates mission-specific utility over the image, maps utility to token-level importance, ranks tokens according to utility and communication constraints, transmits a selected token subset, and evaluates reconstruction with downstream mission utility metrics.",
            s["body"],
        ),
        fig(FIG / "02_working_pipeline.png", 15.8),
        PageBreak(),
        para("4. Core Invention Logic", s["h1"]),
        fig(FIG / "03_token_transmission_working_diagram.png", 15.0),
        para("Example embodiment", s["h2"]),
        *bullet_list(
            [
                "Receive satellite image or image tile.",
                "Generate mission utility map using wildfire, smoke, burn-scar, saliency, or detector-derived relevance.",
                "Encode image into learned visual tokens using a neural encoder.",
                "Map mission utility onto the token grid.",
                "Compute token priority using utility and optional entropy, cost, or reconstruction terms.",
                "Select token subset under bandwidth, latency, packet-loss, or downlink constraints.",
                "Transmit selected tokens and metadata.",
                "Reconstruct imagery or downstream mission product.",
                "Evaluate detector retention and semantic utility before and after reconstruction.",
            ],
            s["bullet"],
        ),
        para("5. Potential Claim Directions For Attorney Review", s["h1"]),
        make_table(
            [
                ["Direction", "Private drafting note"],
                ["Mission-utility token transmission", "Computer-implemented method for assigning mission utility to learned image tokens and transmitting a selected subset."],
                ["Onboard satellite processing", "Satellite payload or edge system configured to prioritise tokens before downlink."],
                ["Detector-retention preservation", "Compression process evaluated or controlled by downstream detector retention."],
                ["Adaptive downlink", "Token retention adjusted according to communication budget and mission priority."],
                ["Wildfire EO embodiment", "Wildfire, smoke, and burn-scar relevance as an example dependent embodiment."],
                ["Future detector-aware training", "Potential later filing if a utility-preserving training objective is developed and kept confidential."],
            ],
            [4.4, 11.4],
            header_color=LIGHT_ORANGE,
        ),
        PageBreak(),
        para("6. Current Technical Evidence", s["h1"]),
        make_table(SENTINEL_RESULTS, [3.2, 3.0, 4.0, 6.0]),
        Spacer(1, 0.2 * cm),
        fig(FIG / "08_model_decision_tradeoff.png", 11.2),
        para(
            "Patent-relevant technical insight: reconstruction quality and detector-facing mission utility can diverge. This supports a technical argument for mission-utility-aware compression rather than generic image compression.",
            s["callout"],
        ),
        para("7. Novelty Arguments To Explore", s["h1"]),
        *bullet_list(
            [
                "Mission-specific utility controls learned token transmission rather than only bitrate or visual distortion.",
                "Downstream detector retention is part of evaluation or control logic.",
                "The pipeline is positioned for onboard Earth Observation and satellite downlink constraints.",
                "Wildfire, smoke, and burn-scar relevance provide concrete mission embodiments.",
                "Token selection happens in learned representation space rather than only pixel-space region-of-interest coding.",
            ],
            s["bullet"],
        ),
        PageBreak(),
        para("8. Prior-Art Risk Areas", s["h1"]),
        make_table(
            [
                ["Prior-art area", "Why it matters"],
                ["JPEG2000 ROI coding", "May cover broad region-of-interest compression concepts."],
                ["CCSDS image compression", "Relevant to satellite image compression standards."],
                ["Task-aware image compression", "May overlap with downstream model performance objectives."],
                ["Semantic communication", "Relevant to transmitting task meaning instead of raw data."],
                ["Learned token pruning", "Relevant to token subset selection and transformer/VQ representations."],
                ["Onboard satellite AI", "Relevant to edge processing before downlink."],
            ],
            [4.3, 11.5],
            header_color=LIGHT_ORANGE,
        ),
        para("9. Questions For Patent Attorney", s["h1"]),
        *bullet_list(
            [
                "Is the invention best framed as a satellite communication process rather than software alone?",
                "Does existing public GitHub material limit broad claim scope?",
                "Should the independent claim focus on learned token mapping, detector retention, adaptive downlink, or the full pipeline?",
                "Should wildfire be in the independent claim or only in dependent claims and examples?",
                "Should detector-retention-aware training be protected in a later filing if developed further?",
                "Should the first filing route be UK, EPO, or PCT?",
                "Which elements should be patent claims and which should remain trade secrets?",
            ],
            s["bullet"],
        ),
        para("10. Disclosure Control", s["h1"]),
        para(
            "Public materials should stay at system and evidence level. Avoid publishing exact claim language, unreleased training objectives, full commercial implementation details, or future detector-retention-aware training designs before legal advice.",
            s["warning"],
        ),
    ]
    build_pdf(path, "Private Patent Attorney Brief", story, confidential=True)


def write_readme(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "# Application Packages",
                "",
                "This folder contains polished PDF packages for outreach and applications.",
                "",
                "## Public PDFs",
                "",
                "- `phd_supervisor_package.pdf`: supervisor-facing research package.",
                "- `founder_visa_package.pdf`: founder, incubator, Innovator Founder, and Global Talent evidence package.",
                "",
                "## Private PDF",
                "",
                "`private/patent_attorney_brief_private.pdf` is ignored by Git and must not be published before patent-attorney review.",
                "",
                "## Regenerate",
                "",
                "```powershell",
                "C:\\Users\\Athul Jayakumar\\.cache\\codex-runtimes\\codex-primary-runtime\\dependencies\\python\\python.exe scripts\\generate_priority1_packages.py",
                "```",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
