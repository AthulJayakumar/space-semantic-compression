"""Render the current application proposal without changing research evidence."""

from __future__ import annotations

import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    KeepTogether,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports" / "phd_proposal_submission_2026.md"
OUTPUT = ROOT / "reports" / "phd_proposal_submission_2026.pdf"
NAVY = colors.HexColor("#123251")
TEAL = colors.HexColor("#237D83")
INK = colors.HexColor("#1B2732")
MUTED = colors.HexColor("#52616C")
RULE = colors.HexColor("#CCD5DA")


def inline_markup(value: str) -> str:
    """Translate the small Markdown subset used by the proposal to Platypus."""
    value = html.escape(value)
    value = re.sub(r"\[([^]]+)\]\((https?://[^)]+)\)", r'<link href="\2" color="#237D83">\1</link>', value)
    value = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", value)
    value = re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', value)
    value = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", value)
    return value


def make_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("ProposalTitle", parent=base["Title"], fontName="Helvetica-Bold", fontSize=18, leading=21, textColor=NAVY, spaceAfter=10, alignment=TA_LEFT),
        "meta": ParagraphStyle("ProposalMeta", parent=base["Normal"], fontName="Helvetica", fontSize=9, leading=13, textColor=MUTED, spaceAfter=4),
        "h2": ParagraphStyle("ProposalH2", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=12, leading=15, textColor=NAVY, spaceBefore=13, spaceAfter=6, keepWithNext=True),
        "body": ParagraphStyle("ProposalBody", parent=base["Normal"], fontName="Helvetica", fontSize=9.2, leading=13.8, textColor=INK, spaceAfter=8),
        "small": ParagraphStyle("ProposalSmall", parent=base["Normal"], fontName="Helvetica", fontSize=7.6, leading=10.5, textColor=INK, alignment=TA_LEFT),
        "table": ParagraphStyle("ProposalTable", parent=base["Normal"], fontName="Helvetica", fontSize=7.1, leading=9.7, textColor=INK),
        "head": ParagraphStyle("ProposalTableHead", parent=base["Normal"], fontName="Helvetica-Bold", fontSize=7.1, leading=9.7, textColor=colors.white),
        "refs": ParagraphStyle("ProposalRefs", parent=base["Normal"], fontName="Helvetica", fontSize=8.0, leading=11.3, textColor=INK, spaceAfter=5, wordWrap="CJK"),
    }


def parse_table(lines: list[str], styles: dict[str, ParagraphStyle]) -> Table:
    rows = [[cell.strip() for cell in line.strip().strip("|").split("|")] for line in lines]
    rows = [rows[0], *rows[2:]]
    ncols = len(rows[0])
    width = A4[0] - 54 * mm
    if ncols == 3:
        weights = [0.21, 0.41, 0.38] if len(rows) > 4 else [0.12, 0.53, 0.35]
    elif ncols == 8:
        weights = [0.20, 0.105, 0.105, 0.09, 0.075, 0.12, 0.17, 0.135]
    else:
        weights = [1 / ncols] * ncols
    data = []
    for row_no, row in enumerate(rows):
        style = styles["head"] if row_no == 0 else styles["table"]
        data.append([Paragraph(inline_markup(item), style) for item in row])
    table = Table(data, colWidths=[width * w for w in weights], repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F6F7")]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("LINEBELOW", (0, -1), (-1, -1), 0.5, RULE),
    ]))
    return table


def page_decoration(canvas, doc) -> None:
    canvas.saveState()
    w, h = A4
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.5)
    canvas.line(27 * mm, h - 17 * mm, w - 27 * mm, h - 17 * mm)
    canvas.setFont("Helvetica", 7.6)
    canvas.setFillColor(MUTED)
    canvas.drawString(27 * mm, h - 14 * mm, "ATHUL JAYAKUMAR  /  PHD RESEARCH PROPOSAL")
    canvas.line(27 * mm, 17 * mm, w - 27 * mm, 17 * mm)
    canvas.drawString(27 * mm, 13 * mm, "September 2026  |  Preliminary evidence; hypothesis unconfirmed")
    canvas.drawRightString(w - 27 * mm, 13 * mm, str(doc.page))
    canvas.restoreState()


def build() -> Path:
    styles = make_styles()
    story = []
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    paragraph = []

    def flush() -> None:
        if paragraph:
            content = " ".join(part.strip().rstrip("  ") for part in paragraph)
            style = styles["refs"] if content.startswith("[") and re.match(r"^\[\d+\]", content) else styles["body"]
            story.append(Paragraph(inline_markup(content), style))
            paragraph.clear()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            flush()
        elif line.startswith("# "):
            flush()
            story.append(Paragraph(inline_markup(line[2:]), styles["title"]))
        elif line.startswith("## "):
            flush()
            story.append(Paragraph(inline_markup(line[3:]), styles["h2"]))
        elif line.startswith("**") and i < 5:
            flush()
            story.append(Paragraph(inline_markup(line.replace("  ", "")), styles["meta"]))
        elif line.startswith("|"):
            flush()
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            story.extend([Spacer(1, 3 * mm), parse_table(table_lines, styles), Spacer(1, 3 * mm)])
            continue
        elif line.startswith("---"):
            flush()
            story.append(HRFlowable(width="100%", thickness=0.7, color=TEAL, spaceBefore=4, spaceAfter=8))
        else:
            paragraph.append(line)
        i += 1
    flush()
    doc = BaseDocTemplate(str(OUTPUT), pagesize=A4, leftMargin=27 * mm, rightMargin=27 * mm, topMargin=22 * mm, bottomMargin=23 * mm, title="Semantic Utility-Aware Communication for Wildfire-Centric Earth Observation", author="Athul Jayakumar")
    doc.addPageTemplates(PageTemplate(id="Proposal", frames=[Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)], onPage=page_decoration))
    doc.build(story)
    return OUTPUT


if __name__ == "__main__":
    print(build())
