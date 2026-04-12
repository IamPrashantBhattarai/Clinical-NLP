"""
Build a single self-contained PDF of the Clinical NLP project documentation.

Output: docs/ClinicalNLP-Documentation.pdf

Renders the markdown files under docs/ into HTML, wraps them in a branded
cover page + table of contents, and converts to PDF via xhtml2pdf.
"""

import re
from datetime import date
from pathlib import Path

import markdown
from pypdf import PdfReader, PdfWriter
from xhtml2pdf import pisa

ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs"
OUT_PDF = DOCS_DIR / "ClinicalNLP-Documentation.pdf"

AUTHOR = "Prashant Bhattarai"
PROJECT = "Clinical NLP"
SUBTITLE = "Predicting 30-Day Hospital Readmission from Discharge Summaries"
VERSION = "1.0"
BUILD_DATE = date.today().strftime("%B %d, %Y")
REPO_URL = "github.com/IamPrashantBhattarai/Clinical-NLP"

SECTIONS = [
    ("01-overview.md",              "1. Project Overview"),
    ("02-setup.md",                 "2. Setup & Running"),
    ("03-pipeline.md",              "3. Pipeline Reference"),
    ("04-models-and-evaluation.md", "4. Models & Evaluation"),
    ("05-dashboard.md",             "5. Dashboard"),
    ("06-configuration.md",         "6. Configuration Reference"),
    ("07-testing.md",               "7. Testing & Troubleshooting"),
]


# ---------------------------------------------------------------------------
# Markdown preprocessing
# ---------------------------------------------------------------------------

def strip_front_heading(md: str) -> str:
    """Drop the first H1 in each file (we render our own section titles)."""
    lines = md.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("# "):
            return "\n".join(lines[:i] + lines[i + 1:]).lstrip()
    return md


def neutralize_links(md: str) -> str:
    """
    PDF is read out of context, so turn relative doc/source links into plain
    monospace text. Keep external http(s) links as plain URLs in parens.
    """
    def repl(match):
        text, target = match.group(1), match.group(2)
        if target.startswith(("http://", "https://")):
            return f"{text} ({target})"
        return f"<code>{text}</code>"

    return re.sub(r"\[([^\]]+)\]\(([^)]+)\)", repl, md)


def load_section(filename: str) -> str:
    path = DOCS_DIR / filename
    md = path.read_text(encoding="utf-8")
    md = strip_front_heading(md)
    md = neutralize_links(md)
    return md


# ---------------------------------------------------------------------------
# HTML assembly
# ---------------------------------------------------------------------------

CSS = r"""
@page {
    size: A4;
    margin: 2.2cm 2cm 2.4cm 2cm;
    @frame footer {
        -pdf-frame-content: footerContent;
        left: 2cm; right: 2cm;
        bottom: 1cm; height: 1cm;
    }
}
@page cover {
    size: A4;
    margin: 0cm;
}
body {
    font-family: Helvetica, Arial, sans-serif;
    font-size: 10.5pt;
    line-height: 1.45;
    color: #1f2937;
}
h1, h2, h3, h4 {
    color: #0b3b66;
    font-weight: 700;
    margin-top: 18pt;
    margin-bottom: 8pt;
    page-break-after: avoid;
}
h1 { font-size: 22pt; border-bottom: 2pt solid #0b3b66; padding-bottom: 6pt; }
h2 { font-size: 16pt; }
h3 { font-size: 13pt; color: #114e89; }
h4 { font-size: 11.5pt; color: #114e89; }
p  { margin: 6pt 0; text-align: justify; }
ul, ol { margin: 6pt 0 6pt 18pt; }
li { margin: 2pt 0; }
code {
    font-family: "Courier New", monospace;
    font-size: 9.5pt;
    background-color: #f1f5f9;
    color: #0b3b66;
    padding: 1pt 3pt;
    border-radius: 2pt;
}
pre {
    background-color: #f8fafc;
    border-left: 3pt solid #0b3b66;
    padding: 8pt 10pt;
    font-family: "Courier New", monospace;
    font-size: 9pt;
    line-height: 1.35;
    color: #1e293b;
    page-break-inside: avoid;
    white-space: pre-wrap;
}
table {
    width: 100%;
    border-collapse: collapse;
    margin: 10pt 0;
    font-size: 9.5pt;
}
th {
    background-color: #0b3b66;
    color: #ffffff;
    padding: 6pt 8pt;
    text-align: left;
    font-weight: 600;
}
td {
    padding: 5pt 8pt;
    border-bottom: 0.5pt solid #cbd5e1;
    vertical-align: top;
}
tr:nth-child(even) td { background-color: #f8fafc; }
blockquote {
    border-left: 3pt solid #94a3b8;
    margin: 8pt 0;
    padding: 4pt 12pt;
    color: #475569;
    font-style: italic;
}
hr { border: 0; border-top: 0.5pt solid #cbd5e1; margin: 12pt 0; }

.cover {
    -pdf-frame-content: coverContent;
    background-color: #0b3b66;
}
.cover-wrap {
    margin-top: 6.5cm;
    padding: 0 2.2cm;
    color: #ffffff;
}
.cover-eyebrow {
    font-size: 11pt;
    letter-spacing: 3pt;
    color: #93c5fd;
    text-transform: uppercase;
    margin-bottom: 10pt;
}
.cover-title {
    font-size: 40pt;
    font-weight: 700;
    line-height: 1.1;
    margin-bottom: 14pt;
    color: #ffffff;
}
.cover-subtitle {
    font-size: 15pt;
    color: #e0e7ff;
    margin-bottom: 2cm;
    line-height: 1.35;
}
.cover-rule {
    border-top: 1.5pt solid #60a5fa;
    width: 5cm;
    margin: 0 0 0.8cm 0;
}
.cover-meta {
    font-size: 11pt;
    color: #dbeafe;
    line-height: 1.7;
}
.cover-author {
    font-size: 14pt;
    color: #ffffff;
    font-weight: 600;
}
.cover-footer {
    position: absolute;
    bottom: 2cm;
    left: 2.2cm;
    right: 2.2cm;
    font-size: 9pt;
    color: #93c5fd;
    border-top: 0.5pt solid #1d4ed8;
    padding-top: 8pt;
}

.toc-title {
    font-size: 22pt;
    color: #0b3b66;
    border-bottom: 2pt solid #0b3b66;
    padding-bottom: 6pt;
    margin-bottom: 18pt;
}
.toc-item {
    font-size: 11.5pt;
    margin: 8pt 0;
    color: #0b3b66;
}
.toc-dot {
    color: #94a3b8;
    margin: 0 4pt;
}

.section-header {
    page-break-before: always;
    margin-top: 0;
}
.section-label {
    color: #2563eb;
    font-size: 10pt;
    letter-spacing: 2pt;
    text-transform: uppercase;
    margin-bottom: 4pt;
}

.footer {
    font-size: 8.5pt;
    color: #64748b;
    border-top: 0.5pt solid #cbd5e1;
    padding-top: 4pt;
    text-align: center;
}
"""


def build_cover() -> str:
    return f"""
<div id="coverContent" class="cover">
  <div class="cover-wrap">
    <div class="cover-eyebrow">Technical Documentation</div>
    <div class="cover-title">{PROJECT}</div>
    <div class="cover-subtitle">{SUBTITLE}</div>
    <div class="cover-rule"></div>
    <div class="cover-meta">
      <div class="cover-author">{AUTHOR}</div>
      Version {VERSION} &nbsp;&middot;&nbsp; {BUILD_DATE}<br/>
      {REPO_URL}
    </div>
  </div>
  <div class="cover-footer">
    End-to-end clinical NLP pipeline &mdash; preprocessing, topic modeling,
    clinical embeddings, prediction, SHAP explainability, and fairness auditing.
  </div>
</div>
"""


def build_toc() -> str:
    items = "\n".join(
        f'<div class="toc-item">{title} '
        f'<span class="toc-dot">&middot;&middot;&middot;&middot;&middot;&middot;&middot;&middot;&middot;&middot;&middot;&middot;&middot;&middot;&middot;&middot;&middot;&middot;&middot;&middot;&middot;&middot;&middot;&middot;&middot;&middot;&middot;&middot;&middot;&middot;&middot;&middot;&middot;&middot;&middot;&middot;&middot;&middot;&middot;&middot;&middot;&middot;</span> '
        f'page {i + 3}</div>'
        for i, (_, title) in enumerate(SECTIONS)
    )
    return f"""
<div style="page-break-before: always;">
  <div class="toc-title">Table of Contents</div>
  {items}
  <div style="margin-top: 1.2cm; font-size: 10pt; color: #475569;">
    <em>Prepared by {AUTHOR}. This document describes the full Clinical NLP
    pipeline for 30-day hospital readmission prediction, from data ingestion
    through modeling, explainability, fairness auditing, and the companion
    FastAPI dashboard.</em>
  </div>
</div>
"""


def build_section(md: str, title: str) -> str:
    html_body = markdown.markdown(
        md,
        extensions=["tables", "fenced_code", "sane_lists", "attr_list"],
    )
    label, name = title.split(". ", 1)
    return f"""
<div class="section-header">
  <div class="section-label">Section {label}</div>
  <h1>{name}</h1>
  {html_body}
</div>
"""


def build_html() -> str:
    cover = build_cover()
    toc = build_toc()
    body = "\n".join(build_section(load_section(fn), title) for fn, title in SECTIONS)

    footer = f"""
<div id="footerContent" class="footer">
  {PROJECT} &nbsp;&middot;&nbsp; {AUTHOR} &nbsp;&middot;&nbsp; v{VERSION}
  &nbsp;&middot;&nbsp; Page <pdf:pagenumber/> of <pdf:pagecount/>
</div>
"""

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<title>{PROJECT} — Documentation</title>
<style>{CSS}</style>
</head>
<body>
{cover}
{toc}
{body}
{footer}
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def stamp_metadata(pdf_path: Path) -> None:
    reader = PdfReader(str(pdf_path))
    writer = PdfWriter(clone_from=reader)
    writer.add_metadata({
        "/Title":    f"{PROJECT} - {SUBTITLE}",
        "/Author":   AUTHOR,
        "/Subject":  "Clinical NLP pipeline for 30-day hospital readmission prediction",
        "/Keywords": "clinical NLP, MIMIC-IV, readmission, BERT, LDA, BERTopic, SHAP, fairness, Fairlearn",
        "/Creator":  f"{AUTHOR} - Clinical NLP Project",
        "/Producer": f"{AUTHOR} - Clinical NLP Project",
    })
    with open(pdf_path, "wb") as f:
        writer.write(f)


def main():
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    html = build_html()

    with open(OUT_PDF, "wb") as f:
        result = pisa.CreatePDF(src=html, dest=f, encoding="utf-8")

    if result.err:
        raise SystemExit(f"xhtml2pdf reported {result.err} error(s) while rendering.")

    stamp_metadata(OUT_PDF)

    size_kb = OUT_PDF.stat().st_size / 1024
    print(f"Wrote {OUT_PDF.relative_to(ROOT)} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
