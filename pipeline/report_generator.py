"""Stage 5 — Report Generator.

Generates PDF triage reports with confidence badge (green/yellow/red), patient
context, diagnosis, and recommendations. Uses Jinja2 + WeasyPrint.

Styling matches the frontend: warm monochrome palette, Newsreader/Geist fonts,
1px borders, pastel badges.
"""
from __future__ import annotations

import logging
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from models import ConfidenceLevel, TriageResult, TriageRoute

log = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
TEMPLATE_DIR = HERE / "templates"

_env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)))
_template = _env.get_template("report.html")


def _badge_class(level: ConfidenceLevel) -> str:
    if level == ConfidenceLevel.GREEN:
        return "badge-green"
    elif level == ConfidenceLevel.YELLOW:
        return "badge-yellow"
    return "badge-red"


def generate_html(result: TriageResult) -> str:
    """Render the HTML report from the Jinja2 template."""
    return _template.render(
        result=result,
        badge_class=_badge_class(result.confidence_level),
        is_red_flag=result.route == TriageRoute.HARD_ESCALATION,
        patient=result.patient,
    )


def generate_pdf(result: TriageResult) -> bytes:
    """Generate a PDF bytes from a TriageResult."""
    html = generate_html(result)
    try:
        from weasyprint import HTML
        pdf_bytes = HTML(string=html).write_pdf()
        return pdf_bytes
    except Exception as exc:
        log.warning("WeasyPrint failed: %s. Returning HTML bytes.", exc)
        return html.encode("utf-8")
