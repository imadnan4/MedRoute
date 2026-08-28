"""Stage 5 — Report Generator.

Generates PDF triage reports with confidence badge (green/yellow/red), patient
context, diagnosis, and recommendations. Uses Jinja2 + WeasyPrint.

Styling matches the frontend: warm monochrome palette, Newsreader/Geist fonts,
1px borders, pastel badges.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, Template
from models import ConfidenceLevel, TriageResult, TriageRoute

log = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
TEMPLATE_DIR = HERE / "templates"


def _badge_class(level: ConfidenceLevel) -> str:
    if level == ConfidenceLevel.GREEN:
        return "badge-green"
    elif level == ConfidenceLevel.YELLOW:
        return "badge-yellow"
    return "badge-red"


def _load_template() -> Template:
    """Load the packaged report template lazily (no import-time side effects)."""
    env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)))
    return env.get_template("report.html")


def _resolve_template(template) -> Template:
    if template is None:
        return _load_template()
    if isinstance(template, str):
        return Template(template)
    return template


def generate_html(result: TriageResult, *, template=None, now=None) -> str:
    """Render the HTML report from the Jinja2 template.

    ``template`` may be a Jinja2 ``Template``, a template source string, or None
    (loads the packaged ``templates/report.html``). ``now`` overrides the
    timestamp so output is deterministic in tests; defaults to ``utcnow``.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    tpl = _resolve_template(template)
    return tpl.render(
        result=result,
        badge_class=_badge_class(result.confidence_level),
        is_red_flag=result.route == TriageRoute.HARD_ESCALATION,
        patient=result.patient,
        report_date=now.strftime("%d %B %Y · %H:%M UTC"),
    )


def generate_pdf(result: TriageResult, *, template=None, now=None) -> bytes:
    """Generate a PDF bytes from a TriageResult."""
    html = generate_html(result, template=template, now=now)
    try:
        from weasyprint import HTML

        pdf_bytes = HTML(string=html).write_pdf()
        return pdf_bytes
    except Exception as exc:
        log.warning("WeasyPrint failed: %s. Returning HTML bytes.", exc)
        return html.encode("utf-8")
