"""Stage 5 — Report Generator.

Generates a print-ready clinical triage & handoff PDF from a TriageResult.
Structured after real-world standards: ED triage note (IHE/MTS/ESI/ATS/CTAS),
SBAR/ISBAR handover, GP consultation summary, and discharge/urgent-care summaries.

Uses Jinja2 + WeasyPrint. Styling is the warm monochrome MedRoute palette.
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

URGENCY_LABEL = {
    "emergency": "Emergency — immediate assessment",
    "urgent": "Urgent — clinician review now",
    "soon": "Soon — within recommended window",
    "routine": "Routine — self-care or scheduled",
}

DISPOSITION_ACTION = {
    "emergency": "Call emergency services / send to ED now",
    "urgent": "Priority clinician review (do not wait)",
    "soon": "Book or attend within the recommended window",
    "routine": "Self-care with safety-netting, or routine appointment",
}


def _badge_class(level: ConfidenceLevel) -> str:
    if level == ConfidenceLevel.GREEN:
        return "badge-green"
    elif level == ConfidenceLevel.YELLOW:
        return "badge-yellow"
    return "badge-red"


def _load_template() -> Template:
    env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)))
    return env.get_template("report.html")


def _resolve_template(template) -> Template:
    if template is None:
        return _load_template()
    if isinstance(template, str):
        return Template(template)
    return template


def generate_html(
    result: TriageResult,
    *,
    template=None,
    now=None,
    transcript: str = "",
    case_id: str = "",
    generated_at: str = "",
) -> str:
    """Render the HTML report from the Jinja2 template.

    ``template`` may be a Jinja2 ``Template``, a template source string, or None
    (loads the packaged ``templates/report.html``). ``now`` overrides the
    timestamp so output is deterministic in tests; defaults to ``utcnow``.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    urgency = (result.urgency or "routine").lower()
    red_flag = result.red_flag
    is_red_flag = bool(red_flag and red_flag.triggered) or result.route == (
        TriageRoute.HARD_ESCALATION
    )
    ctx = {
        "result": result,
        "badge_class": _badge_class(result.confidence_level),
        "is_red_flag": is_red_flag,
        "urgency_label": URGENCY_LABEL.get(urgency, urgency.title()),
        "disposition_action": DISPOSITION_ACTION.get(urgency, ""),
        "red_flag_message": (red_flag.message if red_flag else "") or "",
        "red_flag_matched": (red_flag.matched_symptoms if red_flag else []) or [],
        "patient": result.patient,
        "transcript": transcript or "",
        "case_id": case_id or "",
        "generated_at": generated_at or "",
        "report_date": now.strftime("%d %B %Y · %H:%M UTC"),
    }
    return _resolve_template(template).render(**ctx)


def generate_pdf(
    result: TriageResult,
    *,
    template=None,
    now=None,
    transcript: str = "",
    case_id: str = "",
    generated_at: str = "",
) -> bytes:
    """Generate a PDF bytes from a TriageResult."""
    html = generate_html(
        result,
        template=template,
        now=now,
        transcript=transcript,
        case_id=case_id,
        generated_at=generated_at,
    )
    try:
        from weasyprint import HTML

        return HTML(string=html).write_pdf()
    except Exception as exc:  # noqa: BLE001
        log.warning("WeasyPrint failed: %s. Returning HTML bytes.", exc)
        return html.encode("utf-8")
