"""Tests for the report generator deep module.

Asserts the template is injectable and the timestamp is injectable, so output
is deterministic and testable (no network, no import-time file dependency).
"""

from __future__ import annotations

from datetime import datetime, timezone

from models import (
    ConfidenceLevel,
    PatientContext,
    TriageResult,
    TriageRoute,
    UrgencyLevel,
)
from pipeline.report_generator import generate_html, generate_pdf

FIXED_NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def _sample_result() -> TriageResult:
    return TriageResult(
        route=TriageRoute.LOCAL_ONLY,
        confidence=0.9,
        confidence_level=ConfidenceLevel.GREEN,
        likely_condition="Viral upper respiratory infection",
        recommendation="Supportive care; return if dyspnea.",
        watch_for=["shortness of breath"],
        urgency=UrgencyLevel.ROUTINE,
        patient=PatientContext(age_years=30),
    )


def test_html_deterministic_with_fixed_now() -> None:
    r = _sample_result()
    a = generate_html(r, now=FIXED_NOW)
    b = generate_html(r, now=FIXED_NOW)
    assert a == b
    assert "1 January 2026" in a


def test_pdf_deterministic_with_fixed_now() -> None:
    r = _sample_result()
    a = generate_pdf(r, now=FIXED_NOW)
    b = generate_pdf(r, now=FIXED_NOW)
    assert a == b


def test_custom_template_string_is_used() -> None:
    r = _sample_result()
    tpl = "<html>MESSAGE-{{ result.likely_condition }}-END</html>"
    out = generate_html(r, template=tpl)
    assert "MESSAGE-Viral upper respiratory infection-END" in out


def test_default_loads_packaged_template() -> None:
    r = _sample_result()
    out = generate_html(r)
    # The packaged template renders the confidence badge markup.
    assert "confidence-box" in out
    assert "Viral upper respiratory infection" in out
