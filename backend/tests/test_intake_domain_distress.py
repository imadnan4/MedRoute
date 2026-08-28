"""Offline tests for the intake domain variant + distress/safeguarding layer.

Loads synthetic fixtures and runs the intake path with a fake ``InferenceAdapter``
(LLM output is irrelevant to routing — escalation is deterministic). Asserts the
``disposition`` matches the fixture, that distress/red-flag escalation sets
``needs_human_review`` and populates ``distress``, and that legal-domain transcripts
are routed through the legal prompt path. No network.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pytest

from models import Disposition, DistressResult, Encounter
from pipeline.triage_pipeline import TriagePipeline
from safety.red_flag_checker import check_red_flags
from safety.distress_checker import check_distress
from voice.transcriber import Transcript

FIXTURES = json.loads(
    (Path(__file__).parent / "fixtures" / "intakes.json").read_text()
)

CANNED_INTAKE = {
    "contact_name": "Aisha",
    "care_recipient_name": "Mrs. Khan",
    "phone_or_contact": "03001234567",
    "care_need_summary": "needs wound dressing at home twice a week",
    "condition_or_issue": "leg ulcer",
    "mobility_or_severity_notes": "limited mobility, uses walker",
    "insurance_or_payment_notes": "private insurance",
    "preferred_availability": "weekday mornings",
    "free_text_summary": "Daughter calling about her mother who needs home wound care.",
}


class FakeInference:
    def __init__(self, canned: dict) -> None:
        self.canned = canned
        self.last_payload: Optional[dict] = None

    def infer(self, payload: dict) -> dict:
        self.last_payload = payload
        return dict(self.canned)


@dataclass
class FakeRequest:
    transcript: str = "needs wound care"
    language: str = "en"
    audio_b64: Optional[str] = None
    age_years: Optional[float] = None
    age_months: Optional[float] = None
    pregnancy: Optional[str] = None


class FakeTranscriber:
    def transcribe_bytes(self, audio_bytes: bytes) -> Transcript:
        return Transcript(text="audio transcript", language="en", latency_ms=12)

    def transcribe_text(self, text: str, language: str = "en") -> Transcript:
        return Transcript(text=text, language=language, latency_ms=0)


def _never_called(*_args: Any) -> Any:
    raise AssertionError("intake mode must not invoke the triage extraction stages")


def _build(inference: FakeInference) -> TriagePipeline:
    return TriagePipeline(
        transcriber=FakeTranscriber(),
        parse_input=_never_called,
        check_red_flags=check_red_flags,
        check_distress=check_distress,
        score_complexity=_never_called,
        run_triage=_never_called,
        inference=inference,
    )


@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda f: f["transcript"][:24])
def test_intake_fixture_routing(fixture: dict) -> None:
    inference = FakeInference(CANNED_INTAKE)
    pipeline = _build(inference)
    request = FakeRequest(transcript=fixture["transcript"], language=fixture["language"])

    encounter = pipeline.run(
        request, "case-fixture", mode="intake", domain=fixture["domain"]
    )

    assert isinstance(encounter, Encounter)
    assert encounter.disposition.value == fixture["expect_disposition"]
    assert encounter.domain == fixture["domain"]

    # distress result is always populated with the parallel shape
    assert isinstance(encounter.distress, DistressResult)

    if fixture["expect_disposition"] == Disposition.ESCALATE_TO_CLINICIAN.value:
        assert encounter.needs_human_review is True
        assert encounter.red_flag.triggered or encounter.distress.triggered
    else:
        assert encounter.needs_human_review is False


def test_legal_domain_routes_through_legal_prompt() -> None:
    inference = FakeInference(CANNED_INTAKE)
    pipeline = _build(inference)
    request = FakeRequest(
        transcript="I need help with an eviction and custody matter",
        language="en",
    )

    pipeline.run(request, "case-legal", mode="intake", domain="legal")

    context = inference.last_payload["context"]
    assert "legal" in context.lower()
    assert "matter" in context.lower()


def test_home_health_domain_routes_through_home_health_prompt() -> None:
    inference = FakeInference(CANNED_INTAKE)
    pipeline = _build(inference)
    request = FakeRequest(transcript="my mother needs wound care", language="en")

    pipeline.run(request, "case-hh", mode="intake", domain="home_health")

    context = inference.last_payload["context"]
    assert "home-health" in context.lower()


def test_distress_only_triggers_escalation_without_medical_red_flag() -> None:
    inference = FakeInference(CANNED_INTAKE)
    pipeline = _build(inference)
    request = FakeRequest(transcript="I can't cope, I'm at my wits end with all of this")

    encounter = pipeline.run(request, "case-distress", mode="intake")

    assert encounter.distress.triggered is True
    assert encounter.red_flag.triggered is False
    assert encounter.disposition == Disposition.ESCALATE_TO_CLINICIAN
    assert encounter.needs_human_review is True


def test_clean_transcript_sets_standard_queue_and_no_distress() -> None:
    inference = FakeInference(CANNED_INTAKE)
    pipeline = _build(inference)
    request = FakeRequest(transcript="I would like to arrange physiotherapy for my father")

    encounter = pipeline.run(request, "case-clean", mode="intake")

    assert encounter.red_flag.triggered is False
    assert encounter.distress.triggered is False
    assert encounter.disposition == Disposition.STANDARD_QUEUE
    assert encounter.needs_human_review is False


def test_extract_intake_returns_same_schema_for_legal() -> None:
    from pipeline.extraction.intake_extract import extract_intake

    inference = FakeInference(CANNED_INTAKE)
    out = extract_intake("I need help with an eviction", inference, domain="legal")

    assert set(out) == {
        "contact_name",
        "care_recipient_name",
        "phone_or_contact",
        "care_need_summary",
        "condition_or_issue",
        "mobility_or_severity_notes",
        "insurance_or_payment_notes",
        "preferred_availability",
        "free_text_summary",
    }
