"""Tests for the lighter Intake mode of the shared TriagePipeline.

Offline: a fake InferenceAdapter returns a canned IntakeRecord dict; the REAL
shared Red-Flag checker is exercised so escalation is driven by transcript
content. Asserts the ``Encounter`` shape, disposition mapping, and that the
intake path never touches the triage agent / scorer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import pytest

from models import Disposition, Encounter, TriageScore
from pipeline.triage_pipeline import TriagePipeline
from safety.red_flag_checker import check_red_flags
from voice.transcriber import Transcript

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
        score_complexity=_never_called,
        run_triage=_never_called,
        inference=inference,
    )


def test_intake_returns_encounter_with_standard_queue_when_no_red_flag() -> None:
    pipeline = _build(FakeInference(CANNED_INTAKE))
    request = FakeRequest(
        transcript="My mother needs wound care at home, she has a leg ulcer."
    )

    encounter = pipeline.run(request, "case-intake-1", mode="intake")

    assert isinstance(encounter, Encounter)
    assert encounter.mode == "intake"
    assert encounter.disposition == Disposition.STANDARD_QUEUE
    assert encounter.needs_human_review is False
    assert encounter.acuity is None
    assert encounter.red_flag.triggered is False
    assert encounter.extracted["care_recipient_name"] == "Mrs. Khan"
    assert set(encounter.extracted) == {
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


def test_intake_escalates_to_clinician_when_red_flag_present() -> None:
    pipeline = _build(FakeInference(CANNED_INTAKE))
    request = FakeRequest(transcript="He has chest pain and shortness of breath")

    encounter = pipeline.run(request, "case-intake-2", mode="intake")

    assert isinstance(encounter, Encounter)
    assert encounter.red_flag.triggered is True
    assert encounter.disposition == Disposition.ESCALATE_TO_CLINICIAN
    assert encounter.needs_human_review is True
    assert encounter.input_mode == "text"


def test_intake_audio_path_sets_input_mode_and_runs_red_flag() -> None:
    pipeline = _build(FakeInference(CANNED_INTAKE))
    request = FakeRequest(
        audio_b64="aGVsbG8=", transcript="spoken intake", language="hi-IN"
    )

    encounter = pipeline.run(request, "case-intake-3", mode="intake")

    assert encounter.input_mode == "audio"
    assert encounter.input_language == "en"
    assert encounter.disposition == Disposition.STANDARD_QUEUE


def test_intake_uses_on_stage_for_progress() -> None:
    pipeline = _build(FakeInference(CANNED_INTAKE))
    seen: list[tuple[str, str]] = []

    pipeline.run(
        FakeRequest(transcript="routine home care request"),
        "case-intake-4",
        mode="intake",
        on_stage=lambda n, s, p: seen.append((n, s)),
    )

    names = [n for n, s in seen if s == "completed"]
    assert names == ["asr", "safety", "intake", "done"]
