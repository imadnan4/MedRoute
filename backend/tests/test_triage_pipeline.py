"""Tests for the deep TriagePipeline module.

All stage dependencies are faked — no network, no real ASR / LLM / RAG. We assert
the stage ORDER, the shape of the returned outcome, and that ``on_stage`` fires
for every stage (SSE mode).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import pytest

from models import (
    ConfidenceLevel,
    ParsedInput,
    PatientContext,
    PregnancyStatus,
    RedFlagResult,
    TriageResult,
    TriageRoute,
    TriageScore,
    UrgencyLevel,
)
from pipeline.triage_pipeline import PipelineOutcome, TriagePipeline
from voice.transcriber import Transcript


@dataclass
class FakeRequest:
    transcript: str = "fever and cough"
    language: str = "hi-IN"
    audio_b64: Optional[str] = None
    age_years: Optional[float] = None
    age_months: Optional[float] = None
    pregnancy: Optional[str] = None


class FakeTranscriber:
    def __init__(self) -> None:
        self.bytes_calls: list[bytes] = []
        self.text_calls: list[tuple[str, str]] = []

    def transcribe_bytes(self, audio_bytes: bytes) -> Transcript:
        self.bytes_calls.append(audio_bytes)
        return Transcript(text="audio transcript", language="hi-IN", latency_ms=12)

    def transcribe_text(self, text: str, language: str = "hi-IN") -> Transcript:
        self.text_calls.append((text, language))
        return Transcript(text=text, language=language, latency_ms=0)


def _fake_parsed() -> ParsedInput:
    return ParsedInput(
        transcript="fever and cough",
        language="hi-IN",
        symptoms=["fever", "cough"],
        patient=PatientContext(age_years=30),
        symptom_clusters=["viral_uri"],
    )


def _fake_score() -> TriageScore:
    return TriageScore(
        raw_score=2,
        adjusted_score=2,
        confidence=0.9,
        route=TriageRoute.LOCAL_ONLY,
        syndrome_hits=["viral_uri"],
        reasoning="raw_score=2 adjusted_score=2 route=local_only",
    )


def _fake_result(route: TriageRoute = TriageRoute.LOCAL_ONLY) -> TriageResult:
    return TriageResult(
        route=route,
        confidence=0.9,
        confidence_level=ConfidenceLevel.GREEN,
        likely_condition="Viral URI",
        urgency=UrgencyLevel.ROUTINE,
        cascade_used=["openrouter"],
    )


@dataclass
class Fakes:
    transcriber: FakeTranscriber
    order: list[str]
    captured_score: list[Any]
    captured_red_flag: list[Any]

    def parse_input(self, transcript: Transcript) -> ParsedInput:
        self.order.append("parser")
        return _fake_parsed()

    def check_red_flags(self, parsed: ParsedInput) -> RedFlagResult:
        self.order.append("safety")
        rf = RedFlagResult(triggered=False)
        self.captured_red_flag.append(rf)
        return rf

    def score_complexity(self, parsed: ParsedInput) -> TriageScore:
        self.order.append("scorer")
        return _fake_score()

    def run_triage(self, parsed: ParsedInput, score: TriageScore, red_flag: RedFlagResult) -> TriageResult:
        self.order.append("agent")
        self.captured_score.append(score)
        if red_flag.triggered:
            return _fake_result(TriageRoute.HARD_ESCALATION)
        return _fake_result()


def _build(fakes: Fakes) -> TriagePipeline:
    return TriagePipeline(
        transcriber=fakes.transcriber,
        parse_input=fakes.parse_input,
        check_red_flags=fakes.check_red_flags,
        score_complexity=fakes.score_complexity,
        run_triage=fakes.run_triage,
    )


def test_stage_order_is_asr_then_parser_safety_scorer_agent() -> None:
    fakes = Fakes(FakeTranscriber(), [], [], [])
    pipeline = _build(fakes)

    pipeline.run(FakeRequest(), "case-1")

    assert fakes.order == ["parser", "safety", "scorer", "agent"]
    assert fakes.transcriber.text_calls == [("fever and cough", "hi-IN")]


def test_outcome_contains_expected_fields() -> None:
    fakes = Fakes(FakeTranscriber(), [], [], [])
    pipeline = _build(fakes)

    outcome = pipeline.run(FakeRequest(), "case-7")

    assert isinstance(outcome, PipelineOutcome)
    assert outcome.case_id == "case-7"
    assert set(outcome.stages) == {"asr", "parser", "safety", "scorer", "agent"}
    assert outcome.stages["asr"]["text"] == "fever and cough"
    assert outcome.stages["parser"]["symptoms"] == ["fever", "cough"]
    assert outcome.result.route == TriageRoute.LOCAL_ONLY
    assert outcome.request["transcript"] == "fever and cough"


def test_on_stage_fires_for_every_stage_in_sse_mode() -> None:
    fakes = Fakes(FakeTranscriber(), [], [], [])
    pipeline = _build(fakes)

    seen: list[tuple[str, str]] = []
    pipeline.run(FakeRequest(), "case-sse", on_stage=lambda n, s, p: seen.append((n, s)))

    names = [n for n, s in seen if s == "completed"]
    assert names == ["asr", "parser", "safety", "scorer", "agent", "done"]


def test_red_flag_forces_hard_escalation() -> None:
    fakes = Fakes(FakeTranscriber(), [], [], [])
    pipeline = _build(fakes)

    def _triggered_check(parsed: ParsedInput) -> RedFlagResult:
        fakes.order.append("safety")
        rf = RedFlagResult(
            triggered=True,
            flag_class="stroke",
            message="hard override",
            matched_symptoms=["facial_droop"],
        )
        fakes.captured_red_flag.append(rf)
        return rf

    pipeline._check_red_flags = _triggered_check

    outcome = pipeline.run(FakeRequest(), "case-rf")

    assert outcome.result.route == TriageRoute.HARD_ESCALATION
    assert fakes.captured_score[-1].route == TriageRoute.HARD_ESCALATION


def test_audio_path_uses_transcribe_bytes() -> None:
    fakes = Fakes(FakeTranscriber(), [], [], [])
    pipeline = _build(fakes)

    pipeline.run(
        FakeRequest(audio_b64="aGVsbG8="), "case-audio"
    )  # base64 of "hello"

    assert fakes.transcriber.bytes_calls == [b"hello"]
    assert fakes.transcriber.text_calls == []


def test_on_stage_emits_running_then_completed() -> None:
    fakes = Fakes(FakeTranscriber(), [], [], [])
    pipeline = _build(fakes)

    transitions: list[tuple[str, str]] = []
    pipeline.run(FakeRequest(), "c", on_stage=lambda n, s, p: transitions.append((n, s)))

    for name in ["asr", "parser", "safety", "scorer", "agent"]:
        idx = [n for n, _ in transitions].index(name)
        assert transitions[idx] == (name, "running")
        assert transitions[idx + 1] == (name, "completed")
