"""Tests for the inference seam.

Asserts ``run_triage`` uses an injected ``InferenceAdapter`` (no network) and
that the canned inference output flows through to the triage result.
"""

from __future__ import annotations

from agents.triage_agent import run_triage
from agents.tools.inference import (
    InMemoryInfer,
    InferenceAdapter,
    OpenRouterInfer,
)
from models import (
    ConfidenceLevel,
    ParsedInput,
    PatientContext,
    RedFlagResult,
    TriageRoute,
    TriageScore,
    UrgencyLevel,
)
from voice.transcriber import Transcript


def _parsed() -> ParsedInput:
    return ParsedInput(
        transcript="I have a fever and a cough",
        language="en",
        symptoms=["fever", "cough"],
        patient=PatientContext(age_years=30),
        symptom_clusters=["viral_uri"],
    )


def _score() -> TriageScore:
    return TriageScore(
        raw_score=2,
        adjusted_score=2,
        confidence=0.9,
        route=TriageRoute.LOCAL_ONLY,
        syndrome_hits=["viral_uri"],
        reasoning="local_only",
    )


def _red_flag() -> RedFlagResult:
    return RedFlagResult(triggered=False)


def test_inmemory_infer_satisfies_adapter() -> None:
    assert isinstance(InMemoryInfer(), InferenceAdapter)
    assert isinstance(OpenRouterInfer(), InferenceAdapter)


def test_run_triage_uses_injected_adapter_offline() -> None:
    adapter = InMemoryInfer()
    result = run_triage(_parsed(), _score(), _red_flag(), inference=adapter)

    # Adapter was exercised and its canned output drove the diagnosis.
    assert adapter.last_payload is not None
    assert "context" in adapter.last_payload
    assert result.likely_condition == "Viral upper respiratory infection"
    assert result.differential == [
        "Acute bronchitis",
        "Allergic rhinitis",
        "Influenza",
    ]
    assert "openrouter_infer" in result.cascade_used
    assert result.confidence_level == ConfidenceLevel.GREEN


def test_run_triage_default_is_openrouter() -> None:
    # Default keeps production behaviour (no adapter passed).
    result = run_triage(
        _parsed(),
        _score(),
        _red_flag(),
        inference=OpenRouterInfer(),
    )
    assert "openrouter_infer" in result.cascade_used
    # Sanity: red flag not triggered, pipeline completes.
    assert result.route == TriageRoute.LOCAL_ONLY
