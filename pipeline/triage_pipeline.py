"""Triage orchestration — single deep module owning the full stage sequence.

The pipeline is the one source of truth for the ORDER of stages:

    asr -> parser -> safety -> scorer -> agent

and for assembling the structured outcome (``stages`` dict + ``TriageResult``)
that the web layer previously hand-built in two divergent places (the batch
``/triage`` route and the SSE ``/triage/stream`` route).

Both HTTP endpoints LEVERAGE this one interface. The optional ``on_stage``
callback lets the SSE endpoint observe progress (``running`` / ``completed`` per
stage plus a final ``done``) without the pipeline knowing anything about SSE,
keeping the module pure of web/global state — there is deliberately NO
``_triage_store`` here.

Dependencies are injected at construction (constructor injection) so the module
is testable with fakes and never imports-and-instantiates its stages.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol

from models import (
    Acuity,
    Disposition,
    Encounter,
    ParsedInput,
    RedFlagResult,
    TriageResult,
    TriageRoute,
)
from pipeline.extraction.intake_extract import extract_intake as _default_extract_intake
from safety.distress_checker import check_distress as _default_check_distress
from voice.transcriber import Transcript

OnStage = Callable[[str, str, dict], None]


def _acuity_from_score(adjusted_score: int) -> Acuity:
    """Map a complexity scorer's adjusted score onto the unified Acuity vocabulary."""
    if adjusted_score >= 7:
        return Acuity.URGENT
    if adjusted_score >= 4:
        return Acuity.PRIORITY
    return Acuity.ROUTINE


def _disposition_from_route(route: TriageRoute, red_flag_triggered: bool) -> Disposition:
    """Map a triage TriageRoute onto the unified Disposition vocabulary."""
    if red_flag_triggered or route in (
        TriageRoute.ESCALATION_BIAS,
        TriageRoute.HARD_ESCALATION,
        TriageRoute.OUTAGE_FALLBACK,
    ):
        return Disposition.ESCALATE_TO_CLINICIAN
    if route in (TriageRoute.REMOTE, TriageRoute.LOCAL_WITH_RAG):
        return Disposition.PROVIDE_GUIDANCE
    return Disposition.STANDARD_QUEUE


class TriageInput(Protocol):
    """Structural type for the request the pipeline accepts (decoupled from the
    web-layer ``TriageRequest`` to avoid a circular import)."""

    transcript: str
    language: str
    audio_b64: Optional[str]
    age_years: Optional[float]
    age_months: Optional[float]
    pregnancy: Optional[str]


@dataclass
class PipelineOutcome:
    case_id: str
    request: dict
    stages: dict
    result: TriageResult


class TriagePipeline:
    def __init__(
        self,
        transcriber: Any,
        parse_input: Callable[[Transcript], Any],
        check_red_flags: Callable[[Any], Any],
        score_complexity: Callable[[Any], Any],
        run_triage: Callable[[Any, Any, Any], TriageResult],
        *,
        extract_intake: Callable[[str, Any], dict] = _default_extract_intake,
        inference: Any = None,
        check_distress: Callable[[Any], Any] = _default_check_distress,
    ) -> None:
        self._transcriber = transcriber
        self._parse_input = parse_input
        self._check_red_flags = check_red_flags
        self._check_distress = check_distress
        self._score_complexity = score_complexity
        self._run_triage = run_triage
        self._extract_intake = extract_intake
        self._inference = inference

    def run(
        self,
        request: TriageInput,
        case_id: str,
        *,
        mode: str = "triage",
        on_stage: Optional[OnStage] = None,
        domain: str = "home_health",
    ) -> Any:
        """Run the full triage pipeline and return a structured outcome.

        ``on_stage(name, status, payload)`` is invoked for every stage transition
        (``running`` then ``completed``) and once for ``done``. Pass ``None`` for
        the batch path; pass a callback for SSE streaming.
        """
        stages: dict = {}

        def _emit(name: str, status: str, payload: dict | None = None) -> None:
            if on_stage is not None:
                on_stage(name, status, payload or {})

        # Stage 0 — Voice / Input
        _emit("asr", "running")
        if request.audio_b64:
            try:
                audio_bytes = base64.b64decode(request.audio_b64)
                transcript = self._transcriber.transcribe_bytes(audio_bytes)
                stages["asr"] = {
                    "status": "completed",
                    "text": transcript.text,
                    "language": transcript.language,
                    "latency_ms": transcript.latency_ms,
                }
            except Exception:
                transcript = self._transcriber.transcribe_text(
                    request.transcript, request.language
                )
                stages["asr"] = {
                    "status": "completed",
                    "text": transcript.text,
                    "language": transcript.language,
                    "fallback": True,
                }
        else:
            transcript = self._transcriber.transcribe_text(
                request.transcript, request.language
            )
            stages["asr"] = {
                "status": "completed",
                "text": transcript.text,
                "language": transcript.language,
            }
        _emit("asr", "completed", stages["asr"])

        input_mode = "audio" if request.audio_b64 else "text"

        if mode == "intake":
            return self._run_intake(request, case_id, stages, transcript, input_mode, _emit, domain)

        transcript = self._apply_context_overrides(
            transcript.text, request.language, request
        )

        # Stage 1 — Input Parser
        _emit("parser", "running")
        parsed = self._parse_input(transcript)
        stages["parser"] = {
            "status": "completed",
            "symptoms": parsed.symptoms,
            "clusters": parsed.symptom_clusters,
            "age": parsed.patient.age_for_display,
            "pregnancy": parsed.patient.pregnancy.value,
            "duration_days": parsed.patient.duration_days,
        }
        _emit("parser", "completed", stages["parser"])

        # Stage 2 — Safety Pre-Check (hard override before any LLM)
        _emit("safety", "running")
        red_flag = self._check_red_flags(parsed)
        stages["safety"] = {"status": "completed", "triggered": red_flag.triggered}
        if red_flag.triggered:
            stages["safety"]["flag_class"] = red_flag.flag_class
            stages["safety"]["message"] = red_flag.message
            stages["safety"]["matched"] = red_flag.matched_symptoms
        _emit("safety", "completed", stages["safety"])

        # Stage 3 — Complexity Scorer
        _emit("scorer", "running")
        score = self._score_complexity(parsed)
        stages["scorer"] = {
            "status": "completed",
            "raw_score": score.raw_score,
            "adjusted_score": score.adjusted_score,
            "confidence": score.confidence,
            "route": score.route.value,
            "syndrome_hits": score.syndrome_hits,
            "reasoning": score.reasoning,
        }
        _emit("scorer", "completed", stages["scorer"])

        if red_flag.triggered:
            score.route = TriageRoute.HARD_ESCALATION

        # Stage 4 — Deterministic triage orchestrator
        _emit("agent", "running")
        result = self._run_triage(parsed, score, red_flag)
        stages["agent"] = {
            "status": "completed",
            "route": result.route.value,
            "likely_condition": result.likely_condition,
            "urgency": result.urgency.value,
            "cascade": result.cascade_used,
            "confidence": result.confidence,
            "confidence_level": result.confidence_level.value,
        }
        _emit("agent", "completed", stages["agent"])

        result.acuity = _acuity_from_score(score.adjusted_score)
        result.disposition = _disposition_from_route(result.route, red_flag.triggered)

        outcome = PipelineOutcome(
            case_id=case_id,
            request=_request_dump(request),
            stages=stages,
            result=result,
        )
        _emit(
            "done",
            "completed",
            {
                "case_id": case_id,
                "acuity": result.acuity.value,
                "disposition": result.disposition.value,
                "result": result.model_dump(),
            },
        )
        return outcome

    def _run_intake(
        self,
        request: TriageInput,
        case_id: str,
        stages: dict,
        transcript: Transcript,
        input_mode: str,
        _emit: OnStage,
        domain: str = "home_health",
    ) -> Encounter:
        """Lighter, non-clinical intake path.

        Stage order: asr (shared) -> safety (shared Red-Flag pre-check + intake
        distress/safeguarding pre-check) -> intake extraction -> disposition. It
        never invokes the triage agent, the complexity scorer, or any RAG
        guidance. Both safety checks are deterministic and pre-LLM; EITHER a Red
        Flag OR a distress cue forces ``escalate_to_clinician`` + human review.
        """
        if self._inference is None:
            raise ValueError(
                "Intake mode requires an injected inference adapter "
                "(pass `inference=` to TriagePipeline)."
            )

        # Safety Pre-Check — build a minimal ParsedInput just for the safety passes.
        _emit("safety", "running")
        parsed_for_safety = ParsedInput(transcript=transcript.text, language=transcript.language)
        red_flag = self._check_red_flags(parsed_for_safety)
        distress = self._check_distress(parsed_for_safety)
        stages["safety"] = {
            "status": "completed",
            "triggered": red_flag.triggered or distress.triggered,
        }
        if red_flag.triggered:
            stages["safety"]["flag_class"] = red_flag.flag_class
            stages["safety"]["message"] = red_flag.message
            stages["safety"]["matched"] = red_flag.matched_symptoms
        if distress.triggered:
            stages["safety"]["distress_class"] = distress.klass
            stages["safety"]["distress_reason"] = distress.reason
            stages["safety"]["distress_matched"] = distress.matched
        _emit("safety", "completed", stages["safety"])

        # Intake extraction (new adapter) — never a diagnosis; domain-aware prompt.
        _emit("intake", "running")
        extracted = self._extract_intake(transcript.text, self._inference, domain=domain)
        stages["intake"] = {"status": "completed", "domain": domain}
        _emit("intake", "completed", stages["intake"])

        escalate = red_flag.triggered or distress.triggered
        disposition = (
            Disposition.ESCALATE_TO_CLINICIAN
            if escalate
            else Disposition.STANDARD_QUEUE
        )

        encounter = Encounter(
            mode="intake",
            input_mode=input_mode,
            input_language=transcript.language,
            raw_transcript=transcript.text,
            red_flag=red_flag,
            acuity=None,
            disposition=disposition,
            extracted=extracted,
            case_id=case_id,
            needs_human_review=escalate,
            distress=distress,
            domain=domain,
        )
        _emit(
            "done",
            "completed",
            {"case_id": case_id, "disposition": disposition.value},
        )
        return encounter

    @staticmethod
    def _apply_context_overrides(
        transcript_text: str, language: str, request: TriageInput
    ) -> Transcript:
        """Merge explicit age/pregnancy fields into the transcript for the parser."""
        if (
            request.age_years is None
            and request.age_months is None
            and request.pregnancy is None
        ):
            return Transcript(text=transcript_text, language=language, latency_ms=0)

        user_text = transcript_text
        if request.age_years is not None:
            user_text += f" Age: {request.age_years} years"
        if request.age_months is not None:
            user_text += f" Age: {request.age_months} months"
        if request.pregnancy:
            user_text += f" Pregnancy: {request.pregnancy}"
        return Transcript(text=user_text, language=language, latency_ms=0)


def _request_dump(request: TriageInput) -> dict:
    return {
        "transcript": request.transcript,
        "language": request.language,
        "audio_b64": request.audio_b64,
        "age_years": request.age_years,
        "age_months": request.age_months,
        "pregnancy": request.pregnancy,
    }
