"""Triage Agent — Deterministic route orchestrator with optional ReAct assist.

Architecture (research-aligned: CLARITY FSM + MDAgents complexity routing):
  1. Hard red flags already handled upstream — this module never second-guesses them
  2. Route from Complexity Scorer selects a fixed tool plan (not free-form agent choice)
  3. Cascade on failure: local → remote → heuristics → escalate_uncertain
  4. Confidence is fused: min/blend of scorer confidence and model self-report
  5. RAG evidence is always attached when retrieved

Why not pure ReAct for routing?
  Free-form ReAct often ignores route hints, wastes iterations, and fails open.
  Safety-critical triage needs deterministic control flow; LLMs fill the diagnosis
  slot, not the routing policy.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

from agents.clinical_heuristics import heuristic_assessment
from agents.tools.escalate_uncertain import escalate_uncertain
from agents.tools.local_infer import local_infer
from agents.tools.rag_search import rag_search
from agents.tools.remote_infer import remote_infer
from config import settings
from models import (
    ConfidenceLevel,
    ParsedInput,
    RedFlagResult,
    TriageResult,
    TriageRoute,
    TriageScore,
    UrgencyLevel,
)

log = logging.getLogger(__name__)


def _extract_json(text: str) -> dict:
    """Robustly extract the last valid JSON object from model/tool output."""
    if not text:
        raise ValueError("Empty output")
    text = text.strip()
    # Direct parse
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    # Find embedded object
    matches = list(re.finditer(r"\{", text))
    for m in reversed(matches):
        try:
            candidate = text[m.start():]
            end = candidate.rfind("}")
            if end == -1:
                continue
            data = json.loads(candidate[:end + 1])
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            continue
    raise ValueError(f"No valid JSON in output: {text[:300]}")


def _confidence_level(conf: float) -> ConfidenceLevel:
    if conf >= settings.confidence_high:
        return ConfidenceLevel.GREEN
    if conf >= settings.confidence_medium:
        return ConfidenceLevel.YELLOW
    return ConfidenceLevel.RED


def _parse_urgency(value: Any, route: TriageRoute, conf: float) -> UrgencyLevel:
    if isinstance(value, str):
        v = value.lower().strip()
        for level in UrgencyLevel:
            if level.value == v:
                return level
    if route == TriageRoute.HARD_ESCALATION:
        return UrgencyLevel.EMERGENCY
    if conf < settings.confidence_medium or route in (
        TriageRoute.ESCALATION_BIAS,
        TriageRoute.OUTAGE_FALLBACK,
    ):
        return UrgencyLevel.URGENT
    if route == TriageRoute.REMOTE:
        return UrgencyLevel.SOON
    return UrgencyLevel.ROUTINE


def _patient_context_str(parsed: ParsedInput) -> str:
    symptoms_str = ", ".join(parsed.symptoms) if parsed.symptoms else "unspecified"
    duration = (
        f"{parsed.patient.duration_days:.0f} days"
        if parsed.patient.duration_days is not None
        else "unknown"
    )
    clusters = ", ".join(parsed.symptom_clusters) if parsed.symptom_clusters else "none"
    return (
        f"Symptoms: {symptoms_str}. "
        f"Clusters: {clusters}. "
        f"Age: {parsed.patient.age_for_display}. "
        f"Pregnancy: {parsed.patient.pregnancy.value}. "
        f"Duration: {duration}. "
        f"Transcript: {parsed.transcript}"
    )


def _is_unavailable(data: dict) -> bool:
    status = str(data.get("status", "")).lower()
    if status in ("local_unavailable", "remote_unavailable", "error"):
        return True
    if data.get("action") in ("fallback_to_remote", "refer_to_clinician"):
        return True
    if not data.get("likely_condition") and data.get("escalation"):
        return True
    return False


def _tool_json(tool_fn, arg: str) -> dict:
    """Invoke a LangChain tool and parse JSON (or wrap plain text)."""
    try:
        raw = tool_fn.invoke(arg)
    except Exception as exc:
        log.error("Tool %s failed: %s", getattr(tool_fn, "name", tool_fn), exc)
        return {"status": "error", "reason": str(exc)}

    if isinstance(raw, dict):
        return raw
    try:
        return _extract_json(str(raw))
    except ValueError:
        # Non-JSON tool output (e.g. rag prose) — wrap
        return {"raw": str(raw)}


def _retrieve_rag(query: str) -> list[str]:
    try:
        raw = rag_search.invoke(query)
        if not raw or "No relevant" in raw:
            return []
        # Split multi-doc join from tool
        parts = [p.strip() for p in re.split(r"\n\n---\n\n", str(raw)) if p.strip()]
        return parts[: settings.rag_top_k]
    except Exception as exc:
        log.warning("RAG retrieve failed: %s", exc)
        return []


def _fuse_confidence(
    scorer_conf: float,
    model_conf: Optional[float],
    route: TriageRoute,
) -> float:
    """Conservative fusion of scorer + model confidence.

    Medical LLM literature favors calibrated under-confidence over overconfidence.
    When both signals agree (high), allow green; when they disagree, prefer lower.
    """
    if model_conf is None:
        final = scorer_conf
    else:
        model_conf = max(0.0, min(float(model_conf), 1.0))
        # If either side is low, take the lower (safety)
        if model_conf < settings.confidence_medium or scorer_conf < settings.confidence_medium:
            final = min(scorer_conf, model_conf)
        else:
            # Both reasonably confident — blend, allow modest lift toward agreement
            blend = 0.4 * scorer_conf + 0.6 * model_conf
            final = min(blend, max(scorer_conf, model_conf))

    if route in (TriageRoute.ESCALATION_BIAS, TriageRoute.OUTAGE_FALLBACK):
        final = min(final, settings.confidence_medium - 0.01)

    return max(0.1, min(final, 0.95))


def _normalize_diagnosis(data: dict) -> dict:
    """Normalize tool/heuristic JSON into a common shape."""
    differential = data.get("differential") or []
    if isinstance(differential, str):
        differential = [differential]
    watch = data.get("watch_for") or data.get("watch for") or []
    if isinstance(watch, str):
        watch = [watch]

    conf = data.get("confidence")
    if conf is not None:
        try:
            conf = float(conf)
            if conf > 1.0:  # sometimes models return 0-100
                conf = conf / 100.0
        except (TypeError, ValueError):
            conf = None

    return {
        "likely_condition": str(data.get("likely_condition") or data.get("diagnosis") or "").strip(),
        "differential": [str(x) for x in differential][:5],
        "recommendation": str(data.get("recommendation") or data.get("message") or "").strip(),
        "watch_for": [str(x) for x in watch][:6],
        "confidence": conf,
        "urgency": data.get("urgency"),
    }


def _plan_for_route(route: TriageRoute) -> list[str]:
    """Fixed tool plans — the routing policy is code, not the LLM."""
    if route == TriageRoute.LOCAL_ONLY:
        return ["local"]
    if route == TriageRoute.LOCAL_WITH_RAG:
        return ["rag", "local"]
    if route == TriageRoute.REMOTE:
        return ["rag", "remote"]
    if route == TriageRoute.ESCALATION_BIAS:
        return ["rag", "remote", "local"]
    if route == TriageRoute.OUTAGE_FALLBACK:
        return ["local", "heuristic", "escalate"]
    return ["rag", "local", "remote"]


def run_triage(
    parsed: ParsedInput,
    score: TriageScore,
    red_flag: RedFlagResult,
) -> TriageResult:
    """Run the full triage agent pipeline with deterministic orchestration."""

    if red_flag.triggered:
        return TriageResult(
            route=TriageRoute.HARD_ESCALATION,
            confidence=1.0,
            confidence_level=ConfidenceLevel.RED,
            likely_condition=red_flag.flag_class or "Emergency",
            recommendation=red_flag.message,
            red_flag=red_flag,
            patient=parsed.patient,
            urgency=UrgencyLevel.EMERGENCY,
            scorer_confidence=score.confidence,
            cascade_used=["hard_red_flag"],
        )

    context = _patient_context_str(parsed)
    cascade: list[str] = []
    rag_evidence: list[str] = []
    diagnosis: Optional[dict] = None
    final_route = score.route

    plan = _plan_for_route(score.route)
    log.info("Triage plan for %s: %s", score.route.value, plan)

    for step in plan:
        if step == "rag":
            cascade.append("rag_search")
            query = (
                f"{', '.join(parsed.symptoms) or parsed.transcript}. "
                f"Age {parsed.patient.age_for_display}"
            )
            rag_evidence = _retrieve_rag(query)
            if rag_evidence:
                cascade.append(f"rag_hits={len(rag_evidence)}")
            continue

        if step == "local":
            cascade.append("local_infer")
            payload = context
            if rag_evidence:
                payload += "\n\nEvidence from guidelines:\n" + "\n".join(rag_evidence[:3])
            data = _tool_json(local_infer, payload)
            if _is_unavailable(data):
                cascade.append("local_failed")
                continue
            diagnosis = _normalize_diagnosis(data)
            if diagnosis["likely_condition"]:
                break
            diagnosis = None
            continue

        if step == "remote":
            cascade.append("remote_infer")
            payload = context
            if rag_evidence:
                payload += "\n\nEvidence from guidelines:\n" + "\n".join(rag_evidence[:3])
            data = _tool_json(remote_infer, payload)
            if _is_unavailable(data):
                cascade.append("remote_failed")
                final_route = TriageRoute.OUTAGE_FALLBACK
                continue
            diagnosis = _normalize_diagnosis(data)
            if diagnosis["likely_condition"]:
                break
            diagnosis = None
            continue

        if step == "heuristic":
            cascade.append("clinical_heuristics")
            diagnosis = _normalize_diagnosis(heuristic_assessment(parsed))
            break

        if step == "escalate":
            cascade.append("escalate_uncertain")
            data = _tool_json(escalate_uncertain, context)
            diagnosis = _normalize_diagnosis({
                "likely_condition": "Escalated — insufficient automated confidence",
                "differential": [],
                "recommendation": data.get("message", "Direct clinical interview required."),
                "watch_for": [],
                "confidence": 0.3,
                "urgency": UrgencyLevel.URGENT.value,
            })
            final_route = TriageRoute.OUTAGE_FALLBACK
            break

    # Full cascade fallback if plan exhausted without diagnosis
    if diagnosis is None or not diagnosis.get("likely_condition"):
        cascade.append("heuristic_fallback")
        diagnosis = _normalize_diagnosis(heuristic_assessment(parsed))
        # Distinguish outage (remote required but down) vs low-confidence escalation
        if score.route == TriageRoute.REMOTE and "remote_failed" in cascade:
            final_route = TriageRoute.OUTAGE_FALLBACK
        elif score.route == TriageRoute.ESCALATION_BIAS:
            final_route = TriageRoute.ESCALATION_BIAS
        elif "local_failed" in cascade and "remote_failed" in cascade:
            final_route = TriageRoute.OUTAGE_FALLBACK
        else:
            # Keep original route (e.g. local_only → heuristics still local path)
            final_route = score.route

    model_conf = diagnosis.get("confidence")
    fused = _fuse_confidence(score.confidence, model_conf, final_route)
    urgency = _parse_urgency(diagnosis.get("urgency"), final_route, fused)

    # Vague / escalation-bias cases: force clinician-facing recommendation tone
    recommendation = diagnosis.get("recommendation") or ""
    if final_route == TriageRoute.ESCALATION_BIAS and "clinician" not in recommendation.lower():
        recommendation = (
            recommendation + " Automated confidence is low — clinician confirmation required."
        ).strip()

    reasoning = (
        f"scorer: {score.reasoning}; "
        f"cascade: {' → '.join(cascade)}; "
        f"fused_confidence={fused:.2f}"
    )

    return TriageResult(
        route=final_route,
        confidence=fused,
        confidence_level=_confidence_level(fused),
        likely_condition=diagnosis.get("likely_condition", ""),
        differential=diagnosis.get("differential") or [],
        recommendation=recommendation,
        watch_for=diagnosis.get("watch_for") or [],
        rag_evidence=rag_evidence,
        patient=parsed.patient,
        reasoning=reasoning[:800],
        urgency=urgency,
        model_confidence=model_conf,
        scorer_confidence=score.confidence,
        cascade_used=cascade,
    )
