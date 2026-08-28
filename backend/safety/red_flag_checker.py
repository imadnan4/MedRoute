"""Stage 2 — Safety Pre-Check (Hard Red-Flag Overrides).

Runs BEFORE any LLM / agent is invoked. Pattern classes cover time-critical,
life-threatening presentations. Any match triggers immediate hard escalation
to a clinician — no LLM, no score, no routing.

Design principles:
  - Conservative: over-escalate rather than miss a life threat
  - min_match prevents single-symptom false positives where appropriate
  - Stroke uses FAST-style neuro keys (not generic headache+dizziness)
  - Sepsis requires infection signal + multi-system signs (min_match ≥ 3)
  - Age / pregnancy gates for infant fever and obstetric emergencies

Aligned with common ED red-flag screening (ACS, BE-FAST/FAST, red-flag sepsis
concepts) adapted for keyword triage without vitals.
"""
from __future__ import annotations

import logging
from typing import Optional

from models import ParsedInput, PatientContext, PregnancyStatus, RedFlagResult
from clinical_knowledge import RED_FLAG_PATTERNS, symptom_alias_map, has_symptom

log = logging.getLogger(__name__)


def _patient_meets_age(patient: PatientContext, max_months: Optional[int]) -> bool:
    if max_months is None:
        return True
    if patient.age_months is not None:
        return patient.age_months <= max_months
    if patient.age_years is not None:
        return patient.age_years * 12 <= max_months
    return False


def _patient_is_pregnant(patient: PatientContext) -> bool:
    return patient.pregnancy != PregnancyStatus.NOT_PREGNANT


def check_red_flags(parsed: ParsedInput) -> RedFlagResult:
    """Run hard-override patterns against the parsed input.

    Returns a RedFlagResult. If triggered=True, the pipeline must HALT and
    escalate immediately to a clinician.
    """
    text = parsed.transcript
    patient = parsed.patient
    aliases = symptom_alias_map()
    parsed_symptoms = set(parsed.symptoms)

    for pattern in RED_FLAG_PATTERNS:
        if "age_max_months" in pattern and not _patient_meets_age(patient, pattern["age_max_months"]):
            continue

        if pattern.get("requires_pregnancy") and not _patient_is_pregnant(patient):
            continue

        required_symptoms: set[str] = pattern["symptoms"]
        min_match: int = pattern.get("min_match", 1)
        require_any: Optional[set[str]] = pattern.get("require_any")

        matched = [s for s in required_symptoms if has_symptom(text, s, aliases, parsed_symptoms)]

        if require_any is not None:
            if not any(s in matched for s in require_any):
                continue

        if len(matched) >= min_match:
            message = (
                f"HARD OVERRIDE — {pattern['display']}. "
                f"Matched {len(matched)} criteria: {', '.join(matched)}. "
                f"{pattern['description']} "
                f"Immediate physician assessment required. Do not delay."
            )
            log.warning("RED FLAG TRIGGERED: %s | matched=%s", pattern["class"], matched)
            return RedFlagResult(
                triggered=True,
                flag_class=pattern["class"],
                message=message,
                matched_symptoms=matched,
            )

    return RedFlagResult(triggered=False)
