"""Intake distress / safeguarding pre-check (deterministic, pre-LLM).

Mirrors the shape of ``red_flag_checker.check_red_flags`` but evaluates caller
distress, escalation, and vulnerable-person safeguarding concerns from
``DISTRESS_PATTERNS``. Reuses the shared ``has_symptom`` matcher from
``clinical_knowledge``. The red-flag logic is NOT modified — this is an
independent parallel check; either triggering forces ``escalate_to_clinician``
for an intake Encounter.
"""

from __future__ import annotations

import logging

from models import DistressResult, ParsedInput
from clinical_knowledge import DISTRESS_PATTERNS, symptom_alias_map, has_symptom

log = logging.getLogger(__name__)


def check_distress(parsed: ParsedInput) -> DistressResult:
    """Run safeguarding / distress patterns against the parsed input.

    Returns a DistressResult. If triggered=True, the intake pipeline must
    escalate to a human (clinician / safeguarding) alongside any red-flag hit.
    """
    text = parsed.transcript
    aliases = symptom_alias_map()
    parsed_symptoms = set(parsed.symptoms)

    for pattern in DISTRESS_PATTERNS:
        required_symptoms: set[str] = pattern["symptoms"]
        min_match: int = pattern.get("min_match", 1)
        require_any = pattern.get("require_any")

        matched = [s for s in required_symptoms if has_symptom(text, s, aliases, parsed_symptoms)]

        if require_any is not None:
            if not any(s in matched for s in require_any):
                continue

        if len(matched) >= min_match:
            reason = (
                f"DISTRESS — {pattern['display']}. "
                f"Matched {len(matched)} cues: {', '.join(matched)}. "
                f"{pattern['description']} Human review required."
            )
            log.warning("DISTRESS TRIGGERED: %s | matched=%s", pattern["class"], matched)
            return DistressResult(
                triggered=True,
                klass=pattern["class"],
                reason=reason,
                matched=matched,
            )

    return DistressResult(triggered=False)
