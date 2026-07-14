"""Stage 3 — Complexity Scorer.

Computes a raw clinical complexity score (1-9) from structured symptoms,
syndrome clusters, and patient context, then applies calibrated confidence
and escalation bias to drive routing.

Design goals (aligned with MDAgents / CLARITY-style hybrid triage):
  - Deterministic pre-routing (no LLM) for auditability and low latency
  - Complexity-aware compute allocation: simple → local, complex → remote
  - Conservative under uncertainty: vague/underspecified cases escalate
  - Context multipliers for vulnerable populations (infants, elderly, pregnancy)

Deterministic — no LLM.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from config import settings
from models import (
    ParsedInput,
    PatientContext,
    PregnancyStatus,
    TriageRoute,
    TriageScore,
)

log = logging.getLogger(__name__)

# Base complexity weight per symptom (1-3). Higher = more diagnostically complex.
SYMPTOM_COMPLEXITY = {
    "fever": 1,
    "cough": 1,
    "cold": 1,
    "headache": 1,
    "sore_throat": 1,
    "dizziness": 1,
    "rash": 1,
    "vomiting": 2,
    "diarrhea": 2,
    "abdominal_pain": 2,
    "chest_pain": 3,
    "arm_pain": 3,
    "shortness_of_breath": 3,
    "sweating": 2,
    "fatigue": 1,  # alone is nonspecific; clusters raise score
    "weight_loss": 3,
    "night_sweats": 3,
    "lymph_node_swelling": 3,
    "bleeding": 2,
    "infant_fever": 2,
    "severe_headache": 3,
    "facial_droop": 3,
    "speech_difficulty": 3,
    "unilateral_weakness": 3,
    "confusion": 3,
    "stridor": 3,
    "cyanosis": 3,
    "swelling_face_throat": 3,
    "syncope": 3,
}

# Number of distinct symptoms -> extra complexity
SYMPTOM_COUNT_BONUS = {
    0: 0,
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
}

# Syndrome cluster bonuses — multi-symptom patterns that raise diagnostic complexity
CLUSTER_BONUS = {
    "viral_uri": 0,  # common, low complexity
    "b_symptoms": 3,  # lymphoma/TB/HIV workup territory
    "acs_constellation": 2,  # even if red-flag didn't fire, escalate compute
    "gi_illness": 1,
    "neuro_acute": 3,
    "respiratory_distress": 2,
}

# Confidence heuristics: specific findings boost; vagueness penalizes
CONFIDENCE_BASE = 0.50
SYMPTOM_CONFIDENCE_BOOST = {
    "chest_pain": 0.12,
    "arm_pain": 0.08,
    "shortness_of_breath": 0.08,
    "sweating": 0.06,
    "lymph_node_swelling": 0.10,
    "weight_loss": 0.08,
    "night_sweats": 0.08,
    "fever": 0.05,
    "cough": 0.03,
    "cold": 0.04,
    "facial_droop": 0.15,
    "speech_difficulty": 0.15,
    "unilateral_weakness": 0.15,
    "severe_headache": 0.10,
    "cyanosis": 0.12,
    "stridor": 0.12,
}

# Vague / nonspecific tokens that lower confidence when they dominate
VAGUE_SYMPTOMS = {"fatigue"}


@dataclass
class _ContextOffsets:
    age_offset: int = 0
    pregnancy_offset: int = 0


def _compute_context_offset(patient: PatientContext) -> _ContextOffsets:
    """Apply age and pregnancy multipliers per safety layer."""
    offsets = _ContextOffsets()

    if patient.age_months is not None:
        age_yrs = patient.age_months / 12.0
    elif patient.age_years is not None:
        age_yrs = patient.age_years
    else:
        age_yrs = None

    if age_yrs is not None:
        if age_yrs < 0.25:  # < 3 months
            offsets.age_offset = settings.age_lt_3mo_offset
        elif age_yrs < 2:
            offsets.age_offset = settings.age_lt_2yr_offset
        elif age_yrs > 65:
            offsets.age_offset = settings.age_gt_65_offset

    if patient.pregnancy == PregnancyStatus.PREGNANT_3RD_TRIMESTER:
        offsets.pregnancy_offset = settings.pregnancy_3rd_trimester_offset
    elif patient.pregnancy == PregnancyStatus.PREGNANT:
        offsets.pregnancy_offset = settings.pregnancy_offset

    return offsets


def _calculate_confidence(
    symptoms: list[str], clusters: list[str], patient: PatientContext
) -> tuple[float, float]:
    """Calibrated confidence based on specificity, count, and vagueness.

    Returns (confidence, vagueness_penalty).
    """
    conf = CONFIDENCE_BASE
    for s in symptoms:
        conf += SYMPTOM_CONFIDENCE_BOOST.get(s, 0.02)

    # More *specific* symptoms = more information
    conf += min(len(symptoms) * 0.02, 0.12)

    # Clusters of coherent syndromes raise confidence slightly
    if clusters:
        conf += min(0.04 * len(clusters), 0.08)

    # Clear primary-care URI pattern is high-confidence when no red-cluster noise
    if "viral_uri" in clusters and not (
        set(clusters)
        & {"b_symptoms", "acs_constellation", "neuro_acute", "respiratory_distress"}
    ):
        conf += 0.12

    # Duration known → slight boost (more complete history)
    if patient.duration_days is not None:
        conf += 0.03
        # Prolonged B-symptoms (weeks) are more informative
        if patient.duration_days >= 14 and "b_symptoms" in clusters:
            conf += 0.04

    # Vagueness penalty: no symptoms, or only nonspecific
    vagueness = 0.0
    if len(symptoms) == 0:
        vagueness = 0.25
    elif set(symptoms).issubset(VAGUE_SYMPTOMS):
        vagueness = 0.20
    elif len(symptoms) == 1 and symptoms[0] in VAGUE_SYMPTOMS | {
        "fatigue",
        "dizziness",
        "headache",
    }:
        vagueness = 0.12

    # Unknown age is incomplete intake → mild penalty
    if patient.age_years is None and patient.age_months is None:
        vagueness += 0.05

    conf -= vagueness
    return max(0.15, min(conf, 0.95)), vagueness


def _raw_complexity(
    symptoms: list[str], clusters: list[str], patient: PatientContext
) -> int:
    """Sum of symptom weights + count bonus + cluster bonus + chronicity."""
    if not symptoms:
        # Underspecified case — mid-low raw score; confidence bias will escalate
        return 2

    base = sum(SYMPTOM_COMPLEXITY.get(s, 1) for s in symptoms)
    bonus = SYMPTOM_COUNT_BONUS.get(len(symptoms), 4)
    cluster_extra = sum(CLUSTER_BONUS.get(c, 0) for c in clusters)

    # Multiple coherent URI symptoms add diagnostic signal, not complexity.
    uncomplicated_uri_symptoms = {
        "fever",
        "cough",
        "cold",
        "headache",
        "sore_throat",
    }
    serious_clusters = {
        "b_symptoms",
        "acs_constellation",
        "neuro_acute",
        "respiratory_distress",
    }
    if (
        "viral_uri" in clusters
        and set(symptoms).issubset(uncomplicated_uri_symptoms)
        and not set(clusters) & serious_clusters
    ):
        return min(base + bonus, settings.local_only_max_complexity)

    # Chronicity: multi-week systemic symptoms raise complexity
    chronicity = 0
    if patient.duration_days is not None and patient.duration_days >= 21:
        if any(
            s in symptoms
            for s in ("weight_loss", "night_sweats", "fatigue", "lymph_node_swelling")
        ):
            chronicity = 1

    return min(base + bonus + cluster_extra + chronicity, 9)


def _determine_route(
    adjusted_score: int, confidence: float
) -> tuple[TriageRoute, int, bool]:
    """Map adjusted score + confidence to a routing decision.

    Returns (route, score_after_bias, bias_applied).

    Routing philosophy (complexity-aware orchestration):
      score ≤3 + high conf  → direct model inference
      score ≤6 + high conf  → model + RAG evidence
      score ≥7              → complex-case model inference + RAG
      low conf              → escalation bias (+2) then re-evaluate;
                              if still mid-range → ESCALATION_BIAS path
    """
    bias_applied = False
    score_after_bias = adjusted_score
    if confidence < settings.escalation_bias_threshold:
        score_after_bias = min(adjusted_score + 2, 9)
        bias_applied = True

    if (
        score_after_bias <= settings.local_only_max_complexity
        and confidence >= settings.escalation_bias_threshold
    ):
        return TriageRoute.LOCAL_ONLY, score_after_bias, bias_applied
    if (
        score_after_bias <= settings.rag_max_complexity
        and confidence >= settings.escalation_bias_threshold
    ):
        return TriageRoute.LOCAL_WITH_RAG, score_after_bias, bias_applied
    if score_after_bias >= settings.remote_min_complexity:
        return TriageRoute.REMOTE, score_after_bias, bias_applied

    if bias_applied:
        return TriageRoute.ESCALATION_BIAS, score_after_bias, bias_applied

    return TriageRoute.LOCAL_WITH_RAG, score_after_bias, bias_applied


def score_complexity(parsed: ParsedInput) -> TriageScore:
    """Main entry: compute raw score, apply context offsets, determine route."""
    offsets = _compute_context_offset(parsed.patient)
    context_total = offsets.age_offset + offsets.pregnancy_offset
    clusters = list(parsed.symptom_clusters)

    raw = _raw_complexity(parsed.symptoms, clusters, parsed.patient)
    adjusted = min(raw + context_total, 9)
    confidence, vagueness = _calculate_confidence(
        parsed.symptoms, clusters, parsed.patient
    )

    route, final_score, bias_applied = _determine_route(adjusted, confidence)

    parts = [f"raw_score={raw}"]
    if offsets.age_offset:
        parts.append(f"age_offset=+{offsets.age_offset}")
    if offsets.pregnancy_offset:
        parts.append(f"pregnancy_offset=+{offsets.pregnancy_offset}")
    if clusters:
        parts.append(f"clusters={','.join(clusters)}")
    if vagueness:
        parts.append(f"vagueness_penalty=-{vagueness:.2f}")
    parts.append(f"adjusted_score={final_score}")
    parts.append(f"confidence={confidence:.2f}")
    if bias_applied:
        parts.append("ESCALATION_BIAS_APPLIED")
    parts.append(f"route={route.value}")

    reasoning = " | ".join(parts)
    log.info("Complexity scoring: %s", reasoning)

    return TriageScore(
        raw_score=raw,
        adjusted_score=final_score,
        confidence=confidence,
        escalation_bias_applied=bias_applied,
        context_offset=context_total,
        route=route,
        reasoning=reasoning,
        syndrome_hits=clusters,
        vagueness_penalty=vagueness,
    )
