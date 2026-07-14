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

log = logging.getLogger(__name__)

# Each pattern defines:
#   class, display, symptoms, min_match
#   optional: age_max_months, requires_pregnancy, require_any (must include ≥1 of)
# Order matters: more specific syndromes before broad patterns.
RED_FLAG_PATTERNS: list[dict] = [
    {
        "class": "suicidal_crisis",
        "display": "Suicidal Ideation / Mental Health Crisis",
        "symptoms": {"suicidal_ideation"},
        "min_match": 1,
        "description": "Expressed suicidal ideation or self-harm intent — immediate human crisis response.",
    },
    {
        "class": "anaphylaxis",
        "display": "Suspected Anaphylaxis / Airway Threat",
        "symptoms": {"swelling_face_throat", "shortness_of_breath", "rash", "stridor", "syncope"},
        "min_match": 2,
        "require_any": {"swelling_face_throat", "stridor", "shortness_of_breath"},
        "description": "Face/throat swelling, airway symptoms, or collapse with allergic features.",
    },
    {
        "class": "seizure",
        "display": "Seizure / Active Neurological Emergency",
        "symptoms": {"seizure", "confusion", "loss_of_consciousness", "unilateral_weakness"},
        "min_match": 1,
        "require_any": {"seizure"},
        "description": "Reported seizure or convulsion — urgent evaluation (status risk, first seizure workup).",
    },
    {
        "class": "gi_bleed",
        "display": "Suspected Gastrointestinal Bleed",
        "symptoms": {"hematemesis", "melena", "rectal_bleeding", "dizziness", "syncope", "fatigue"},
        "min_match": 1,
        "require_any": {"hematemesis", "melena", "rectal_bleeding"},
        "description": "Hematemesis, melena, or significant rectal bleeding — emergency evaluation.",
    },
    {
        "class": "meningitis",
        "display": "Suspected Meningitis / CNS Infection",
        "symptoms": {"fever", "severe_headache", "neck_stiffness", "photophobia", "confusion", "rash", "headache"},
        "min_match": 2,
        "require_any": {"neck_stiffness", "photophobia"},
        "description": "Fever with neck stiffness or photophobia (± severe headache, confusion, rash).",
    },
    {
        "class": "stroke",
        "display": "Suspected Stroke / TIA (FAST)",
        # Core FAST only — isolated headache/confusion are nonspecific and handled elsewhere
        "symptoms": {
            "facial_droop",
            "speech_difficulty",
            "unilateral_weakness",
        },
        "min_match": 1,
        "description": "Facial droop, arm/leg weakness, or speech difficulty — emergency stroke pathway.",
    },
    {
        "class": "thunderclap_headache",
        "display": "Thunderclap / Sudden Severe Headache (rule out SAH)",
        "symptoms": {"severe_headache", "syncope", "vomiting", "neck_stiffness", "confusion"},
        "min_match": 1,
        "require_any": {"severe_headache"},
        "description": "Sudden worst-ever headache — rule out subarachnoid hemorrhage urgently.",
    },
    {
        "class": "acute_coronary_syndrome",
        "display": "Suspected Acute Coronary Syndrome (MI / Unstable Angina)",
        "symptoms": {"chest_pain", "arm_pain", "sweating", "shortness_of_breath", "syncope"},
        "min_match": 2,
        "require_any": {"chest_pain", "arm_pain"},
        "description": "Chest pain/tightness with radiation, diaphoresis, dyspnea, or syncope.",
    },
    {
        "class": "pulmonary_embolism",
        "display": "Suspected Pulmonary Embolism",
        "symptoms": {"shortness_of_breath", "chest_pain", "leg_swelling", "hemoptysis", "syncope", "sweating"},
        "min_match": 2,
        "require_any": {"shortness_of_breath", "hemoptysis", "leg_swelling"},
        "description": "Sudden dyspnea/chest pain with DVT signs, hemoptysis, or syncope.",
    },
    {
        "class": "respiratory_emergency",
        "display": "Respiratory Emergency",
        "symptoms": {"shortness_of_breath", "stridor", "cyanosis", "chest_pain", "sweating"},
        "min_match": 2,
        "require_any": {"shortness_of_breath", "stridor", "cyanosis"},
        "description": "Severe dyspnea, stridor, cyanosis, or inability to speak in sentences.",
    },
    {
        "class": "head_trauma",
        "display": "Significant Head Injury",
        "symptoms": {"head_injury", "loss_of_consciousness", "vomiting", "confusion", "severe_headache", "seizure"},
        "min_match": 2,
        "require_any": {"head_injury", "loss_of_consciousness"},
        "description": "Head trauma with LOC, vomiting, confusion, severe headache, or seizure.",
    },
    {
        "class": "diabetic_emergency",
        "display": "Suspected Diabetic Emergency (DKA / HHS)",
        "symptoms": {"polyuria", "polydipsia", "vomiting", "confusion", "fatigue", "abdominal_pain", "shortness_of_breath"},
        "min_match": 3,
        "require_any": {"polyuria", "polydipsia"},
        "description": "Polyuria/polydipsia with systemic signs — rule out DKA/HHS urgently.",
    },
    {
        "class": "sepsis",
        "display": "Sepsis / Septic Shock (suspected)",
        "symptoms": {"fever", "vomiting", "diarrhea", "fatigue", "sweating", "confusion", "shortness_of_breath"},
        "min_match": 3,
        "require_any": {"fever"},
        "description": "Infection signal with multi-system signs (e.g. fever + GI + systemic).",
    },
    {
        "class": "infant_fever",
        "display": "Infant Fever (< 3 months)",
        "symptoms": {"fever", "infant_fever"},
        "min_match": 1,
        "age_max_months": 3,
        "description": "Any fever in infant < 90 days — always escalate.",
    },
    {
        "class": "sick_infant",
        "display": "Sick Infant / Poor Feeding (< 12 months)",
        "symptoms": {"poor_feeding", "lethargy_child", "fever", "vomiting", "cyanosis", "seizure"},
        "min_match": 1,
        "age_max_months": 12,
        "require_any": {"poor_feeding", "lethargy_child", "cyanosis", "seizure"},
        "description": "Poor feeding, lethargy, or cyanosis in infant — escalate promptly.",
    },
    {
        "class": "severe_dehydration",
        "display": "Severe Dehydration",
        "symptoms": {"vomiting", "diarrhea", "fatigue", "dizziness", "syncope"},
        "min_match": 3,
        "require_any": {"vomiting", "diarrhea"},
        "description": "Persistent vomiting/diarrhea with volume-depletion signs.",
    },
    {
        "class": "obstetric_emergency",
        "display": "Obstetric Emergency",
        "symptoms": {"bleeding", "abdominal_pain", "severe_headache", "dizziness", "syncope"},
        "min_match": 2,
        "requires_pregnancy": True,
        "description": "Bleeding, severe pain, severe headache, or syncope in pregnancy.",
    },
]


def _symptom_alias_map() -> dict[str, set[str]]:
    from pipeline.input_parser import SYMPTOM_LEXICON
    return {k: set(v) for k, v in SYMPTOM_LEXICON.items()}


def _has_symptom(text: str, symptom_key: str, aliases: dict[str, set[str]], parsed_symptoms: set[str]) -> bool:
    """Match via structured symptom keys first, then transcript aliases."""
    if symptom_key in parsed_symptoms:
        return True
    aliases_for_key = aliases.get(symptom_key, {symptom_key})
    lowered = text.lower()
    return any(alias in lowered for alias in aliases_for_key)


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
    aliases = _symptom_alias_map()
    parsed_symptoms = set(parsed.symptoms)

    for pattern in RED_FLAG_PATTERNS:
        if "age_max_months" in pattern and not _patient_meets_age(patient, pattern["age_max_months"]):
            continue

        if pattern.get("requires_pregnancy") and not _patient_is_pregnant(patient):
            continue

        required_symptoms: set[str] = pattern["symptoms"]
        min_match: int = pattern.get("min_match", 1)
        require_any: Optional[set[str]] = pattern.get("require_any")

        matched = [s for s in required_symptoms if _has_symptom(text, s, aliases, parsed_symptoms)]

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
