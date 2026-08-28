"""Stage 1 — Input Parser.

Extracts structured symptoms and mandatory patient context (age, pregnancy,
duration) from the raw ASR transcript. Deterministic — no LLM. Falls back to
light keyword heuristics so it works on Urdu/Hindi romanized transcripts too.
"""
from __future__ import annotations

import logging
import re

from models import ParsedInput, PatientContext, PregnancyStatus
from voice.transcriber import Transcript
from clinical_knowledge import SYMPTOM_LEXICON, SYNDROME_CLUSTERS

log = logging.getLogger(__name__)

AGE_PATTERNS = [
    re.compile(r"(\d+(?:\.\d+)?)\s*(?:year|yr|saal|sal)\s*(?:old)?", re.I),
    re.compile(r"(\d+(?:\.\d+)?)\s*(?:month|mo|mahina)\s*(?:old)?", re.I),
    re.compile(r"\bage[:\s]+(\d+(?:\.\d+)?)", re.I),
]

DURATION_PATTERNS = [
    re.compile(r"(?:for|since|from|do|se)\s+(\d+(?:\.\d+)?)\s*(?:day|din|days)", re.I),
    re.compile(r"(\d+(?:\.\d+)?)\s*(?:day|din|days)\s*(?:se|for|of)?", re.I),
    re.compile(r"(?:for|since)\s+(\d+(?:\.\d+)?)\s*(?:week|hafte|weeks)", re.I),
    re.compile(r"(\d+(?:\.\d+)?)\s*(?:week|hafte|weeks)", re.I),
]

PREGNANCY_PATTERNS = [
    (re.compile(r"3rd\s*trimester|third trimester|teesra trimester", re.I), PregnancyStatus.PREGNANT_3RD_TRIMESTER),
    (re.compile(r"pregnant|haal hamla|hamla|expecting", re.I), PregnancyStatus.PREGNANT),
]


def _extract_age(text: str, patient: PatientContext) -> None:
    for pat in AGE_PATTERNS:
        m = pat.search(text)
        if not m:
            continue
        val = float(m.group(1))
        if "month" in pat.pattern.lower() or "mo" in pat.pattern.lower() or "mahina" in pat.pattern.lower():
            if val < 24:
                patient.age_months = val
                patient.age_years = val / 12.0
            else:
                patient.age_years = val / 12.0
        else:
            patient.age_years = val
        return


def _extract_pregnancy(text: str, patient: PatientContext) -> None:
    for pat, status in PREGNANCY_PATTERNS:
        if pat.search(text):
            patient.pregnancy = status
            return


def _extract_duration(text: str, patient: PatientContext) -> None:
    for pat in DURATION_PATTERNS:
        m = pat.search(text)
        if not m:
            continue
        val = float(m.group(1))
        if "week" in pat.pattern.lower() or "hafte" in pat.pattern.lower():
            patient.duration_days = val * 7.0
        else:
            patient.duration_days = val
        return


def _extract_symptoms(text: str) -> list[str]:
    lowered = text.lower()
    found = []
    for symptom, aliases in SYMPTOM_LEXICON.items():
        if any(alias in lowered for alias in aliases):
            found.append(symptom)
    return found


def _detect_clusters(symptoms: list[str]) -> list[str]:
    symptom_set = set(symptoms)
    hits = []
    for name, members in SYNDROME_CLUSTERS.items():
        overlap = symptom_set & members
        # Cluster hit if ≥2 members (or ≥3 for b_symptoms to avoid noise)
        threshold = 3 if name == "b_symptoms" else 2
        if len(overlap) >= threshold:
            hits.append(name)
    return hits


def parse(transcript: Transcript) -> ParsedInput:
    """Parse a Transcript into structured ParsedInput."""
    text = transcript.text
    patient = PatientContext()
    _extract_age(text, patient)
    _extract_pregnancy(text, patient)
    _extract_duration(text, patient)
    symptoms = _extract_symptoms(text)
    clusters = _detect_clusters(symptoms)

    parsed = ParsedInput(
        transcript=text,
        language=transcript.language,
        symptoms=symptoms,
        patient=patient,
        symptom_clusters=clusters,
    )
    log.info(
        "Parsed input: symptoms=%s clusters=%s age=%s pregnancy=%s duration=%s",
        symptoms,
        clusters,
        patient.age_for_display,
        patient.pregnancy.value,
        patient.duration_days,
    )
    return parsed
