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

log = logging.getLogger(__name__)

# Symptom lexicon (English + romanized Urdu/Hindi). Keys feed scorer + red flags.
SYMPTOM_LEXICON: dict[str, list[str]] = {
    "fever": ["bukhar", "fever", "tap", "jwar", "temperature", "pyrexia"],
    "cough": ["khansi", "cough", "khaansi"],
    "cold": ["zukam", "cold", "nasal", "sneeze", "runny nose", "congestion"],
    "headache": ["sar dard", "headache", "sir dard", "sirdard", "head pain"],
    "severe_headache": ["thunderclap", "worst headache", "sudden severe headache", "sab se zyada sar dard"],
    "chest_pain": ["chest pain", "chati dard", "chest tightness", "seena dard", "chest pressure"],
    "arm_pain": ["arm pain", "haath dard", "left arm", "bazoo dard", "jaw pain", "radiation to arm"],
    "sweating": ["sweating", "paseena", "diaphoresis", "pasina", "cold sweat"],
    "shortness_of_breath": [
        "dyspnea", "saans", "breathless", "saans lena", "shortness of breath",
        "can't breathe", "difficulty breathing", "saans phoolna",
    ],
    "fatigue": ["fatigue", "thakaan", "weakness", "kamzor", "tired", "feel off", "something is wrong"],
    "weight_loss": ["weight loss", "wazan kam", "weight kam", "losing weight"],
    "night_sweats": ["night sweats", "raat paseena", "night sweat"],
    "lymph_node_swelling": ["lymph", "gland", "gland swelling", "lymph node", "swollen nodes"],
    "abdominal_pain": ["stomach pain", "pet dard", "abdominal pain", "belly", "pet mein dard"],
    "vomiting": ["vomiting", "ulti", "nausea", "ulta", "throwing up"],
    "diarrhea": ["diarrhea", "dast", "loose motion", "loose stools"],
    "rash": ["rash", "chhap", "skin rash", "dhaal", "hives", "urticaria"],
    "sore_throat": ["sore throat", "galay dard", "throat pain", "galay mein dard"],
    "dizziness": ["dizziness", "chakkar", "vertigo", "lightheaded", "faint"],
    "bleeding": ["bleeding", "khoon", "haemorrhage", "hemorrhage", "blood", "vaginal bleeding"],
    "infant_fever": ["infant fever", "baby fever", "baccha bukhar", "newborn fever"],
    # Neuro / FAST stroke keys
    "facial_droop": ["facial droop", "face droop", "crooked smile", "face asymmetry", "chehre ka jhukna", "half face"],
    "speech_difficulty": ["slurred speech", "can't speak", "speech difficulty", "aphasia", "bolne mein mushkil", "slurred"],
    "unilateral_weakness": ["arm weakness", "leg weakness", "one side weak", "hemiparesis", "left side weak", "right side weak", "paralysis"],
    "confusion": ["confused", "confusion", "altered mental", "disoriented", "not making sense", "behosh"],
    # Respiratory severity
    "stridor": ["stridor", "noisy breathing", "wheeze severe"],
    "cyanosis": ["cyanosis", "blue lips", "blue face", "neele hont"],
    # Allergic / anaphylaxis
    "swelling_face_throat": ["face swelling", "throat swelling", "angioedema", "swollen tongue", "galay soojna"],
    "syncope": ["passed out", "fainted", "syncope", "collapse", "behosh ho gaya"],
    # Meningitis / CNS infection
    "neck_stiffness": ["neck stiffness", "stiff neck", "can't bend neck", "nuchal", "gardan sakht", "gardan mein dard"],
    "photophobia": ["photophobia", "light hurts eyes", "sensitive to light", "roshni se dard"],
    # Seizure
    "seizure": ["seizure", "fit", "convulsion", "shaking spell", "dore", "mirgi ka daura", "tonic clonic"],
    # GI bleed
    "hematemesis": ["vomiting blood", "vomited blood", "hematemesis", "khoon ulti", "coffee ground"],
    "melena": ["black stool", "tarry stool", "melena", "kala stool", "black poop"],
    "rectal_bleeding": ["bloody stool", "rectal bleeding", "blood in stool", "stool mein khoon"],
    # PE / DVT signals
    "leg_swelling": ["leg swelling", "swollen calf", "one leg swollen", "calf pain", "dvt"],
    "hemoptysis": ["coughing blood", "blood in sputum", "hemoptysis", "khoon khansi"],
    # Trauma / head injury
    "head_injury": ["head injury", "hit head", "fell on head", "sar mein chot", "trauma head"],
    "loss_of_consciousness": ["lost consciousness", "knocked out", "unconscious", "behosh ho gaya tha"],
    # Mental health crisis
    "suicidal_ideation": [
        "want to die", "kill myself", "suicide", "suicidal", "end my life",
        "self harm", "hurt myself", "no reason to live",
    ],
    # Diabetic emergency
    "polyuria": ["urinating a lot", "frequent urination", "polyuria", "zyada peshab"],
    "polydipsia": ["very thirsty", "extreme thirst", "polydipsia", "bohot pyas"],
    # Pediatric severe illness cues
    "poor_feeding": ["not feeding", "poor feeding", "won't eat", "not taking milk", "doodh nahi le raha"],
    "lethargy_child": ["floppy baby", "very sleepy baby", "won't wake", "lethargic infant"],
}

# Clinical syndrome clusters for scoring (not red flags by themselves)
SYNDROME_CLUSTERS: dict[str, set[str]] = {
    "viral_uri": {"cold", "cough", "fever", "sore_throat", "headache"},
    "b_symptoms": {"fever", "night_sweats", "weight_loss", "fatigue", "lymph_node_swelling"},
    "acs_constellation": {"chest_pain", "arm_pain", "sweating", "shortness_of_breath", "dizziness"},
    "gi_illness": {"vomiting", "diarrhea", "abdominal_pain", "fever"},
    "neuro_acute": {"facial_droop", "speech_difficulty", "unilateral_weakness", "severe_headache", "confusion", "seizure"},
    "respiratory_distress": {"shortness_of_breath", "stridor", "cyanosis", "chest_pain"},
    "meningitis_constellation": {"fever", "severe_headache", "neck_stiffness", "photophobia", "confusion", "rash"},
    "pe_constellation": {"shortness_of_breath", "chest_pain", "leg_swelling", "hemoptysis", "syncope"},
    "gi_bleed": {"hematemesis", "melena", "rectal_bleeding", "dizziness", "fatigue", "syncope"},
    "dka_constellation": {"polyuria", "polydipsia", "vomiting", "fatigue", "confusion", "abdominal_pain"},
}

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
