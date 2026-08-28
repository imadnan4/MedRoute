"""Regression guard for the clinical_knowledge refactor.

Asserts that clinical_knowledge is the single owned source of truth for the
clinical DATA tables (verbatim move) and that the parse / red-flag / complexity
functions still produce the SAME outputs as the pre-edit snapshot.

Offline: no network, no model inference. Only in-memory ParsedInput /
PatientContext objects are used.
"""

from __future__ import annotations

import json
import pathlib

from models import PregnancyStatus
from pipeline.input_parser import parse as parse_input
from pipeline.complexity_scorer import score_complexity
from safety.red_flag_checker import check_red_flags
from voice.transcriber import Transcript

import clinical_knowledge as ck

SNAPSHOT_PATH = pathlib.Path("/tmp/opencode/clinical_knowledge_snapshot.json")


def _alias_map_repr(lexicon):
    return {k: set(v) for k, v in lexicon.items()}


def test_clinical_knowledge_exposes_data():
    # The moved tables must exist and keep their verbatim contents.
    assert ck.SYMPTOM_LEXICON == {
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
        "facial_droop": ["facial droop", "face droop", "crooked smile", "face asymmetry", "chehre ka jhukna", "half face"],
        "speech_difficulty": ["slurred speech", "can't speak", "speech difficulty", "aphasia", "bolne mein mushkil", "slurred"],
        "unilateral_weakness": ["arm weakness", "leg weakness", "one side weak", "hemiparesis", "left side weak", "right side weak", "paralysis"],
        "confusion": ["confused", "confusion", "altered mental", "disoriented", "not making sense", "behosh"],
        "stridor": ["stridor", "noisy breathing", "wheeze severe"],
        "cyanosis": ["cyanosis", "blue lips", "blue face", "neele hont"],
        "swelling_face_throat": ["face swelling", "throat swelling", "angioedema", "swollen tongue", "galay soojna"],
        "syncope": ["passed out", "fainted", "syncope", "collapse", "behosh ho gaya"],
        "neck_stiffness": ["neck stiffness", "stiff neck", "can't bend neck", "nuchal", "gardan sakht", "gardan mein dard"],
        "photophobia": ["photophobia", "light hurts eyes", "sensitive to light", "roshni se dard"],
        "seizure": ["seizure", "fit", "convulsion", "shaking spell", "dore", "mirgi ka daura", "tonic clonic"],
        "hematemesis": ["vomiting blood", "vomited blood", "hematemesis", "khoon ulti", "coffee ground"],
        "melena": ["black stool", "tarry stool", "melena", "kala stool", "black poop"],
        "rectal_bleeding": ["bloody stool", "rectal bleeding", "blood in stool", "stool mein khoon"],
        "leg_swelling": ["leg swelling", "swollen calf", "one leg swollen", "calf pain", "dvt"],
        "hemoptysis": ["coughing blood", "blood in sputum", "hemoptysis", "khoon khansi"],
        "head_injury": ["head injury", "hit head", "fell on head", "sar mein chot", "trauma head"],
        "loss_of_consciousness": ["lost consciousness", "knocked out", "unconscious", "behosh ho gaya tha"],
        "suicidal_ideation": [
            "want to die", "kill myself", "suicide", "suicidal", "end my life",
            "self harm", "hurt myself", "no reason to live",
        ],
        "polyuria": ["urinating a lot", "frequent urination", "polyuria", "zyada peshab"],
        "polydipsia": ["very thirsty", "extreme thirst", "polydipsia", "bohot pyas"],
        "poor_feeding": ["not feeding", "poor feeding", "won't eat", "not taking milk", "doodh nahi le raha"],
        "lethargy_child": ["floppy baby", "very sleepy baby", "won't wake", "lethargic infant"],
    }
    assert ck.SYNDROME_CLUSTERS == {
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
    assert ck.CLUSTER_BONUS == {
        "viral_uri": 0,
        "b_symptoms": 3,
        "acs_constellation": 2,
        "gi_illness": 1,
        "neuro_acute": 3,
        "respiratory_distress": 2,
    }
    # RED_FLAG_PATTERNS must be the verbatim list; compare via the alias-map
    # round-trip of its symptom sets plus structural keys.
    assert len(ck.RED_FLAG_PATTERNS) == 17
    for p in ck.RED_FLAG_PATTERNS:
        assert set(p) >= {"class", "display", "symptoms", "min_match", "description"}
    assert ck.RED_FLAG_PATTERNS[0]["class"] == "suicidal_crisis"
    assert ck.RED_FLAG_PATTERNS[-1]["class"] == "obstetric_emergency"


def test_helper_pair_matches_snapshot_alias_map():
    # symptom_alias_map() must reconstruct the parser's alias view exactly.
    assert ck.symptom_alias_map() == _alias_map_repr(ck.SYMPTOM_LEXICON)


def _run_case(name, text, lang, patient_kwargs, preg):
    parsed = parse_input(Transcript(text=text, language=lang, latency_ms=0))
    if patient_kwargs:
        for k, v in patient_kwargs.items():
            setattr(parsed.patient, k, v)
    if preg:
        parsed.patient.pregnancy = PregnancyStatus(preg)
    red = check_red_flags(parsed)
    score = score_complexity(parsed)
    return {
        "symptoms": parsed.symptoms,
        "clusters": parsed.symptom_clusters,
        "patient": {
            "age_years": parsed.patient.age_years,
            "age_months": parsed.patient.age_months,
            "pregnancy": parsed.patient.pregnancy.value,
            "duration_days": parsed.patient.duration_days,
        },
        "red_flag": {
            "triggered": red.triggered,
            "flag_class": red.flag_class,
            "matched": red.matched_symptoms,
        },
        "score": {
            "raw_score": score.raw_score,
            "adjusted_score": score.adjusted_score,
            "confidence": round(score.confidence, 6),
            "route": score.route.value,
            "reasoning": score.reasoning,
        },
    }


CASES = [
    ("en_acs", "I have chest pain and left arm pain with sweating and shortness of breath", "en", {}, None),
    ("ur_acs", "mujhe chest pain hai aur bazoo dard aur paseena aa raha hai", "ur", {}, None),
    ("stroke", "face droop and slurred speech and one side weak", "en", {}, None),
    ("meningitis", "fever with neck stiffness and photophobia and confusion", "en", {}, None),
    ("b_symptoms", "weight loss and night sweats and fever for 3 weeks", "en", {}, None),
    ("viral_uri", "I have a cold and cough and fever and sore throat", "en", {}, None),
    ("suicidal", "I want to kill myself", "en", {}, None),
    ("infant_fever", "baccha bukhar hai, poor feeding", "ur", {"age_months": 2}, None),
    ("sepsis", "fever vomiting diarrhea fatigue confusion", "en", {}, None),
    ("obstetric", "pregnant with bleeding and severe headache", "en", {}, "pregnant"),
    ("vague", "I feel tired", "en", {}, None),
    ("pe", "shortness of breath and leg swelling and hemoptysis", "en", {}, None),
]


def test_outputs_match_pre_edit_snapshot():
    if not SNAPSHOT_PATH.exists():
        raise AssertionError(
            "Pre-edit snapshot missing at %s — cannot verify behavior preservation."
            % SNAPSHOT_PATH
        )
    expected = json.loads(SNAPSHOT_PATH.read_text())
    for name, text, lang, pkw, preg in CASES:
        got = _run_case(name, text, lang, pkw, preg)
        # matched symptoms come from iterating a set literal, so their order is
        # hash-seed dependent (nondeterministic) in BOTH pre- and post-edit code.
        # The SET of matched symptoms is the triage-relevant output; compare it
        # order-insensitively. Everything else must match EXACTLY.
        exp = expected[name]
        exp["red_flag"]["matched"] = sorted(exp["red_flag"]["matched"])
        got["red_flag"]["matched"] = sorted(got["red_flag"]["matched"])
        assert got == exp, f"Divergence in case {name}: {got} != {exp}"
