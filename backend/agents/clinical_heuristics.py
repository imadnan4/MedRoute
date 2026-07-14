"""Rule-based clinical templates for offline / cascade fallback.

Used when LLMs are unavailable so demo scenarios and low-resource clinics still
get structured, safe guidance. Never replaces red-flag hard escalation.
"""
from __future__ import annotations

from models import ParsedInput, UrgencyLevel


def heuristic_assessment(parsed: ParsedInput) -> dict:
    """Return a structured diagnosis dict from deterministic rules.

    Keys: likely_condition, differential, recommendation, watch_for,
          confidence, urgency
    """
    symptoms = set(parsed.symptoms)
    clusters = set(parsed.symptom_clusters)
    age = parsed.patient.age_years
    duration = parsed.patient.duration_days

    # --- B-symptoms / constitutional (demo C: lymphoma/TB territory) ---
    if "b_symptoms" in clusters or (
        {"weight_loss", "night_sweats"} <= symptoms
        or ({"weight_loss", "fatigue"} <= symptoms and (duration or 0) >= 14)
    ):
        return {
            "likely_condition": "Constitutional / B-symptom syndrome — evaluate for TB, lymphoma, HIV, chronic infection",
            "differential": [
                "Tuberculosis / chronic infection",
                "Lymphoproliferative disorder",
                "HIV / occult malignancy",
            ],
            "recommendation": (
                "Same-day or urgent clinician review. Obtain vitals, weight trend, "
                "lymph node exam, and basic labs (CBC, ESR/CRP). Consider chest imaging "
                "and infectious workup based on local epidemiology."
            ),
            "watch_for": [
                "Progressive weight loss",
                "Night sweats worsening",
                "New lymphadenopathy",
                "Hemoptysis or high fever",
            ],
            "confidence": 0.72,
            "urgency": UrgencyLevel.URGENT.value,
        }

    # --- Viral URI pattern (demo A) ---
    if "viral_uri" in clusters or (
        len(symptoms & {"cold", "cough", "fever", "sore_throat", "headache"}) >= 2
        and not (symptoms & {"chest_pain", "shortness_of_breath", "weight_loss", "night_sweats"})
    ):
        return {
            "likely_condition": "Likely viral upper respiratory infection (common cold / viral URI)",
            "differential": [
                "Viral URI / common cold",
                "Seasonal influenza-like illness",
                "Allergic rhinitis (if predominant sneezing/congestion without fever)",
            ],
            "recommendation": (
                "Supportive care: rest, fluids, antipyretic if needed. Return if fever >3 days, "
                "breathing difficulty, chest pain, or inability to take fluids. "
                "No antibiotics routinely for uncomplicated viral URI."
            ),
            "watch_for": [
                "Breathing difficulty or chest pain",
                "High fever lasting >3 days",
                "Neck stiffness or severe headache",
                "Dehydration (no urine >8h)",
            ],
            "confidence": 0.82,
            "urgency": UrgencyLevel.ROUTINE.value,
        }

    # --- Meningitis-like (should usually be hard-escalated; heuristic if gate missed) ---
    if "meningitis_constellation" in clusters or (
        {"fever", "neck_stiffness"} <= symptoms or {"fever", "photophobia"} <= symptoms
    ):
        return {
            "likely_condition": "Possible CNS infection / meningitis spectrum — emergency evaluation",
            "differential": [
                "Bacterial meningitis",
                "Viral meningoencephalitis",
                "Severe migraine (less likely if fever + neck stiffness)",
            ],
            "recommendation": (
                "Immediate emergency care. Do not delay for outpatient wait. "
                "This is decision support only — clinician must assess for lumbar puncture / antibiotics."
            ),
            "watch_for": ["Worsening headache", "Confusion", "Rash", "Seizure"],
            "confidence": 0.78,
            "urgency": UrgencyLevel.EMERGENCY.value,
        }

    # --- PE constellation ---
    if "pe_constellation" in clusters or (
        {"shortness_of_breath", "leg_swelling"} <= symptoms
        or {"shortness_of_breath", "hemoptysis"} <= symptoms
    ):
        return {
            "likely_condition": "Possible pulmonary embolism / serious cardiopulmonary cause",
            "differential": [
                "Pulmonary embolism",
                "Pneumonia / pneumothorax",
                "ACS with atypical features",
            ],
            "recommendation": "Urgent emergency assessment with vitals, SpO2, ECG, and PE workup as indicated.",
            "watch_for": ["Syncope", "Worsening dyspnea", "Hemoptysis", "Chest pain"],
            "confidence": 0.74,
            "urgency": UrgencyLevel.EMERGENCY.value,
        }

    # --- DKA / metabolic ---
    if "dka_constellation" in clusters or (
        {"polyuria", "polydipsia"} <= symptoms and symptoms & {"vomiting", "confusion", "abdominal_pain"}
    ):
        return {
            "likely_condition": "Possible diabetic emergency (DKA / hyperosmolar state)",
            "differential": [
                "Diabetic ketoacidosis",
                "Hyperosmolar hyperglycemic state",
                "Uncontrolled diabetes with dehydration",
            ],
            "recommendation": "Urgent clinical care with glucose check, electrolytes, and fluids. Do not delay.",
            "watch_for": ["Confusion", "Vomiting", "Deep rapid breathing", "Syncope"],
            "confidence": 0.76,
            "urgency": UrgencyLevel.EMERGENCY.value,
        }

    # --- GI illness ---
    if "gi_illness" in clusters or len(symptoms & {"vomiting", "diarrhea", "abdominal_pain"}) >= 2:
        return {
            "likely_condition": "Acute gastroenteritis / GI illness (dehydration risk)",
            "differential": [
                "Viral gastroenteritis",
                "Food-borne illness",
                "Bacterial enteritis (if bloody stools / high fever)",
            ],
            "recommendation": (
                "Oral rehydration salts, small frequent sips. Seek care if unable to keep fluids, "
                "bloody stool, severe abdominal pain, or signs of dehydration."
            ),
            "watch_for": [
                "No urine output >8 hours",
                "Bloody stool",
                "Severe abdominal pain",
                "Persistent vomiting",
            ],
            "confidence": 0.75,
            "urgency": UrgencyLevel.SOON.value,
        }

    # --- Vague / low information (demo D) ---
    if not symptoms or symptoms.issubset({"fatigue"}) or (
        len(symptoms) <= 1 and "fatigue" in symptoms
    ):
        return {
            "likely_condition": "Insufficient information for automated triage",
            "differential": [
                "Nonspecific presentation — needs clinical interview",
                "Possible systemic illness (cannot exclude)",
                "Psychosocial / functional fatigue (after organic workup)",
            ],
            "recommendation": (
                "Direct clinician interview required. Expand history: onset, associated symptoms, "
                "vitals, comorbidities, medications. Do not rely on automated diagnosis."
            ),
            "watch_for": [
                "Chest pain or breathlessness",
                "Neurologic symptoms (speech, face, weakness)",
                "High fever or confusion",
                "Syncope or severe pain",
            ],
            "confidence": 0.35,
            "urgency": UrgencyLevel.URGENT.value,
        }

    # --- Elderly with chest/respiratory ---
    if age is not None and age >= 65 and symptoms & {"chest_pain", "shortness_of_breath", "dizziness"}:
        return {
            "likely_condition": "Concerning cardiopulmonary symptoms in older adult — urgent evaluation",
            "differential": [
                "Acute coronary syndrome",
                "Heart failure / arrhythmia",
                "Pulmonary embolism / pneumonia",
            ],
            "recommendation": "Urgent same-day clinical assessment with ECG and vitals. Do not delay.",
            "watch_for": ["Worsening chest pain", "Syncope", "Severe breathlessness", "Sweating"],
            "confidence": 0.70,
            "urgency": UrgencyLevel.URGENT.value,
        }

    # --- Default: structured handoff ---
    symptom_list = ", ".join(symptoms) if symptoms else "unspecified"
    return {
        "likely_condition": f"Symptom-directed assessment needed ({symptom_list})",
        "differential": [
            "Requires fuller history and exam",
            "Cannot exclude serious causes from text alone",
        ],
        "recommendation": (
            "Clinician review recommended. Document vitals, red-flag screen, and focused exam. "
            "This automated path is decision support only."
        ),
        "watch_for": [
            "Any red-flag symptom (chest pain, neuro deficit, severe dyspnea)",
            "Rapid clinical deterioration",
        ],
        "confidence": 0.55,
        "urgency": UrgencyLevel.SOON.value,
    }
