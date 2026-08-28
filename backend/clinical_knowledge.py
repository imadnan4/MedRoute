"""Single owned source of truth for clinical-knowledge DATA.

Holds the symptom lexicon, syndrome clusters, cluster-scoring bonuses, and
red-flag patterns plus the shared alias-matching helpers. Imported by the input
parser, complexity scorer, and red-flag checker so these tables live in ONE
place. Imports ONLY from `models` to avoid import cycles.
"""

from __future__ import annotations

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

# Syndrome cluster bonuses — multi-symptom patterns that raise diagnostic complexity
CLUSTER_BONUS = {
    "viral_uri": 0,  # common, low complexity
    "b_symptoms": 3,  # lymphoma/TB/HIV workup territory
    "acs_constellation": 2,  # even if red-flag didn't fire, escalate compute
    "gi_illness": 1,
    "neuro_acute": 3,
    "respiratory_distress": 2,
}

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


# Intake distress / safeguarding patterns — caller distress, escalation, and
# vulnerable-person risk. Deterministic, pre-LLM, sample-appropriate. Each pattern
# mirrors RED_FLAG_PATTERNS' shape (class, display, symptoms, min_match,
# optional require_any) but the phrases are safeguarding cues, not clinical
# symptoms. `symptoms` entries are verbatim phrases matched as substrings.
DISTRESS_PATTERNS: list[dict] = [
    {
        "class": "caller_distress",
        "display": "Caller Unable To Cope",
        "symptoms": {"can't cope", "cannot cope", "unable to cope", "at my wits end", "overwhelmed", "can't manage"},
        "min_match": 1,
        "description": "Caller reports being unable to cope or manage — offer supportive handoff.",
    },
    {
        "class": "abusive_escalating_tone",
        "display": "Abusive / Escalating Tone",
        "symptoms": {"abusive", "shouting", "yelling", "threatening me", "swearing", "being aggressive"},
        "min_match": 1,
        "description": "Abusive, threatening, or escalating tone — safe-handling and de-escalation.",
    },
    {
        "class": "vulnerable_adult_at_risk",
        "display": "Vulnerable Adult At Risk",
        "symptoms": {"vulnerable adult", "adult at risk", "social services", "being neglected", "neglected at home"},
        "min_match": 1,
        "description": "Possible vulnerable adult at risk of neglect — safeguarding referral.",
    },
    {
        "class": "child_at_risk",
        "display": "Child At Risk",
        "symptoms": {"child at risk", "child protection", "children not safe", "kids aren't safe", "harming the children"},
        "min_match": 1,
        "description": "Disclosure suggesting a child may be at risk — safeguarding referral.",
    },
    {
        "class": "domestic_abuse",
        "display": "Domestic Abuse Disclosure",
        "symptoms": {"domestic abuse", "partner hits me", "afraid of my husband", "controlling my money", "threatened by my partner", "scared at home"},
        "min_match": 1,
        "description": "Domestic abuse disclosure — safe contact and specialist referral.",
    },
    {
        "class": "self_neglect",
        "display": "Self-Neglect",
        "symptoms": {"not been eating", "not looking after myself", "not been out", "self neglect", "not taking care", "not been washing"},
        "min_match": 1,
        "description": "Signs of self-neglect — assess support and safeguarding need.",
    },
    {
        "class": "urgent_safeguarding_escalation",
        "display": "Urgent Safeguarding Escalation",
        "symptoms": {"need someone now", "can't wait", "before something happens", "urgent help", "someone needs to come"},
        "min_match": 1,
        "description": "Caller demands urgent intervention / feared imminent harm — escalate now.",
    },
]


def symptom_alias_map() -> dict[str, set[str]]:
    return {k: set(v) for k, v in SYMPTOM_LEXICON.items()}


def has_symptom(text: str, symptom_key: str, aliases: dict[str, set[str]], parsed_symptoms: set[str]) -> bool:
    """Match via structured symptom keys first, then transcript aliases."""
    if symptom_key in parsed_symptoms:
        return True
    aliases_for_key = aliases.get(symptom_key, {symptom_key})
    lowered = text.lower()
    return any(alias in lowered for alias in aliases_for_key)
