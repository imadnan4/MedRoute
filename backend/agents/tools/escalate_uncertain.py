"""Tool: Escalate Uncertain — clinician referral when confidence is low."""
from __future__ import annotations

import json

from langchain_core.tools import tool


@tool
def escalate_uncertain(symptoms_and_context: str) -> str:
    """Escalate to a clinician when confidence is too low for automated triage.
    Input: string describing patient context. Returns: escalation notice JSON.
    """
    return json.dumps({
        "escalation": True,
        "urgency": "urgent",
        "likely_condition": "Escalated — insufficient automated confidence",
        "differential": [],
        "recommendation": (
            f"Insufficient confidence for automated triage. "
            f"Patient: {symptoms_and_context}. Direct clinical interview required."
        ),
        "watch_for": [
            "Chest pain or breathlessness",
            "Neurologic deficit (face, speech, weakness)",
            "High fever or confusion",
            "Syncope or severe pain",
        ],
        "confidence": 0.3,
        "message": (
            f"Insufficient confidence for automated triage. "
            f"Patient: {symptoms_and_context}. Direct clinical interview required."
        ),
    })
