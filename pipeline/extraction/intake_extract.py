"""Intake mode — structured capture adapter (MedRoute).

Turns a Presentation transcript into an IntakeRecord-shaped dict. This is the
lighter, non-clinical sibling of the triage agent: it extracts who is calling,
the care recipient, the care need, and logistics — never a diagnosis.

The call goes through the injected ``InferenceAdapter`` (exactly like
``run_triage``), so no network happens here. The prompt and the strict JSON
schema live in this module; the adapter merely forwards the payload.
"""

from __future__ import annotations

from typing import Any

from models import IntakeRecord

INTAKE_PROMPT_HOME_HEALTH = """You are an intake coordinator for a home-health and care-coordination service.
Read the caller's message and capture structured intake information. Be neutral and factual.
Fill every field you can; use null when the information is not stated. Do NOT diagnose or give medical advice.

Return JSON with these fields:
- contact_name: the person who made contact (caller/caregiver), or null
- care_recipient_name: the person whose care is discussed, or null
- phone_or_contact: a phone number or contact handle, or null
- care_need_summary: a short summary of what care or help is needed
- condition_or_issue: the stated condition, issue, or concern, or null
- mobility_or_severity_notes: any notes on mobility, severity, or functioning, or null
- insurance_or_payment_notes: any insurance or payment details, or null
- preferred_availability: when the person wants to be seen / available, or null
- free_text_summary: a one-paragraph plain-language summary of the whole message

Caller message:
\"\"\"{transcript}\"\"\"
"""

INTAKE_PROMPT_LEGAL = """You are an intake coordinator for a legal-aid and advice service (non-clinical).
Read the caller's message and capture structured intake information. Be neutral and factual.
Fill every field you can; use null when the information is not stated. Do NOT give legal advice.

For this legal domain, map the same generic fields as follows:
- contact_name: the person who made contact (caller/client), or null
- care_recipient_name: the person affected by the matter, or null
- phone_or_contact: a phone number or contact handle, or null
- care_need_summary: a short summary of the MATTER (what legal help is needed)
- condition_or_issue: the MATTER TYPE (e.g. eviction, custody, debt, benefits, domestic abuse)
- mobility_or_severity_notes: the URGENCY of the matter and any risk/deadline notes, or null
- insurance_or_payment_notes: any eligibility / means-test / funding notes, or null
- preferred_availability: when the person wants to be contacted, or null
- free_text_summary: a one-paragraph plain-language summary of the whole message

Caller message:
\"\"\"{transcript}\"\"\"
"""

INTAKE_JSON_SCHEMA: dict = {
    "type": "json_schema",
    "json_schema": {
        "name": "intake_record",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "contact_name",
                "care_recipient_name",
                "phone_or_contact",
                "care_need_summary",
                "condition_or_issue",
                "mobility_or_severity_notes",
                "insurance_or_payment_notes",
                "preferred_availability",
                "free_text_summary",
            ],
            "properties": {
                "contact_name": {"type": ["string", "null"]},
                "care_recipient_name": {"type": ["string", "null"]},
                "phone_or_contact": {"type": ["string", "null"]},
                "care_need_summary": {"type": "string"},
                "condition_or_issue": {"type": ["string", "null"]},
                "mobility_or_severity_notes": {"type": ["string", "null"]},
                "insurance_or_payment_notes": {"type": ["string", "null"]},
                "preferred_availability": {"type": ["string", "null"]},
                "free_text_summary": {"type": "string"},
            },
        },
    },
}


def extract_intake(transcript: str, inference: Any, domain: str = "home_health") -> dict:
    """Extract an IntakeRecord-shaped dict from a transcript via the inference adapter.

    ``inference`` satisfies the ``InferenceAdapter`` protocol (one ``infer`` method).
    ``domain`` selects the intake variant prompt ("home_health" | "legal") but the
    returned schema is identical — generic fields that serve both domains. Only the
    known IntakeRecord fields are returned, so ``Encounter.extracted`` is always
    well-shaped even if the model adds noise.
    """
    prompt = (
        INTAKE_PROMPT_LEGAL
        if domain == "legal"
        else INTAKE_PROMPT_HOME_HEALTH
    )
    payload = {
        "context": prompt.format(transcript=transcript),
        "response_format": INTAKE_JSON_SCHEMA,
    }
    data = inference.infer(payload)
    if not isinstance(data, dict):
        data = {}
    return IntakeRecord.from_dict(data).to_dict()
