"""Groq-hosted LLM inference (free tier, no OpenRouter credits required).

We use JSON-object mode (``response_format: {type: "json_object"}``) so the same
code works across free Groq models (Qwen, gpt-oss) without depending on strict
``json_schema`` support. The expected field shapes are described in the prompts.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import requests

from config import settings
from agents.tools.openrouter_infer import (
    SYSTEM_PROMPT,
    INTAKE_SYSTEM_HOME_HEALTH,
    INTAKE_SYSTEM_LEGAL,
)

log = logging.getLogger(__name__)

_GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
_JSON_OBJECT = {"type": "json_object"}

INTAKE_SCHEMA_TEXT = (
    "Respond with a single JSON object using exactly these keys "
    "(use null for anything unstated): "
    "contact_name (string), care_recipient_name (string), "
    "phone_or_contact (string), care_need_summary (string), "
    "condition_or_issue (string), mobility_or_severity_notes (string), "
    "insurance_or_payment_notes (string), preferred_availability (string), "
    "free_text_summary (string)."
)


def _looks_like_roman_urdu(text: str) -> bool:
    markers = [
        "dard", "buhat", "bukhar", "thakan", "ghabrahat", "chakkar", "kamzori",
        "mummy", "papa", "doctor", "theek", "nahi", "hai", "hain", "mere", "ko",
    ]
    lowered = (text or "").lower()
    return sum(1 for m in markers if m in lowered) >= 2


def _unavailable(reason: str) -> str:
    return json.dumps(
        {
            "likely_condition": "Assessment unavailable",
            "differential": [],
            "recommendation": (
                "Automated inference is temporarily unavailable. "
                "Please consult a clinician for this case."
            ),
            "watch_for": [],
            "confidence": 0.0,
            "urgency": "routine",
            "unavailable": True,
            "reason": reason,
        }
    )


def _chat(system: str, user: str, max_tokens: int):
    if not settings.groq_api_key:
        return None
    body = {
        "model": settings.groq_llm_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "response_format": _JSON_OBJECT,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    headers = {
        "Authorization": f"Bearer {settings.groq_api_key}",
        "Content-Type": "application/json",
    }
    attempts = max(int(settings.openrouter_max_attempts), 1)
    for attempt in range(1, attempts + 1):
        try:
            resp = requests.post(
                _GROQ_URL, headers=headers, json=body, timeout=settings.openrouter_timeout
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:  # noqa: BLE001
            log.warning("Groq inference attempt %d failed: %s", attempt, exc)
    return None


def infer_text(symptoms_and_context: str) -> str:
    """Run medical decision-support inference through Groq. Returns JSON string."""
    lang_note = ""
    if _looks_like_roman_urdu(symptoms_and_context):
        lang_note = (
            " The patient's family may write in Roman Urdu (e.g. 'meri mummy ko "
            "dard hai'). Understand it, but write the assessment in English."
        )
    system = SYSTEM_PROMPT + lang_note
    payload = _chat(system, f"Patient: {symptoms_and_context}", max_tokens=800)
    if payload is None:
        return _unavailable("groq_unreachable")

    content = (
        payload.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
    )
    content = content.strip()
    if not content:
        return _unavailable("groq_empty_response")
    try:
        json.loads(content)
    except json.JSONDecodeError:
        return _unavailable("groq_unparseable")
    return content


def infer_intake(transcript: str, domain: str = "home_health") -> dict:
    """Run intake extraction through Groq using JSON-object mode."""
    base = (
        INTAKE_SYSTEM_LEGAL
        if (domain or "home_health") == "legal"
        else INTAKE_SYSTEM_HOME_HEALTH
    )
    system = base + " " + INTAKE_SCHEMA_TEXT
    payload = _chat(system, transcript, max_tokens=600)
    if payload is None:
        return {}
    content = (
        payload.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
    )
    content = content.strip()
    if not content:
        return {}
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {}
