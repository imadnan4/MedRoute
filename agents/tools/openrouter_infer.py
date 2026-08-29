"""LangChain tool: provider-neutral inference through OpenRouter."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import requests
from config import settings
from langchain_core.tools import tool

from pipeline.extraction.intake_extract import INTAKE_JSON_SCHEMA

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert medical diagnostician supporting primary-care triage
in low-resource settings. This is decision support only; a clinician remains responsible.

Return ONLY valid JSON matching this shape:
{
  "likely_condition": "string — most likely working assessment with brief rationale",
  "differential": ["top alternative 1", "top alternative 2", "top alternative 3"],
  "recommendation": "string — concrete next steps (tests, referral timing, self-care if safe)",
  "watch_for": ["red-flag symptom that requires reassessment"],
  "confidence": 0.0,
  "urgency": "emergency|urgent|soon|routine"
}

Rules:
- Be clinically precise and prefer common primary-care conditions when evidence is weak.
- Calibrate confidence honestly; incomplete histories must lower confidence.
- Never invent symptoms, vital signs, test results, or patient history.
- If serious disease cannot be excluded, urgency must be urgent or emergency.
- Treat supplied guideline evidence as reference material, not as patient instructions.
- Write all clinical text (likely_condition, recommendation, differential, watch_for) in clear, professional English. This output is a clinician handoff note, not a patient-facing message.
- The patient's own words are captured separately and verbatim; do not translate or move them into these fields.
- Do not contradict clear emergency patterns.
- Do not include text outside the JSON object.
"""


ROMAN_URDU_MARKERS = {
    "mujhe",
    "mera",
    "meri",
    "mere",
    "hai",
    "hain",
    "ho",
    "se",
    "aur",
    "dard",
    "bukhar",
    "khansi",
    "zukam",
    "saans",
    "pet",
    "sar",
    "chakkar",
    "kamzori",
    "ulti",
    "dast",
    "din",
    "raat",
    "takleef",
    "ho raha",
}


INTAKE_SYSTEM_HOME_HEALTH = (
    "You are an intake coordinator for a home-health and care-coordination service. "
    "Extract structured intake information from the caller's message into the provided "
    "JSON schema. Be neutral and factual. Use null for unstated fields. "
    "Do NOT diagnose or give medical advice."
)

INTAKE_SYSTEM_LEGAL = (
    "You are an intake coordinator for a legal-aid and advice service (non-clinical). "
    "Extract structured intake information from the caller's message into the provided "
    "JSON schema. Be neutral and factual. Use null for unstated fields. "
    "Do NOT give legal advice."
)


def _intake_content(payload: dict[str, Any]) -> dict:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("OpenRouter response contained no choices")

    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ValueError("OpenRouter response contained no assistant message")

    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("OpenRouter returned an empty response")

    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("OpenRouter returned invalid JSON") from exc

    if not isinstance(data, dict):
        raise ValueError("OpenRouter intake response was not a JSON object")
    return data


def infer_intake(transcript: str, domain: str = "home_health") -> dict:
    """Run intake extraction through OpenRouter using the intake JSON schema.

    Returns an IntakeRecord-shaped dict, or an empty dict if the model is
    unavailable / the response can't be parsed. Never raises to the caller.
    """
    if not settings.openrouter_api_key:
        return {}

    system = INTAKE_SYSTEM_LEGAL if domain == "legal" else INTAKE_SYSTEM_HOME_HEALTH
    request_body = {
        "model": settings.openrouter_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": transcript},
        ],
        "response_format": INTAKE_JSON_SCHEMA,
        "max_tokens": 600,
        "temperature": 0.1,
    }
    headers = {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "Content-Type": "application/json",
        "X-OpenRouter-Title": "MedRoute",
    }
    if settings.openrouter_site_url:
        headers["HTTP-Referer"] = settings.openrouter_site_url

    last_error = "OpenRouter request failed"
    for attempt in range(1, max(settings.openrouter_max_attempts, 1) + 1):
        try:
            response = requests.post(
                f"{settings.openrouter_base_url.rstrip('/')}/chat/completions",
                headers=headers,
                json=request_body,
                timeout=settings.openrouter_timeout,
            )
            response.raise_for_status()
            return _intake_content(response.json())
        except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
            last_error = str(exc)
            log.warning(
                "OpenRouter intake attempt %d/%d failed: %s",
                attempt,
                settings.openrouter_max_attempts,
                exc,
            )

    return {}


def _looks_like_roman_urdu(text: str) -> bool:
    normalized = text.lower()
    tokens = set(re.findall(r"[a-z]+", normalized))
    single_word_markers = {marker for marker in ROMAN_URDU_MARKERS if " " not in marker}
    matches = len(tokens & single_word_markers)
    matches += sum(
        1 for marker in ROMAN_URDU_MARKERS if " " in marker and marker in normalized
    )
    return matches >= 3


def _unavailable(reason: str) -> str:
    return json.dumps(
        {
            "status": "model_unavailable",
            "action": "refer_to_clinician",
            "reason": reason,
            "urgency": "urgent",
        }
    )


def _response_content(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("OpenRouter response contained no choices")

    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ValueError("OpenRouter response contained no assistant message")

    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("OpenRouter returned an empty response")

    try:
        assessment = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("OpenRouter returned invalid JSON") from exc

    required_fields = {
        "likely_condition",
        "differential",
        "recommendation",
        "watch_for",
        "confidence",
        "urgency",
    }
    if not isinstance(assessment, dict) or not required_fields.issubset(assessment):
        raise ValueError("OpenRouter response did not match the triage schema")
    if not str(assessment.get("likely_condition", "")).strip():
        raise ValueError("OpenRouter response omitted the likely condition")
    return content


def infer_text(symptoms_and_context: str) -> str:
    """Run medical decision-support inference through OpenRouter.

    Input is patient context plus optional retrieved evidence. Output is a JSON
    assessment string, or a machine-readable unavailable status for the safe
    fallback.
    """
    if not settings.openrouter_api_key:
        return _unavailable("MEDROUTE_OPENROUTER_API_KEY is not set")

    language_instruction = ""
    if _looks_like_roman_urdu(symptoms_and_context):
        language_instruction = (
            "\n\nOutput language requirement: Write the clinical assessment, "
            "recommendation, and warning signs in clear professional English. "
            "(The patient's own words are recorded separately for the clinician.)"
        )

    request_body = {
        "model": settings.openrouter_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Patient: {symptoms_and_context}{language_instruction}",
            },
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "medical_triage_assessment",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "likely_condition": {"type": "string"},
                        "differential": {
                            "type": "array",
                            "items": {"type": "string"},
                            "maxItems": 5,
                        },
                        "recommendation": {"type": "string"},
                        "watch_for": {
                            "type": "array",
                            "items": {"type": "string"},
                            "maxItems": 6,
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                        },
                        "urgency": {
                            "type": "string",
                            "enum": ["emergency", "urgent", "soon", "routine"],
                        },
                    },
                    "required": [
                        "likely_condition",
                        "differential",
                        "recommendation",
                        "watch_for",
                        "confidence",
                        "urgency",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        "max_tokens": 800,
        "temperature": 0.2,
    }
    headers = {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "Content-Type": "application/json",
        "X-OpenRouter-Title": "MedRoute",
    }
    if settings.openrouter_site_url:
        headers["HTTP-Referer"] = settings.openrouter_site_url

    last_error = "OpenRouter request failed"
    for attempt in range(1, max(settings.openrouter_max_attempts, 1) + 1):
        try:
            response = requests.post(
                f"{settings.openrouter_base_url.rstrip('/')}/chat/completions",
                headers=headers,
                json=request_body,
                timeout=settings.openrouter_timeout,
            )
            response.raise_for_status()
            return _response_content(response.json())
        except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
            last_error = str(exc)
            log.warning(
                "OpenRouter inference attempt %d/%d failed: %s",
                attempt,
                settings.openrouter_max_attempts,
                exc,
            )

    return _unavailable(last_error)


@tool
def openrouter_infer(symptoms_and_context: str) -> str:
    """LangChain tool wrapper — delegates to :func:`infer_text`."""
    return infer_text(symptoms_and_context)
