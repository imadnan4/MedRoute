"""LangChain tool: remote inference via DeepSeek on Fireworks AI."""
from __future__ import annotations

import json
import logging

from langchain_core.tools import tool

from config import settings

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert medical diagnostician supporting primary-care triage
in low-resource settings. Decision support only — clinician remains responsible.

Return ONLY valid JSON:
{
  "likely_condition": "string — most likely working assessment with brief rationale",
  "differential": ["top alternative 1", "top alternative 2", "top alternative 3"],
  "recommendation": "string — concrete next steps (tests, referral timing, self-care if safe)",
  "watch_for": ["red-flag symptom to re-present for"],
  "confidence": 0.0-1.0,
  "urgency": "emergency|urgent|soon|routine"
}

Rules:
- Be clinically precise; name syndromes when appropriate (e.g. B-symptoms).
- Calibrate confidence honestly; incomplete histories must lower confidence.
- If serious disease cannot be excluded, urgency must be urgent or emergency.
- Use any provided guideline evidence; do not contradict clear emergency patterns.
- No text outside the JSON object.
"""


@tool
def remote_infer(symptoms_and_context: str) -> str:
    """Run remote medical inference via DeepSeek on Fireworks AI.
    Input: string describing symptoms, age, pregnancy status, optional evidence.
    Returns: JSON with likely_condition, differential, recommendation, watch_for, confidence, urgency.
    """
    if not settings.fireworks_api_key:
        return json.dumps({
            "status": "remote_unavailable",
            "action": "refer_to_clinician",
            "reason": "MEDROUTE_FIREWORKS_API_KEY not set",
            "urgency": "urgent",
        })

    try:
        from fireworks.client import Fireworks

        client = Fireworks(api_key=settings.fireworks_api_key)
        response = client.chat.completions.create(
            model=settings.remote_llm,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Patient: {symptoms_and_context}"},
            ],
            response_format={"type": "json_object"},
            max_tokens=800,
            temperature=0.2,
            timeout=settings.fireworks_timeout,
        )
        return response.choices[0].message.content
    except Exception as e:
        log.error("Fireworks AI error: %s", e)
        return json.dumps({
            "status": "remote_unavailable",
            "action": "refer_to_clinician",
            "reason": str(e),
            "urgency": "treat_as_high",
        })
