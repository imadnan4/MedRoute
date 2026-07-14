"""LangChain tool: local inference via Hippo-Mistral-7B served via Ollama."""
from __future__ import annotations

import json
import logging

import requests
from langchain_core.tools import tool

from config import settings

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a medical triage assistant for low-resource primary care clinics.
Your role is DECISION SUPPORT only — not a final diagnosis.

Given patient symptoms (may include retrieved guideline evidence), return ONLY valid JSON:
{
  "likely_condition": "string — most likely working assessment",
  "differential": ["string", "string", "string"],
  "recommendation": "string — concrete next steps for the clinic",
  "watch_for": ["string", "string"],
  "confidence": 0.0-1.0,
  "urgency": "emergency|urgent|soon|routine"
}

Rules:
- Prefer common conditions in primary care when evidence is weak.
- If information is insufficient, set confidence < 0.5 and urgency "urgent".
- Never invent vital signs that were not provided.
- Keep recommendation actionable and short.
- No text outside the JSON object.
"""


@tool
def local_infer(symptoms_and_context: str) -> str:
    """Run local medical inference via Hippo-Mistral-7B.
    Input: string describing symptoms, age, pregnancy status, optional evidence.
    Returns: JSON with likely_condition, differential, recommendation, watch_for, confidence, urgency.
    """
    payload = {
        "model": settings.local_llm_ollama_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Patient: {symptoms_and_context}"},
        ],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.2},
    }

    try:
        resp = requests.post(
            f"{settings.ollama_base_url}/api/chat",
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]
    except Exception as e:
        log.error("Ollama local inference error: %s", e)
        return json.dumps({
            "status": "local_unavailable",
            "reason": str(e),
            "action": "fallback_to_remote",
        })
