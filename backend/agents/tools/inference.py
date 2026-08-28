"""Inference seam — a swappable interface between the triage agent and the
model backend.

The concrete ``OpenRouterInfer`` reproduces the existing OpenRouter behaviour
exactly; ``InMemoryInfer`` lets the cascade be exercised in tests with no
network. Both satisfy ``InferenceAdapter`` so callers (``run_triage``) depend
only on the small interface, not on the network tool.
"""

from __future__ import annotations

import json
from typing import Protocol, runtime_checkable


@runtime_checkable
class InferenceAdapter(Protocol):
    """One method: turn a dict payload into a triage JSON-shaped dict."""

    def infer(self, payload: dict) -> dict:
        """Return the triage assessment dict (or ``{"status": "unavailable"}``)."""
        ...


class OpenRouterInfer:
    """Adapter that drives the real OpenRouter network tool."""

    def infer(self, payload: dict) -> dict:
        from agents.tools.openrouter_infer import infer_text

        raw = infer_text(payload.get("context", ""))
        if isinstance(raw, dict):
            return raw
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {"status": "error", "reason": "non_json_inference_output"}

    def infer_intake(self, transcript: str, domain: str = "home_health") -> dict:
        from agents.tools.openrouter_infer import infer_intake

        return infer_intake(transcript, domain)


class InMemoryInfer:
    """Test double returning a canned, realistic triage dict (no network)."""

    def __init__(self, canned: dict | None = None) -> None:
        self.canned = canned or {
            "likely_condition": "Viral upper respiratory infection",
            "differential": [
                "Acute bronchitis",
                "Allergic rhinitis",
                "Influenza",
            ],
            "recommendation": (
                "Supportive care: fluids, rest, antipyretics. "
                "Return if dyspnea or persistent high fever."
            ),
            "watch_for": ["shortness of breath", "chest pain", "fever > 3 days"],
            "confidence": 0.82,
            "urgency": "routine",
        }
        self.last_payload: dict | None = None

    def infer(self, payload: dict) -> dict:
        self.last_payload = payload
        return dict(self.canned)

    def infer_intake(self, transcript: str, domain: str = "home_health") -> dict:
        return {
            "contact_name": "Aisha",
            "care_recipient_name": "Mrs. Khan",
            "phone_or_contact": "03001234567",
            "care_need_summary": "needs wound dressing at home twice a week",
            "condition_or_issue": "leg ulcer",
            "mobility_or_severity_notes": "limited mobility, uses walker",
            "insurance_or_payment_notes": "private insurance",
            "preferred_availability": "weekday mornings",
            "free_text_summary": "Daughter calling about her mother who needs home wound care.",
        }
