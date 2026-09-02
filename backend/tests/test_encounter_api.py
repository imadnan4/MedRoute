"""Offline tests for the unified /encounter API (intake + triage).

Network is never touched: a fake ``InferenceAdapter`` is injected into the
global pipeline before each call, mirroring ``InMemoryInfer``. Asserts the new
``/encounter`` route exposes ``disposition``/``mode`` for intake, maps triage
``TriageRoute`` onto the unified ``Disposition`` vocabulary plus ``acuity``, and
that the legacy ``/triage`` shape is unchanged.
"""

from __future__ import annotations

from agents import triage_agent
from agents.tools.inference import InMemoryInfer
from fastapi.testclient import TestClient
import pytest

import main
from auth import AuthenticatedUser, get_current_user
from models import Acuity, Disposition, TriageRoute
from pipeline.triage_pipeline import _disposition_from_route

CANNED_INTAKE = {
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


@pytest.fixture
def client(monkeypatch):
    # Triage mode builds its inference adapter from run_triage's default
    # (OpenRouterInfer, baked as a default arg); redirect the call to an offline
    # fake so no network is hit.
    real_run_triage = triage_agent.run_triage

    def _fake_run_triage(parsed, score, red_flag):
        return real_run_triage(parsed, score, red_flag, inference=InMemoryInfer())

    monkeypatch.setattr(main._pipeline, "_run_triage", _fake_run_triage)
    # Intake mode reads the adapter injected into the pipeline.
    main._pipeline._inference = InMemoryInfer(CANNED_INTAKE)
    main.app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser(
        user_id="offline-test-user", claims={"sub": "offline-test-user"}
    )
    client = TestClient(main.app)
    yield client
    main.app.dependency_overrides.pop(get_current_user, None)


def test_encounter_intake_returns_disposition_and_mode(client) -> None:
    resp = client.post(
        "/encounter",
        json={
            "transcript": "My mother needs wound care at home, she has a leg ulcer.",
            "mode": "intake",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "intake"
    assert body["disposition"] == Disposition.STANDARD_QUEUE.value


def test_encounter_triage_maps_route_to_disposition_and_acuity(client) -> None:
    resp = client.post(
        "/encounter",
        json={"transcript": "I have a fever and a cough", "mode": "triage"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "triage"
    result = body["result"]
    assert result["route"] == TriageRoute.LOCAL_ONLY.value
    assert result["disposition"] == Disposition.STANDARD_QUEUE.value
    assert result["acuity"] == Acuity.ROUTINE.value


def test_encounter_triage_red_flag_escalates(client) -> None:
    resp = client.post(
        "/encounter",
        json={
            "transcript": "He has chest pain and shortness of breath",
            "mode": "triage",
        },
    )
    assert resp.status_code == 200
    result = resp.json()["result"]
    assert result["route"] == TriageRoute.HARD_ESCALATION.value
    assert result["disposition"] == Disposition.ESCALATE_TO_CLINICIAN.value


def test_legacy_triage_shape_unchanged(client) -> None:
    resp = client.post(
        "/triage",
        json={"transcript": "I have a fever and a cough"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) >= {"case_id", "request", "stages", "result"}
    assert body["result"]["route"] == TriageRoute.LOCAL_ONLY.value


def test_encounter_stream_returns_events(client) -> None:
    resp = client.get(
        "/encounter/stream",
        params={"transcript": "I have a fever and a cough", "mode": "triage"},
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    assert "data:" in resp.text


def test_disposition_mapping_table() -> None:
    assert (
        _disposition_from_route(TriageRoute.ESCALATION_BIAS, False)
        == Disposition.ESCALATE_TO_CLINICIAN
    )
    assert (
        _disposition_from_route(TriageRoute.HARD_ESCALATION, True)
        == Disposition.ESCALATE_TO_CLINICIAN
    )
    assert (
        _disposition_from_route(TriageRoute.REMOTE, False)
        == Disposition.PROVIDE_GUIDANCE
    )
    assert (
        _disposition_from_route(TriageRoute.LOCAL_WITH_RAG, False)
        == Disposition.PROVIDE_GUIDANCE
    )
    assert (
        _disposition_from_route(TriageRoute.LOCAL_ONLY, False)
        == Disposition.STANDARD_QUEUE
    )
