"""Durable encounter storage backed by Neon Postgres."""

from __future__ import annotations

import base64
import json
from typing import Any, Protocol

from config import settings


class EncounterStore(Protocol):
    def save(self, entry: dict[str, Any], user_id: str) -> None: ...

    def get(self, case_id: str, user_id: str) -> dict[str, Any] | None: ...


def _serializable_entry(entry: dict[str, Any]) -> tuple[dict[str, Any], bytes | None]:
    request = dict(entry.get("request") or {})
    audio_b64 = request.pop("audio_b64", None)
    audio_bytes = None
    if audio_b64:
        try:
            audio_bytes = base64.b64decode(audio_b64, validate=True)
        except (ValueError, TypeError):
            audio_bytes = None
    public_entry = {key: value for key, value in entry.items() if key != "pdf_bytes"}
    public_entry["request"] = request
    return public_entry, audio_bytes


def _json_object(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) else value


class InMemoryEncounterStore:
    """Test/local fallback used only when no Neon URL is configured."""

    def __init__(self) -> None:
        self._entries: dict[tuple[str, str], dict[str, Any]] = {}

    def save(self, entry: dict[str, Any], user_id: str) -> None:
        public_entry, _ = _serializable_entry(entry)
        stored = dict(public_entry)
        stored["pdf_bytes"] = entry.get("pdf_bytes")
        self._entries[(user_id, str(entry["case_id"]))] = stored

    def get(self, case_id: str, user_id: str) -> dict[str, Any] | None:
        return self._entries.get((user_id, case_id))


class NeonEncounterStore:
    """Small SQL adapter; migrations live in ``migrations/``."""

    def __init__(self, database_url: str) -> None:
        self.database_url = database_url

    def _connect(self):
        try:
            import psycopg
        except ImportError as exc:
            raise RuntimeError("psycopg is required when Neon storage is enabled") from exc
        return psycopg.connect(self.database_url)

    def save(self, entry: dict[str, Any], user_id: str) -> None:
        public_entry, audio_bytes = _serializable_entry(entry)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO medroute_encounters
                    (case_id, user_id, mode, request, stages, result, audio_bytes, pdf_bytes)
                VALUES (%s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s, %s)
                ON CONFLICT (case_id) DO UPDATE SET
                    user_id = EXCLUDED.user_id,
                    mode = EXCLUDED.mode,
                    request = EXCLUDED.request,
                    stages = EXCLUDED.stages,
                    result = EXCLUDED.result,
                    audio_bytes = EXCLUDED.audio_bytes,
                    pdf_bytes = EXCLUDED.pdf_bytes,
                    updated_at = now()
                """,
                (
                    str(entry["case_id"]), user_id, str(public_entry.get("mode", "triage")),
                    json.dumps(public_entry.get("request", {})),
                    json.dumps(public_entry.get("stages", {})),
                    json.dumps(public_entry.get("result", {})), audio_bytes,
                    entry.get("pdf_bytes"),
                ),
            )

    def get(self, case_id: str, user_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT case_id, mode, request, stages, result, pdf_bytes
                FROM medroute_encounters
                WHERE case_id = %s AND user_id = %s
                """, (case_id, user_id),
            ).fetchone()
        if row is None:
            return None
        return {
            "case_id": str(row[0]), "mode": row[1], "request": _json_object(row[2]),
            "stages": _json_object(row[3]), "result": _json_object(row[4]),
            "pdf_bytes": bytes(row[5]) if row[5] is not None else None,
        }


def create_encounter_store() -> EncounterStore:
    if settings.database_url:
        return NeonEncounterStore(settings.database_url)
    return InMemoryEncounterStore()
