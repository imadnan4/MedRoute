"""MedRoute — FastAPI Application.

Exposes the full triage pipeline as REST + SSE endpoints. One POST endpoint
runs the complete pipeline synchronously; a GET SSE endpoint streams progress
updates for a real-time UX.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, Literal, Optional

from agents.triage_agent import run_triage
from agents.tools.inference import OpenRouterInfer
from config import settings

# Propagate HuggingFace token to env so all downstream libs (sentence-transformers,
# huggingface_hub, chromadb embedding functions) can use it for model downloads.
if settings.hf_token and not os.environ.get("HF_TOKEN"):
    os.environ["HF_TOKEN"] = settings.hf_token
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from models import TriageResult
from pipeline.complexity_scorer import score_complexity
from pipeline.input_parser import parse as parse_input
from pipeline.report_generator import generate_pdf
from pipeline.triage_pipeline import TriagePipeline
from pydantic import BaseModel, Field
from rag.retriever import get_or_create_collection
from safety.red_flag_checker import check_red_flags
from safety.distress_checker import check_distress
from voice.transcriber import Transcript, transcriber
from auth import AuthenticatedUser, get_current_user
from storage import EncounterStore, create_encounter_store

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ------------------------------------------------------------------ #
# Durable Neon store, with an in-memory fallback for offline tests.
# ------------------------------------------------------------------ #
_encounter_store: EncounterStore = create_encounter_store()


# ------------------------------------------------------------------ #
# Request / Response schemas
# ------------------------------------------------------------------ #
class TriageRequest(BaseModel):
    transcript: str = Field(..., description="Patient voice transcript or typed input")
    language: str = Field("hi-IN", description="Language locale tag")
    audio_b64: Optional[str] = Field(
        None, description="Base64-encoded audio bytes (optional)"
    )
    age_years: Optional[float] = Field(
        None, description="Patient age in years (optional if in transcript)"
    )
    age_months: Optional[float] = Field(
        None, description="Patient age in months (optional if in transcript)"
    )
    pregnancy: Optional[str] = Field(
        None,
        description="Pregnancy status: not_pregnant, pregnant, pregnant_3rd_trimester",
    )


class EncounterRequest(TriageRequest):
    mode: str = Field(
        "triage", description="First-contact mode: intake | triage"
    )
    domain: Literal["home_health", "legal"] = Field(
        "home_health", description="Intake variant: home_health | legal"
    )


class PipelineEvent(BaseModel):
    stage: str
    status: str  # running | completed | error
    data: dict = Field(default_factory=dict)


# ------------------------------------------------------------------ #
# Lifespan — warm RAG on startup
# ------------------------------------------------------------------ #
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("MedRoute starting up...")
    try:
        coll = get_or_create_collection()
        log.info("ChromaDB ready: %d docs", coll.count())
    except Exception as exc:
        log.warning("ChromaDB init failed (seed RAG still available): %s", exc)
    yield
    log.info("MedRoute shutting down.")


# ------------------------------------------------------------------ #
# App
# ------------------------------------------------------------------ #
app = FastAPI(
    title="MedRoute — Medical Triage & Routing Agent",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


_pipeline = TriagePipeline(
    transcriber=transcriber,
    parse_input=parse_input,
    check_red_flags=check_red_flags,
    check_distress=check_distress,
    score_complexity=score_complexity,
    run_triage=run_triage,
    inference=OpenRouterInfer(),
)


def _emit(stage: str, status: str, data: dict | None = None):
    ev = PipelineEvent(stage=stage, status=status, data=data or {})
    return f"data: {ev.model_dump_json()}\n\n".encode()


def _sse_on_stage(events: list[bytes], name: str, status: str, payload: dict) -> None:
    """Translate a pipeline stage transition into the original SSE event shape."""
    data: dict = payload
    if status == "completed":
        if name == "asr":
            data = {"text": payload.get("text"), "language": payload.get("language")}
        elif name == "parser":
            data = {
                "symptoms": payload.get("symptoms"),
                "clusters": payload.get("clusters"),
                "age": payload.get("age"),
            }
        elif name == "safety":
            data = {
                "triggered": payload.get("triggered"),
                "flag_class": payload.get("flag_class"),
                "matched": payload.get("matched"),
            }
        elif name == "scorer":
            data = {
                "score": payload.get("adjusted_score"),
                "confidence": payload.get("confidence"),
                "route": payload.get("route"),
                "syndrome_hits": payload.get("syndrome_hits"),
            }
        elif name == "agent":
            data = {
                "route": payload.get("route"),
                "condition": payload.get("likely_condition"),
                "urgency": payload.get("urgency"),
                "cascade": payload.get("cascade"),
            }
        elif name == "done":
            data = payload
    events.append(_emit(name, status, data))


def _encounter_to_dict(encounter: "Encounter") -> dict:
    """Serialize an intake-mode ``Encounter`` dataclass to a JSON-safe dict."""
    return {
        "mode": encounter.mode,
        "input_mode": encounter.input_mode,
        "input_language": encounter.input_language,
        "raw_transcript": encounter.raw_transcript,
        "red_flag": encounter.red_flag.model_dump() if encounter.red_flag else None,
        "distress": encounter.distress.to_dict() if encounter.distress else None,
        "acuity": encounter.acuity.value if encounter.acuity else None,
        "disposition": encounter.disposition.value,
        "extracted": encounter.extracted,
        "case_id": encounter.case_id,
        "needs_human_review": encounter.needs_human_review,
        "domain": encounter.domain,
    }


def _build_triage_store(req: object, case_id: str, on_stage=None) -> dict:
    """Run a triage-mode encounter and assemble the persisted store entry.

    Shared by ``/triage`` and ``/encounter`` (triage mode) so the two routes stay
    behaviour-identical. The returned dict's ``result`` now carries the unified
    ``acuity`` and ``disposition`` fields.
    """
    outcome = _pipeline.run(req, case_id, mode="triage", on_stage=on_stage)
    store_entry = {
        "case_id": outcome.case_id,
        "request": outcome.request,
        "stages": outcome.stages,
        "result": outcome.result.model_dump(),
        "pdf_bytes": None,
    }
    result_obj = TriageResult(**store_entry["result"])
    pdf_bytes = generate_pdf(result_obj)
    store_entry["pdf_bytes"] = pdf_bytes
    return store_entry


def _public_store_entry(entry: dict) -> dict:
    public_entry = {key: value for key, value in entry.items() if key != "pdf_bytes"}
    if isinstance(public_entry.get("request"), dict):
        public_entry["request"] = {
            key: value for key, value in public_entry["request"].items() if key != "audio_b64"
        }
    return public_entry


def _save_entry(entry: dict, user: AuthenticatedUser) -> None:
    _encounter_store.save(entry, user.user_id)


# ------------------------------------------------------------------ #
# REST Endpoints
# ------------------------------------------------------------------ #
@app.post("/triage")
async def triage_post(req: TriageRequest, user: AuthenticatedUser = Depends(get_current_user)):
    """Run the full triage pipeline and return structured result."""
    case_id = str(uuid.uuid4())
    try:
        store_entry = _build_triage_store(req, case_id)
        _save_entry(store_entry, user)
        return JSONResponse(content=_public_store_entry(store_entry), status_code=200)
    except Exception as exc:
        log.exception("Pipeline error for case %s", case_id)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/encounter")
async def encounter_post(req: EncounterRequest, user: AuthenticatedUser = Depends(get_current_user)):
    """Unified first-contact endpoint: runs intake OR triage by ``mode``.

    Accepts the same request shape as ``/triage`` (text or base64 audio) plus a
    ``mode`` field (``intake`` | ``triage``, default ``triage``). Returns an
    ``Encounter`` for intake, or the triage result (now carrying ``acuity`` and
    ``disposition``) for triage.
    """
    case_id = str(uuid.uuid4())
    try:
        if req.mode == "intake":
            encounter = _pipeline.run(req, case_id, mode="intake", domain=req.domain)
            body = _encounter_to_dict(encounter)
            store_entry = {
                "case_id": case_id,
                "mode": "intake",
                "request": req.model_dump(),
                "stages": {},
                "result": body,
                "pdf_bytes": None,
            }
            _save_entry(store_entry, user)
            return JSONResponse(content=body, status_code=200)

        store_entry = _build_triage_store(req, case_id)
        store_entry["mode"] = "triage"
        _save_entry(store_entry, user)
        return JSONResponse(content=_public_store_entry(store_entry), status_code=200)
    except Exception as exc:
        log.exception("Pipeline error for case %s", case_id)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/triage/stream")
async def triage_stream(
    transcript: str = Query(..., description="Patient transcript"),
    language: str = Query("hi-IN"),
    age_years: Optional[float] = Query(None),
    age_months: Optional[float] = Query(None),
    pregnancy: Optional[str] = Query(None),
    user: AuthenticatedUser = Depends(get_current_user),
):
    """SSE endpoint that streams pipeline stage events as they complete."""

    async def event_stream() -> AsyncGenerator[bytes, None]:
        req = TriageRequest(
            transcript=transcript,
            language=language,
            age_years=age_years,
            age_months=age_months,
            pregnancy=pregnancy,
        )
        case_id = str(uuid.uuid4())

        events: list[bytes] = []

        def _on_stage(name: str, status: str, payload: dict) -> None:
            _sse_on_stage(events, name, status, payload)

        try:
            outcome = _pipeline.run(req, case_id, on_stage=_on_stage)

            # Persist for PDF download (SSE path historically stored no stages)
            store_entry = {
                "case_id": case_id,
                "request": req.model_dump(),
                "stages": {},
                "result": outcome.result.model_dump(),
                "pdf_bytes": None,
            }
            _save_entry(store_entry, user)
        except Exception as exc:
            events.append(_emit("error", "error", {"message": str(exc)}))

        for ev in events:
            yield ev

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/encounter/stream")
async def encounter_stream(
    transcript: str = Query(..., description="Patient transcript"),
    language: str = Query("hi-IN"),
    age_years: Optional[float] = Query(None),
    age_months: Optional[float] = Query(None),
    pregnancy: Optional[str] = Query(None),
    mode: str = Query("triage", description="intake | triage"),
    domain: Literal["home_health", "legal"] = Query(
        "home_health", description="Intake variant: home_health | legal"
    ),
    user: AuthenticatedUser = Depends(get_current_user),
):
    """SSE endpoint mirroring ``/triage/stream`` but with a ``mode`` selector."""

    async def event_stream() -> AsyncGenerator[bytes, None]:
        req = EncounterRequest(
            transcript=transcript,
            language=language,
            age_years=age_years,
            age_months=age_months,
            pregnancy=pregnancy,
            mode=mode,
            domain=domain,
        )
        case_id = str(uuid.uuid4())

        events: list[bytes] = []

        def _on_stage(name: str, status: str, payload: dict) -> None:
            _sse_on_stage(events, name, status, payload)

        try:
            if mode == "intake":
                encounter = _pipeline.run(
                    req, case_id, mode="intake", on_stage=_on_stage, domain=domain
                )
                store_entry = {
                    "case_id": case_id,
                    "request": req.model_dump(),
                    "stages": {},
                    "result": _encounter_to_dict(encounter),
                    "pdf_bytes": None,
                }
                _save_entry(store_entry, user)
            else:
                outcome = _pipeline.run(req, case_id, mode="triage", on_stage=_on_stage)
                store_entry = {
                    "case_id": case_id,
                    "request": req.model_dump(),
                    "stages": {},
                    "result": outcome.result.model_dump(),
                    "pdf_bytes": None,
                }
                _save_entry(store_entry, user)
        except Exception as exc:
            events.append(_emit("error", "error", {"message": str(exc)}))

        for ev in events:
            yield ev

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/report/{case_id}")
async def get_report(case_id: str, user: AuthenticatedUser = Depends(get_current_user)):
    """Download a PDF report for a previously run triage."""
    entry = _encounter_store.get(case_id, user.user_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Case not found")

    pdf = entry.get("pdf_bytes")
    if pdf is None:
        result_obj = TriageResult(**entry["result"])
        pdf = generate_pdf(result_obj, case_id=case_id)
        entry["pdf_bytes"] = pdf

    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"attachment; filename=medroute_triage_{case_id[:8]}.pdf"
        },
    )


class ReportRequest(BaseModel):
    case_id: str = ""
    transcript: str = ""
    generated_at: str = ""
    result: TriageResult


@app.post("/report")
async def post_report(
    body: ReportRequest,
    user: AuthenticatedUser = Depends(get_current_user),
):
    """Generate a PDF report directly from a triage result.

    The frontend already holds the full result, so this avoids any dependency on
    the in-memory case store (which is cleared on every dyno restart/deploy).
    """
    pdf = generate_pdf(
        body.result,
        transcript=body.transcript,
        case_id=body.case_id,
        generated_at=body.generated_at,
    )
    safe_id = (body.case_id or "report").replace("/", "_")[:12]
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"attachment; filename=medroute_triage_{safe_id}.pdf"
        },
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "MedRoute",
        "version": "1.1.0",
        "architecture": "deterministic_orchestrator_v1",
    }


class TranscribeRequest(BaseModel):
    audio_b64: str = Field(..., description="Base64-encoded audio (webm, wav, or mp3)")
    language: str = Field(
        "ur",
        description="Language hint. Clinic default ur; fallback order is Urdu → English only.",
    )


@app.post("/transcribe")
async def transcribe_audio(
    req: TranscribeRequest,
    user: AuthenticatedUser = Depends(get_current_user),
):
    """Transcribe audio via local Whisper (preferred) or a remote ASR server."""
    try:
        audio_bytes = base64.b64decode(req.audio_b64)
        result = transcriber.transcribe_bytes(audio_bytes, language=req.language)
        return {
            "text": result.text,
            "language": result.language,
            "latency_ms": result.latency_ms,
            "source": getattr(result, "source", "unknown"),
        }
    except Exception as exc:
        log.exception("ASR failed")
        raise HTTPException(status_code=503, detail=f"ASR unavailable: {exc}")


# ------------------------------------------------------------------ #
# Static file serving for frontend (if built)
# ------------------------------------------------------------------ #
FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if FRONTEND_DIST.exists():
    app.mount(
        "/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend"
    )
