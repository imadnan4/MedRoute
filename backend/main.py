"""MedRoute — FastAPI Application.

Exposes the full triage pipeline as REST + SSE endpoints. One POST endpoint
runs the complete pipeline synchronously; a GET SSE endpoint streams progress
updates for a real-time UX.
"""
from __future__ import annotations

import base64
import json
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agents.triage_agent import run_triage
from config import settings
from models import TriageResult, TriageRoute
from pipeline.complexity_scorer import score_complexity
from pipeline.input_parser import parse as parse_input
from pipeline.report_generator import generate_pdf
from rag.retriever import get_or_create_collection
from safety.red_flag_checker import check_red_flags
from voice.transcriber import Transcript, transcriber

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ------------------------------------------------------------------ #
# In-memory store (hackathon — no DB)
# ------------------------------------------------------------------ #
_triage_store: dict[str, dict] = {}


# ------------------------------------------------------------------ #
# Request / Response schemas
# ------------------------------------------------------------------ #
class TriageRequest(BaseModel):
    transcript: str = Field(..., description="Patient voice transcript or typed input")
    language: str = Field("hi-IN", description="Language locale tag")
    audio_b64: Optional[str] = Field(None, description="Base64-encoded audio bytes (optional)")
    age_years: Optional[float] = Field(None, description="Patient age in years (optional if in transcript)")
    age_months: Optional[float] = Field(None, description="Patient age in months (optional if in transcript)")
    pregnancy: Optional[str] = Field(
        None,
        description="Pregnancy status: not_pregnant, pregnant, pregnant_3rd_trimester",
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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _apply_context_overrides(transcript_text: str, language: str, req: TriageRequest) -> Transcript:
    """Merge explicit age/pregnancy fields into the transcript for the parser."""
    if req.age_years is None and req.age_months is None and req.pregnancy is None:
        return Transcript(text=transcript_text, language=language, latency_ms=0)

    user_text = transcript_text
    if req.age_years is not None:
        user_text += f" Age: {req.age_years} years"
    if req.age_months is not None:
        user_text += f" Age: {req.age_months} months"
    if req.pregnancy:
        user_text += f" Pregnancy: {req.pregnancy}"
    return Transcript(text=user_text, language=language, latency_ms=0)


# ------------------------------------------------------------------ #
# Pipeline runner
# ------------------------------------------------------------------ #
def run_pipeline(request: TriageRequest, case_id: str) -> dict:
    """Execute the full triage pipeline synchronously."""
    stages: dict = {}

    # Stage 0 — Voice / Input
    if request.audio_b64:
        try:
            audio_bytes = base64.b64decode(request.audio_b64)
            transcript = transcriber.transcribe_bytes(audio_bytes)
            stages["asr"] = {
                "status": "completed",
                "text": transcript.text,
                "language": transcript.language,
                "latency_ms": transcript.latency_ms,
            }
            log.info("ASR transcribed %d chars in %dms", len(transcript.text), transcript.latency_ms)
        except Exception as exc:
            log.warning("ASR failed, falling back to text input: %s", exc)
            transcript = transcriber.transcribe_text(request.transcript, request.language)
            stages["asr"] = {
                "status": "completed",
                "text": transcript.text,
                "language": transcript.language,
                "fallback": True,
            }
    else:
        transcript = transcriber.transcribe_text(request.transcript, request.language)
        stages["asr"] = {
            "status": "completed",
            "text": transcript.text,
            "language": transcript.language,
        }

    transcript = _apply_context_overrides(transcript.text, request.language, request)

    # Stage 1 — Input Parser
    parsed = parse_input(transcript)
    stages["parser"] = {
        "status": "completed",
        "symptoms": parsed.symptoms,
        "clusters": parsed.symptom_clusters,
        "age": parsed.patient.age_for_display,
        "pregnancy": parsed.patient.pregnancy.value,
        "duration_days": parsed.patient.duration_days,
    }

    # Stage 2 — Safety Pre-Check (hard override before any LLM)
    red_flag = check_red_flags(parsed)
    stages["safety"] = {"status": "completed", "triggered": red_flag.triggered}
    if red_flag.triggered:
        stages["safety"]["flag_class"] = red_flag.flag_class
        stages["safety"]["message"] = red_flag.message
        stages["safety"]["matched"] = red_flag.matched_symptoms

    # Stage 3 — Complexity Scorer
    score = score_complexity(parsed)
    stages["scorer"] = {
        "status": "completed",
        "raw_score": score.raw_score,
        "adjusted_score": score.adjusted_score,
        "confidence": score.confidence,
        "route": score.route.value,
        "syndrome_hits": score.syndrome_hits,
        "reasoning": score.reasoning,
    }

    if red_flag.triggered:
        score.route = TriageRoute.HARD_ESCALATION

    # Stage 4 — Deterministic triage orchestrator
    result = run_triage(parsed, score, red_flag)
    stages["agent"] = {
        "status": "completed",
        "route": result.route.value,
        "likely_condition": result.likely_condition,
        "urgency": result.urgency.value,
        "cascade": result.cascade_used,
        "confidence": result.confidence,
        "confidence_level": result.confidence_level.value,
    }

    store_entry = {
        "case_id": case_id,
        "request": request.model_dump(),
        "stages": stages,
        "result": result.model_dump(),
        "pdf_bytes": None,
    }
    _triage_store[case_id] = store_entry
    return store_entry


# ------------------------------------------------------------------ #
# REST Endpoints
# ------------------------------------------------------------------ #
@app.post("/triage")
async def triage_post(req: TriageRequest):
    """Run the full triage pipeline and return structured result."""
    case_id = str(uuid.uuid4())
    try:
        entry = run_pipeline(req, case_id)
        result_obj = TriageResult(**entry["result"])
        pdf_bytes = generate_pdf(result_obj)
        _triage_store[case_id]["pdf_bytes"] = pdf_bytes
        entry.pop("pdf_bytes", None)
        return JSONResponse(content=entry, status_code=200)
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

        def _emit(stage: str, status: str, data: dict | None = None):
            ev = PipelineEvent(stage=stage, status=status, data=data or {})
            return f"data: {ev.model_dump_json()}\n\n".encode()

        try:
            yield _emit("asr", "running")
            t = transcriber.transcribe_text(transcript, language)
            yield _emit("asr", "completed", {"text": t.text, "language": t.language})

            t = _apply_context_overrides(t.text, language, req)

            yield _emit("parser", "running")
            parsed = parse_input(t)
            yield _emit(
                "parser",
                "completed",
                {
                    "symptoms": parsed.symptoms,
                    "clusters": parsed.symptom_clusters,
                    "age": parsed.patient.age_for_display,
                },
            )

            yield _emit("safety", "running")
            red_flag = check_red_flags(parsed)
            yield _emit(
                "safety",
                "completed",
                {
                    "triggered": red_flag.triggered,
                    "flag_class": red_flag.flag_class,
                    "matched": red_flag.matched_symptoms,
                },
            )

            yield _emit("scorer", "running")
            score = score_complexity(parsed)
            yield _emit(
                "scorer",
                "completed",
                {
                    "score": score.adjusted_score,
                    "confidence": score.confidence,
                    "route": score.route.value,
                    "syndrome_hits": score.syndrome_hits,
                },
            )

            if red_flag.triggered:
                score.route = TriageRoute.HARD_ESCALATION

            yield _emit("agent", "running")
            result = run_triage(parsed, score, red_flag)
            yield _emit(
                "agent",
                "completed",
                {
                    "route": result.route.value,
                    "condition": result.likely_condition,
                    "urgency": result.urgency.value,
                    "cascade": result.cascade_used,
                },
            )

            # Persist for PDF download
            _triage_store[case_id] = {
                "case_id": case_id,
                "request": req.model_dump(),
                "stages": {},
                "result": result.model_dump(),
                "pdf_bytes": None,
            }

            yield _emit("done", "completed", {"case_id": case_id, "result": result.model_dump()})

        except Exception as exc:
            yield _emit("error", "error", {"message": str(exc)})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/report/{case_id}")
async def get_report(case_id: str):
    """Download a PDF report for a previously run triage."""
    entry = _triage_store.get(case_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Case not found")

    pdf = entry.get("pdf_bytes")
    if pdf is None:
        result_obj = TriageResult(**entry["result"])
        pdf = generate_pdf(result_obj)
        entry["pdf_bytes"] = pdf

    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=medroute_triage_{case_id[:8]}.pdf"},
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
async def transcribe_audio(req: TranscribeRequest):
    """Transcribe audio via local Nemotron ONNX (preferred) or remote ASR server."""
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
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
