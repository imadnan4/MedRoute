"""Stage 0 — Nemotron 3.5 ASR transcription.

Priority (``MEDROUTE_ASR_MODE``):
  - ``local``  — ONNX INT4 via onnxruntime-genai (CPU, offline after download)
  - ``remote`` — HTTP ASR server (``MEDROUTE_ASR_SERVER_URL``)
  - ``auto``   — local if model is available, else remote; on failure, try the other

Typed text uses ``transcribe_text()`` (no ASR).
"""
from __future__ import annotations

import base64
import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

import requests

from config import settings

log = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = os.path.expanduser("~/.cache/medroute/nemotron-asr")


@dataclass
class Transcript:
    text: str
    language: str
    latency_ms: int
    source: str = "passthrough"  # local | remote | passthrough


def _local_model_ready(model_dir: str = DEFAULT_MODEL_DIR) -> bool:
    return os.path.isfile(os.path.join(model_dir, "genai_config.json")) and os.path.isfile(
        os.path.join(model_dir, "encoder.onnx.data")
    )


class Transcriber:
    def __init__(self) -> None:
        self._local_error: Optional[str] = None

    def _mode(self) -> str:
        mode = (getattr(settings, "asr_mode", None) or "auto").strip().lower()
        if mode not in {"auto", "local", "remote"}:
            return "auto"
        return mode

    def _transcribe_remote(
        self,
        audio_bytes: bytes,
        sample_rate: int,
        language: str,
    ) -> Transcript:
        if not settings.asr_server_url:
            raise RuntimeError(
                "Remote ASR not configured. Set MEDROUTE_ASR_SERVER_URL "
                "(e.g. http://host:8080) or use local Nemotron (MEDROUTE_ASR_MODE=local)."
            )

        audio_b64 = base64.b64encode(audio_bytes).decode()
        payload = {
            "audio": audio_b64,
            "sample_rate": sample_rate,
            "target_lang": language,
        }
        t0 = time.perf_counter()
        resp = requests.post(
            f"{settings.asr_server_url.rstrip('/')}/transcribe",
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        latency = data.get("latency_ms")
        if latency is None:
            latency = int((time.perf_counter() - t0) * 1000)
        return Transcript(
            text=(data.get("text") or "").strip(),
            language=data.get("language", language or "auto"),
            latency_ms=int(latency),
            source="remote",
        )

    def _transcribe_local(self, audio_bytes: bytes, language: str) -> Transcript:
        from voice.nemotron_local import get_local_asr
        from voice.roman_urdu import prefer_clinic_transcript

        t0 = time.perf_counter()
        asr = get_local_asr()
        text, detected = asr.transcribe(audio_bytes, language=language)
        latency = int((time.perf_counter() - t0) * 1000)
        if not text:
            # Soft-fail: 200 with empty text is easier for the UI than a 503
            # that looks like a server outage when the model simply heard no speech.
            log.warning(
                "Local ASR returned empty transcript after language fallbacks "
                "(%.1fs audio, %d bytes). Speak closer / louder, or use typed text.",
                len(audio_bytes) / 16000.0,  # rough; real duration logged in nemotron_local
                len(audio_bytes),
            )
            return Transcript(
                text="",
                language=detected or language or "auto",
                latency_ms=latency,
                source="local",
            )
        # Urdu speech via Hindi encoder → Devanagari; show Roman Urdu in UI
        text = prefer_clinic_transcript(text, language=detected or language)
        return Transcript(
            text=text,
            language=detected or language or "auto",
            latency_ms=latency,
            source="local",
        )

    def transcribe_bytes(
        self,
        audio_bytes: bytes,
        sample_rate: int = 16000,
        language: str = "auto",
    ) -> Transcript:
        """Transcribe audio with local Nemotron and/or remote server."""
        if not audio_bytes or len(audio_bytes) < 64:
            raise ValueError("Audio payload is empty or too short to transcribe.")

        mode = self._mode()
        local_ok = _local_model_ready()
        remote_ok = bool(settings.asr_server_url)
        errors: list[str] = []

        # Build attempt order
        order: list[str] = []
        if mode == "local":
            order = ["local"]
        elif mode == "remote":
            order = ["remote"]
        else:  # auto: prefer local when model is on disk (no extra server required)
            if local_ok:
                order.append("local")
            if remote_ok:
                order.append("remote")
            if not order:
                # last resort: try local (will download) then fail clearly
                order = ["local"]

        for backend in order:
            try:
                if backend == "local":
                    log.info("ASR: using local Nemotron ONNX")
                    return self._transcribe_local(audio_bytes, language=language)
                log.info("ASR: using remote server %s", settings.asr_server_url)
                return self._transcribe_remote(audio_bytes, sample_rate, language)
            except Exception as exc:
                msg = f"{backend}: {exc}"
                errors.append(msg)
                log.warning("ASR %s failed: %s", backend, exc)

        detail = "; ".join(errors) if errors else "no ASR backend available"
        raise RuntimeError(
            f"ASR unavailable ({detail}). "
            "Install local model: `python scripts/download_asr_model.py`, "
            "or set a reachable MEDROUTE_ASR_SERVER_URL, "
            "or use typed text input."
        )

    def transcribe_text(self, text: str, language: str = "hi-IN") -> Transcript:
        """Passthrough for typed text — no ASR needed."""
        return Transcript(text=text, language=language, latency_ms=0, source="passthrough")


transcriber = Transcriber()
