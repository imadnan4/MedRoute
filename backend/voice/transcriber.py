"""Stage 0 — multilingual Whisper ASR transcription.

Priority (``MEDROUTE_ASR_MODE``):
  - ``local``  — Whisper Large V3 Turbo via faster-whisper
  - ``remote`` — HTTP ASR server (``MEDROUTE_ASR_SERVER_URL``)
  - ``auto``   — local if model is available, else remote; on failure, try the other

Typed text uses ``transcribe_text()`` (no ASR).
"""

from __future__ import annotations

import base64
import logging
import time
from dataclasses import dataclass
from typing import Optional

import requests
from config import settings

log = logging.getLogger(__name__)


@dataclass
class Transcript:
    text: str
    language: str
    latency_ms: int
    source: str = "passthrough"  # local | remote | passthrough


def _local_backend_available() -> bool:
    try:
        import faster_whisper  # noqa: F401

        return True
    except ImportError:
        return False


class Transcriber:
    def __init__(self) -> None:
        self._local_error: Optional[str] = None

    def _mode(self) -> str:
        mode = (getattr(settings, "asr_mode", None) or "auto").strip().lower()
        if mode not in {"auto", "local", "remote", "hf"}:
            return "auto"
        return mode

    def _transcribe_hf(
        self,
        audio_bytes: bytes,
        language: str,
    ) -> Transcript:
        """Transcribe via HF Inference Providers — no local GPU/RAM needed.

        Uses serverless inference API at router.huggingface.co/hf-inference.
        Docs: https://huggingface.co/docs/huggingface_hub/guides/inference
        """
        if not settings.hf_token:
            raise RuntimeError(
                "HuggingFace token not set. Add MEDROUTE_HF_TOKEN to .env."
            )

        hf_asr_model = getattr(settings, "hf_asr_model", None) or "openai/whisper-large-v3"
        url = f"https://router.huggingface.co/hf-inference/models/{hf_asr_model}"

        t0 = time.perf_counter()
        resp = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {settings.hf_token}",
                "Content-Type": "audio/wav",
            },
            data=audio_bytes,
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        latency = int((time.perf_counter() - t0) * 1000)
        return Transcript(
            text=(data.get("text") or "").strip(),
            language=language or "auto",
            latency_ms=latency,
            source="hf",
        )

    def _transcribe_remote(
        self,
        audio_bytes: bytes,
        sample_rate: int,
        language: str,
    ) -> Transcript:
        if not settings.asr_server_url:
            raise RuntimeError(
                "Remote ASR not configured. Set MEDROUTE_ASR_SERVER_URL "
                "(e.g. http://host:8080) or use local Whisper (MEDROUTE_ASR_MODE=local)."
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
        from voice.roman_urdu import prefer_clinic_transcript
        from voice.whisper_local import get_local_asr

        t0 = time.perf_counter()
        asr = get_local_asr()
        text, detected = asr.transcribe(audio_bytes, language=language)
        latency = int((time.perf_counter() - t0) * 1000)
        if not text:
            # Soft-fail: 200 with empty text is easier for the UI than a 503
            # that looks like a server outage when the model simply heard no speech.
            log.warning(
                "Local Whisper returned an empty transcript (%d bytes). "
                "Speak closer or use typed text.",
                len(audio_bytes),
            )
            return Transcript(
                text="",
                language=detected or language or "auto",
                latency_ms=latency,
                source="local",
            )
        # Native Urdu output is romanized for the clinic UI and symptom parser.
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
        """Transcribe audio with local Whisper, HF Inference API, or a remote server."""
        if not audio_bytes or len(audio_bytes) < 64:
            raise ValueError("Audio payload is empty or too short to transcribe.")

        mode = self._mode()
        local_ok = _local_backend_available()
        remote_ok = bool(settings.asr_server_url)
        hf_ok = bool(settings.hf_token)
        errors: list[str] = []

        # Build attempt order
        order: list[str] = []
        if mode == "local":
            order = ["local"]
        elif mode == "remote":
            order = ["remote"]
        elif mode == "hf":
            order = ["hf"]
        else:  # auto: prefer local, then HF (free), then custom remote
            if local_ok:
                order.append("local")
            if hf_ok:
                order.append("hf")
            if remote_ok:
                order.append("remote")
            if not order:
                order = ["local"]  # last resort: download & run locally

        for backend in order:
            try:
                if backend == "local":
                    log.info("ASR: using local Whisper %s", settings.asr_model)
                    return self._transcribe_local(audio_bytes, language=language)
                elif backend == "hf":
                    log.info("ASR: using HuggingFace Inference API")
                    return self._transcribe_hf(audio_bytes, language)
                log.info("ASR: using remote server %s", settings.asr_server_url)
                return self._transcribe_remote(audio_bytes, sample_rate, language)
            except Exception as exc:
                msg = f"{backend}: {exc}"
                errors.append(msg)
                log.warning("ASR %s failed: %s", backend, exc)

        detail = "; ".join(errors) if errors else "no ASR backend available"
        raise RuntimeError(
            f"ASR unavailable ({detail}). "
            "Set MEDROUTE_ASR_MODE=hf for HuggingFace hosted inference, "
            "or install local Whisper: `python scripts/download_asr_model.py`, "
            "or use typed text input."
        )

    def transcribe_text(self, text: str, language: str = "hi-IN") -> Transcript:
        """Passthrough for typed text — no ASR needed."""
        return Transcript(
            text=text, language=language, latency_ms=0, source="passthrough"
        )


transcriber = Transcriber()
