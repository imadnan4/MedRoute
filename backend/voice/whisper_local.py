"""Local multilingual ASR using Whisper Large V3 Turbo via faster-whisper."""

from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path
from typing import Optional

from config import settings

log = logging.getLogger(__name__)


def _audio_suffix(audio_bytes: bytes) -> str:
    if audio_bytes[:4] == b"RIFF":
        return ".wav"
    if audio_bytes[:4] == b"fLaC":
        return ".flac"
    if audio_bytes[:4] == b"OggS":
        return ".ogg"
    if len(audio_bytes) >= 12 and audio_bytes[4:8] == b"ftyp":
        return ".mp4"
    if audio_bytes[:3] == b"ID3" or audio_bytes[:2] == b"\xff\xfb":
        return ".mp3"
    return ".webm"


def _whisper_language(language: str) -> Optional[str]:
    normalized = (language or "auto").lower().split("-")[0]
    if normalized in {"ur", "hi", "en"}:
        return normalized
    return None


class WhisperLocalASR:
    """Lazy-loaded Whisper transcription optimized for conversational Urdu."""

    def __init__(self) -> None:
        self._model = None
        self._device = "cpu"
        self._compute_type = "int8"

    def _load(self):
        if self._model is not None:
            return self._model

        from faster_whisper import WhisperModel

        if settings.hf_token:
            from huggingface_hub import login

            login(token=settings.hf_token, add_to_git_credential=False)

        device = settings.asr_device.strip().lower()
        if device == "auto":
            try:
                import ctranslate2

                device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
            except Exception:
                device = "cpu"

        compute_type = settings.asr_compute_type.strip().lower()
        if compute_type == "auto":
            compute_type = "float16" if device == "cuda" else "int8"

        log.info(
            "Loading Whisper ASR model %s on %s (%s)",
            settings.asr_model,
            device,
            compute_type,
        )
        self._model = WhisperModel(
            settings.asr_model,
            device=device,
            compute_type=compute_type,
            download_root=settings.asr_cache_dir,
        )
        self._device = device
        self._compute_type = compute_type
        return self._model

    def ensure_model(self) -> None:
        """Download and initialize the configured model."""
        self._load()

    def transcribe(self, audio_bytes: bytes, language: str = "auto") -> tuple[str, str]:
        from voice.roman_urdu import prefer_clinic_transcript

        suffix = _audio_suffix(audio_bytes)
        temp_path = ""
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as audio_file:
                audio_file.write(audio_bytes)
                temp_path = audio_file.name

            model = self._load()
            language_hint = _whisper_language(language)
            initial_prompt = None
            if language_hint == "ur":
                initial_prompt = (
                    "طبی علامات: بخار، کھانسی، زکام، سر درد، سینے میں درد، "
                    "سانس کی تکلیف، الٹی، چکر، کمزوری۔"
                )

            started_at = time.perf_counter()
            segments, info = model.transcribe(
                temp_path,
                language=language_hint,
                task="transcribe",
                beam_size=settings.asr_beam_size,
                best_of=settings.asr_beam_size,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 350},
                condition_on_previous_text=False,
                initial_prompt=initial_prompt,
                word_timestamps=False,
            )
            text = " ".join(
                segment.text.strip() for segment in segments if segment.text.strip()
            ).strip()
            detected_language = (
                getattr(info, "language", None) or language_hint or "auto"
            )
            text = prefer_clinic_transcript(text, language=detected_language)
            log.info(
                "Whisper ASR: %d chars, language=%s, device=%s, latency=%dms",
                len(text),
                detected_language,
                self._device,
                int((time.perf_counter() - started_at) * 1000),
            )
            return text, detected_language
        finally:
            if temp_path:
                try:
                    Path(temp_path).unlink(missing_ok=True)
                except OSError:
                    pass


_local_asr: Optional[WhisperLocalASR] = None


def get_local_asr() -> WhisperLocalASR:
    global _local_asr
    if _local_asr is None:
        _local_asr = WhisperLocalASR()
    return _local_asr
