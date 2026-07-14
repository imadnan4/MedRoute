"""Local Nemotron 3.5 ASR via ONNX Runtime GenAI.

Downloads the ONNX INT4 model (~760MB) on first use. Runs entirely on CPU.
No GPU, no API key, no network needed after model download.

Model: onnx-community/nemotron-3.5-asr-streaming-0.6b-onnx-int4
- 600M params quantized to INT4
- Multilingual + auto language detection
"""
from __future__ import annotations

import io
import json
import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

MODEL_REPO = "onnx-community/nemotron-3.5-asr-streaming-0.6b-onnx-int4"
DEFAULT_MODEL_DIR = os.path.expanduser("~/.cache/medroute/nemotron-asr")

# Language ID mapping (matches onnxruntime-genai examples/python/nemotron_speech.py)
LANG_TO_ID: dict[str, tuple[int, str]] = {
    "en": (0, "English"), "en-US": (0, "English"),
    "en-GB": (1, "English (UK)"), "es-ES": (2, "Spanish (Spain)"),
    "es": (3, "Spanish"), "es-US": (3, "Spanish (LATAM)"),
    "zh-CN": (4, "Chinese"), "hi": (6, "Hindi"), "hi-IN": (6, "Hindi"),
    # Urdu is not a first-class Nemotron locale; Hindi is the closest Indic option
    # No native Urdu locale on this Nemotron build — Hindi encoder is closest Indic option
    "ur": (6, "Urdu (via Hindi encoder)"), "ur-PK": (6, "Urdu (via Hindi encoder)"),
    "ar": (7, "Arabic"), "fr": (8, "French"), "fr-FR": (8, "French"),
    "de": (9, "German"), "de-DE": (9, "German"),
    "ja": (10, "Japanese"), "ja-JP": (10, "Japanese"),
    "ru": (11, "Russian"), "ru-RU": (11, "Russian"),
    "pt-BR": (12, "Portuguese (Brazil)"),
    "pt": (13, "Portuguese"), "pt-PT": (13, "Portuguese"),
    "ko": (14, "Korean"), "ko-KR": (14, "Korean"),
    "it": (15, "Italian"), "it-IT": (15, "Italian"),
    "nl": (16, "Dutch"), "nl-NL": (16, "Dutch"),
    "pl": (17, "Polish"), "pl-PL": (17, "Polish"),
    "tr": (18, "Turkish"), "tr-TR": (18, "Turkish"),
    "uk": (19, "Ukrainian"), "uk-UA": (19, "Ukrainian"),
    "ro": (20, "Romanian"), "ro-RO": (20, "Romanian"),
    "el": (21, "Greek"), "el-GR": (21, "Greek"),
    "cs": (22, "Czech"), "cs-CZ": (22, "Czech"),
    "hu": (23, "Hungarian"), "hu-HU": (23, "Hungarian"),
    "sv": (24, "Swedish"), "sv-SE": (24, "Swedish"),
    "da": (25, "Danish"), "da-DK": (25, "Danish"),
    "fi": (26, "Finnish"), "fi-FI": (26, "Finnish"),
    "sk": (28, "Slovak"), "sk-SK": (28, "Slovak"),
    "hr": (29, "Croatian"), "hr-HR": (29, "Croatian"),
    "bg": (30, "Bulgarian"), "bg-BG": (30, "Bulgarian"),
    "lt": (31, "Lithuanian"), "lt-LT": (31, "Lithuanian"),
    "th": (32, "Thai"), "th-TH": (32, "Thai"),
    "vi": (33, "Vietnamese"), "vi-VN": (33, "Vietnamese"),
    "et": (60, "Estonian"), "et-EE": (60, "Estonian"),
    "lv": (61, "Latvian"), "lv-LV": (61, "Latvian"),
    "sl": (62, "Slovenian"), "sl-SI": (62, "Slovenian"),
    "he": (64, "Hebrew"), "he-IL": (64, "Hebrew"),
    "fr-CA": (100, "French (Canada)"),
    "auto": (101, "Auto-detect"), "mt": (102, "Maltese"),
    "nb": (103, "Norwegian"), "nn": (104, "Norwegian Nynorsk"),
}


def _ensure_model(model_dir: str) -> str:
    """Download the ONNX model from HuggingFace if not already cached."""
    if os.path.exists(os.path.join(model_dir, "genai_config.json")) and \
       os.path.exists(os.path.join(model_dir, "encoder.onnx.data")):
        return model_dir

    log.info("Downloading Nemotron 3.5 ASR ONNX INT4 model (~760MB)...")
    os.makedirs(model_dir, exist_ok=True)

    from huggingface_hub import snapshot_download

    token = None
    try:
        from config import settings
        token = settings.hf_token or os.environ.get("HF_TOKEN") or None
    except Exception:
        token = os.environ.get("HF_TOKEN")

    snapshot_download(
        MODEL_REPO,
        local_dir=model_dir,
        token=token,
        ignore_patterns=["README.md", ".gitattributes"],
    )
    log.info("Model downloaded to %s", model_dir)
    return model_dir


def _decode_with_ffmpeg(audio_bytes: bytes, target_sr: int, suffix: str = ".webm") -> np.ndarray:
    """Decode any ffmpeg-supported format to mono float32 PCM at target_sr."""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", tmp_path,
            "-ac", "1",
            "-ar", str(target_sr),
            "-f", "f32le",
            "pipe:1",
        ]
        completed = subprocess.run(cmd, check=True, capture_output=True)
        audio = np.frombuffer(completed.stdout, dtype=np.float32).copy()
        if audio.size == 0:
            raise RuntimeError("ffmpeg produced empty audio")
        return np.ascontiguousarray(audio, dtype=np.float32)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _decode_with_pydub(audio_bytes: bytes, target_sr: int, suffix: str = ".webm") -> np.ndarray:
    """Pydub fallback. Must scale by sample_width (webm/opus is often 32-bit)."""
    from pydub import AudioSegment

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        audio = AudioSegment.from_file(tmp_path)
        audio = audio.set_frame_rate(target_sr).set_channels(1)
        raw = np.array(audio.get_array_of_samples(), dtype=np.float32)
        # sample_width: 1=8bit, 2=16bit, 4=32bit — browser webm/opus often lands as 32-bit
        max_val = float(1 << (8 * audio.sample_width - 1))
        if max_val <= 0:
            max_val = 32768.0
        return np.ascontiguousarray(raw / max_val, dtype=np.float32)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _sniff_suffix(audio_bytes: bytes) -> str:
    """Best-effort container suffix for temp files / ffmpeg."""
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
    # Browser MediaRecorder default
    if audio_bytes[:4] == b"\x1a\x45\xdf\xa3" or b"webm" in audio_bytes[:64].lower():
        return ".webm"
    return ".webm"


def _decode_audio(audio_bytes: bytes, target_sr: int = 16000) -> np.ndarray:
    """Decode audio bytes (wav, webm, mp3, etc.) to float32 mono numpy array in [-1, 1].

    Browser MediaRecorder sends webm/opus. pydub often yields 32-bit samples; dividing
    by 32768 (16-bit max) explodes amplitude and Nemotron returns empty transcripts.
    Prefer ffmpeg → float32 LE, which matches the official ONNX Runtime GenAI path.
    """
    if not audio_bytes:
        raise ValueError("Empty audio payload")

    suffix = _sniff_suffix(audio_bytes)

    # WAV / FLAC / OGG via soundfile (fast path)
    if suffix in {".wav", ".flac", ".ogg"}:
        try:
            import soundfile as sf
            data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1)
            if sr != target_sr:
                from scipy.signal import resample
                num_samples = int(len(data) * target_sr / sr)
                data = resample(data, num_samples).astype(np.float32)
            return np.ascontiguousarray(data, dtype=np.float32)
        except Exception as exc:
            log.debug("soundfile decode failed (%s), trying ffmpeg", exc)

    # Compressed / browser formats: ffmpeg → mono f32le (correct amplitude)
    try:
        return _decode_with_ffmpeg(audio_bytes, target_sr, suffix=suffix)
    except Exception as exc:
        log.warning("ffmpeg decode failed (%s), trying pydub", exc)

    try:
        return _decode_with_pydub(audio_bytes, target_sr, suffix=suffix)
    except Exception as exc:
        raise RuntimeError(
            f"Could not decode audio ({suffix}). Install ffmpeg and ensure the "
            f"browser sent webm/wav/mp3. Underlying error: {exc}"
        ) from exc


def _decode_tokens(generator, tokenizer_stream) -> str:
    """Decode all available tokens (matches official nemotron_speech.py)."""
    text = ""
    while not generator.is_done():
        generator.generate_next_token()
        tokens = generator.get_next_tokens()
        if len(tokens) > 0:
            token_text = tokenizer_stream.decode(tokens[0])
            if token_text:
                text += token_text
    return text


class NemotronLocalASR:
    """Local Nemotron 3.5 ASR via ONNX Runtime GenAI.

    Downloads the INT4 model on first use. Runs on CPU — no GPU required.
    Creates a fresh StreamingProcessor + Generator per transcription (required;
    reusing the processor across calls leaves bad internal state).
    """

    def __init__(self, model_dir: str = DEFAULT_MODEL_DIR):
        self._model_dir = _ensure_model(model_dir)
        self._model: Optional[object] = None
        self._sample_rate: int = 16000
        self._chunk_samples: int = 8960

    def _load(self):
        if self._model is not None:
            return

        import onnxruntime_genai as og

        log.info("Loading Nemotron 3.5 ASR model from %s...", self._model_dir)
        t0 = time.perf_counter()

        config = og.Config(self._model_dir)
        # Prefer CPU when no GPU EP is configured
        try:
            config.clear_providers()
        except Exception:
            pass

        self._model = og.Model(config)

        config_path = os.path.join(self._model_dir, "genai_config.json")
        with open(config_path) as f:
            cfg = json.load(f)["model"]
        self._sample_rate = int(cfg["sample_rate"])
        self._chunk_samples = int(cfg["chunk_samples"])

        log.info("Model loaded in %.2fs", time.perf_counter() - t0)

    def _new_session(self, lang_id: int):
        """Fresh StreamingProcessor + Generator for one utterance."""
        import onnxruntime_genai as og

        processor = og.StreamingProcessor(self._model)
        # VAD drops low-volume mic input on CPU; keep off for clinic mics
        try:
            processor.set_option("use_vad", "false")
        except Exception:
            pass

        tokenizer = og.Tokenizer(self._model)
        tokenizer_stream = tokenizer.create_stream()
        params = og.GeneratorParams(self._model)
        generator = og.Generator(self._model, params)
        generator.set_runtime_option("lang_id", str(int(lang_id)))
        return processor, tokenizer_stream, generator

    @staticmethod
    def _normalize_audio(audio: np.ndarray, target_peak: float = 0.85) -> np.ndarray:
        """Peak-normalize so quiet mics still sit in a healthy dynamic range."""
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak < 1e-6:
            return audio
        # Only boost quiet audio or gently pull back clipping; leave healthy levels alone
        if peak < 0.05 or peak > 0.99:
            scale = target_peak / peak
            audio = (audio * scale).astype(np.float32)
            log.info("ASR peak-normalize: peak=%.4f → scale=%.2f", peak, scale)
        return np.ascontiguousarray(audio, dtype=np.float32)

    def _run_once(self, audio: np.ndarray, lang_key: str) -> tuple[str, int, int, int]:
        """Single language pass. Returns (text, latency_ms, chunks_processed, chunks_total)."""
        lang_id, lang_name = LANG_TO_ID[lang_key]
        processor, tokenizer_stream, generator = self._new_session(lang_id)
        t0 = time.perf_counter()

        full_text = ""
        chunks_total = chunks_processed = 0
        cs = self._chunk_samples

        for i in range(0, len(audio), cs):
            chunk = audio[i:i + cs]
            # Pad final short frame — StreamingProcessor can drop undersized tails
            if len(chunk) < cs:
                chunk = np.pad(chunk, (0, cs - len(chunk)))
            chunk = np.ascontiguousarray(chunk, dtype=np.float32)
            chunks_total += 1
            inputs = processor.process(chunk)
            if inputs is not None:
                chunks_processed += 1
                generator.set_inputs(inputs)
                full_text += _decode_tokens(generator, tokenizer_stream)

        inputs = processor.flush()
        if inputs is not None:
            generator.set_inputs(inputs)
            full_text += _decode_tokens(generator, tokenizer_stream)

        latency_ms = int((time.perf_counter() - t0) * 1000)
        text = full_text.strip()
        log.info(
            "ASR pass lang=%s/%s → %d chars in %dms (chunks %d/%d)",
            lang_key, lang_name, len(text), latency_ms, chunks_processed, chunks_total,
        )
        return text, latency_ms, chunks_processed, chunks_total

    def _dump_debug_audio(self, audio: np.ndarray, audio_bytes: bytes) -> None:
        """Persist last failed clip for offline diagnosis."""
        try:
            debug_dir = Path(os.path.expanduser("~/.cache/medroute/asr-debug"))
            debug_dir.mkdir(parents=True, exist_ok=True)
            ts = int(time.time())
            raw_path = debug_dir / f"fail_{ts}.bin"
            raw_path.write_bytes(audio_bytes[: min(len(audio_bytes), 2_000_000)])
            # Also write float32 PCM wav via soundfile if available
            try:
                import soundfile as sf
                wav_path = debug_dir / f"fail_{ts}.wav"
                sf.write(str(wav_path), audio, self._sample_rate)
                log.warning("ASR empty — saved debug clip to %s", wav_path)
            except Exception:
                log.warning("ASR empty — saved raw bytes to %s", raw_path)
        except Exception as exc:
            log.debug("Could not dump ASR debug audio: %s", exc)

    def transcribe(self, audio_bytes: bytes, language: str = "auto") -> tuple[str, str]:
        """Transcribe audio bytes to text.

        Clinic languages only: **Urdu → English** (no Hindi pass).
        Explicit ``language`` is tried first if it is not ``auto``.

        Note: Nemotron has no dedicated Urdu locale; ``ur`` uses the Hindi
        encoder id (closest Indic option for Urdu speech).

        Args:
            audio_bytes: Raw audio (webm, wav, mp3, etc.)
            language: Language code (e.g. 'ur', 'en', 'auto').

        Returns:
            (transcript_text, language_key_used)
        """
        self._load()

        audio = _decode_audio(audio_bytes, self._sample_rate)
        audio = self._normalize_audio(audio)
        duration = len(audio) / float(self._sample_rate) if len(audio) else 0.0
        rms = float(np.sqrt(np.mean(np.square(audio)))) if len(audio) else 0.0
        peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
        log.info(
            "ASR input: %d samples (%.2fs) rms=%.4f peak=%.4f bytes=%d",
            len(audio), duration, rms, peak, len(audio_bytes),
        )
        if len(audio) < self._sample_rate * 0.2:
            log.warning("Audio shorter than 0.2s — likely no speech captured")
        if rms < 1e-4:
            log.warning("Audio is near-silent (rms=%.6f)", rms)

        preferred = language if language in LANG_TO_ID else "ur"

        # Near-silence: skip expensive multi-pass inference
        if rms < 1e-3 or peak < 0.005:
            log.warning("Skipping ASR — signal too quiet (rms=%.5f peak=%.5f)", rms, peak)
            self._dump_debug_audio(audio, audio_bytes)
            return "", preferred

        # Clinic priority: Urdu → English only
        clinic_order = ("ur", "en")
        candidates: list[str] = []
        if preferred not in ("auto",) and preferred in LANG_TO_ID:
            # Map Hindi requests to Urdu path (same encoder; clinic is Urdu+English only)
            if preferred in ("hi", "hi-IN"):
                preferred = "ur"
            candidates.append(preferred)
        for key in clinic_order:
            if key not in candidates:
                candidates.append(key)

        attempt_order: list[str] = []
        seen_ids: set[int] = set()
        for key in candidates:
            lang_id = LANG_TO_ID[key][0]
            if lang_id in seen_ids:
                continue
            seen_ids.add(lang_id)
            attempt_order.append(key)

        best_text = ""
        best_lang = preferred if preferred != "auto" else "ur"
        for lang_key in attempt_order:
            text, _lat, _p, _t = self._run_once(audio, lang_key)
            if text:
                return text, lang_key
            if len(text) > len(best_text):
                best_text, best_lang = text, lang_key

        self._dump_debug_audio(audio, audio_bytes)
        return best_text, best_lang


# Singleton (model weights only — sessions are per-call)
_asr: Optional[NemotronLocalASR] = None


def get_local_asr() -> NemotronLocalASR:
    global _asr
    if _asr is None:
        _asr = NemotronLocalASR()
    return _asr
