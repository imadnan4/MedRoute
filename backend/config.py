from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).resolve().parent / ".env"),
        env_prefix="MEDROUTE_",
        extra="ignore",
    )

    # — Models
    asr_model: str = "large-v3-turbo"
    embed_model: str = "all-MiniLM-L6-v2"

    # — Hosted inference (OpenRouter's free router is intended for testing)
    openrouter_api_key: str = ""
    openrouter_model: str = "openrouter/free"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_timeout: float = 60.0
    openrouter_max_attempts: int = 2
    openrouter_site_url: str = ""

    # — Optional remote ASR endpoint
    asr_server_url: str = ""
    # auto | local | remote | hf — hf uses HuggingFace hosted Inference API (no local GPU/RAM)
    asr_mode: str = "auto"
    # Model to use on HuggingFace Inference API when asr_mode=hf
    hf_asr_model: str = "openai/whisper-large-v3"

    # — HuggingFace (optional, speeds up model downloads)
    hf_token: str = ""

    # — Routing thresholds
    escalation_bias_threshold: float = 0.65
    local_only_max_complexity: int = 3
    rag_max_complexity: int = 6
    remote_min_complexity: int = 7
    confidence_high: float = 0.80
    confidence_medium: float = 0.65

    # — Patient context multipliers
    age_lt_3mo_offset: int = 4
    age_lt_2yr_offset: int = 2
    age_gt_65_offset: int = 1
    pregnancy_offset: int = 2
    pregnancy_3rd_trimester_offset: int = 3

    # — ASR
    asr_device: str = "auto"  # auto | cpu | cuda
    asr_compute_type: str = "auto"  # auto | int8 | float16 | float32
    asr_beam_size: int = 5
    asr_cache_dir: str = str(Path.home() / ".cache" / "medroute" / "whisper")

    # — RAG (path resolved relative to project root, not CWD)
    chroma_path: str = str(Path(__file__).resolve().parent.parent / "data" / "chroma")
    rag_top_k: int = 4

    # — WHO ICD-11
    icd_api_base: str = "https://id.who.int/icd"
    icd_api_key: str = ""

    # — Server
    host: str = "0.0.0.0"
    port: int = 8000


settings = Settings()
