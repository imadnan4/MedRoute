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
    asr_model: str = "nvidia/nemotron-3.5-asr-streaming-0.6b"
    local_llm: str = "emrecanacikgoz/hippomistral"
    local_llm_ollama_name: str = "hippomistral"
    embed_model: str = "sentence-transformers/embeddinggemma-300m-medical"
    remote_llm: str = "accounts/fireworks/models/deepseek-v4"

    # — AMD Developer Cloud endpoints
    ollama_base_url: str = "http://localhost:11434"
    asr_server_url: str = ""
    # auto | local | remote — auto prefers local ONNX when model is cached
    asr_mode: str = "auto"

    # — Together AI (Nemotron 3.5 ASR hosted, fallback)
    together_api_key: str = ""

    # — HuggingFace (optional, speeds up model downloads)
    hf_token: str = ""

    # — Fireworks
    fireworks_api_key: str = ""
    fireworks_timeout: float = 30.0

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
    asr_chunk_ms: int = 160

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
