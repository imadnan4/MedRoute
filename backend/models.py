"""Shared data models for the MedRoute pipeline."""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class PregnancyStatus(str, Enum):
    NOT_PREGNANT = "not_pregnant"
    PREGNANT = "pregnant"
    PREGNANT_3RD_TRIMESTER = "pregnant_3rd_trimester"


class PatientContext(BaseModel):
    """Structured patient context extracted by the Input Parser."""

    age_years: Optional[float] = None
    age_months: Optional[float] = None  # for infants < 2yr
    pregnancy: PregnancyStatus = PregnancyStatus.NOT_PREGNANT
    duration_days: Optional[float] = None  # symptom duration if stated

    @property
    def age_for_display(self) -> str:
        if self.age_months is not None and self.age_months < 24:
            return f"{self.age_months:.0f} months"
        if self.age_years is not None:
            return f"{self.age_years:.0f} years"
        return "unknown"


class ParsedInput(BaseModel):
    """Output of Stage 1 (Input Parser)."""

    transcript: str
    language: str = "unknown"  # e.g. hi-IN, auto-detected by Nemotron
    symptoms: list[str] = Field(default_factory=list)
    patient: PatientContext = Field(default_factory=PatientContext)
    symptom_clusters: list[str] = Field(default_factory=list)


class RedFlagResult(BaseModel):
    """Output of Stage 2 (Safety Pre-Check)."""

    triggered: bool = False
    flag_class: Optional[str] = None  # MI, stroke, sepsis, ...
    message: str = ""
    matched_symptoms: list[str] = Field(default_factory=list)


class TriageRoute(str, Enum):
    HARD_ESCALATION = "hard_escalation"          # red flag, no LLM
    LOCAL_ONLY = "local_only"                    # Hippo-Mistral-7B
    LOCAL_WITH_RAG = "local_with_rag"            # Hippo + EmbeddingGemma RAG
    REMOTE = "remote"                            # DeepSeek V4
    OUTAGE_FALLBACK = "outage_fallback"          # Fireworks down -> clinician
    ESCALATION_BIAS = "escalation_bias"          # low confidence, re-route up


class TriageScore(BaseModel):
    """Output of Stage 3 pre-routing (Complexity Scorer)."""

    raw_score: int
    adjusted_score: int
    confidence: float
    escalation_bias_applied: bool = False
    context_offset: int = 0
    route: TriageRoute
    reasoning: str = ""
    syndrome_hits: list[str] = Field(default_factory=list)
    vagueness_penalty: float = 0.0


class ConfidenceLevel(str, Enum):
    GREEN = "green"    # > 0.80
    YELLOW = "yellow"  # 0.65 - 0.80
    RED = "red"        # < 0.65


class UrgencyLevel(str, Enum):
    EMERGENCY = "emergency"      # immediate care / ED
    URGENT = "urgent"            # same-day clinician
    SOON = "soon"                # within 24–48h
    ROUTINE = "routine"          # primary care / self-care guidance


class TriageResult(BaseModel):
    """Final output of the Triage Agent for a single case."""

    route: TriageRoute
    confidence: float
    confidence_level: ConfidenceLevel
    likely_condition: str = ""
    differential: list[str] = Field(default_factory=list)
    recommendation: str = ""
    watch_for: list[str] = Field(default_factory=list)
    rag_evidence: list[str] = Field(default_factory=list)
    red_flag: Optional[RedFlagResult] = None
    reasoning: str = ""
    patient: PatientContext = Field(default_factory=PatientContext)
    urgency: UrgencyLevel = UrgencyLevel.SOON
    model_confidence: Optional[float] = None
    scorer_confidence: Optional[float] = None
    cascade_used: list[str] = Field(default_factory=list)
