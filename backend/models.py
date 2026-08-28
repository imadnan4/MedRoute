"""Shared data models for the MedRoute pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

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
    language: str = "unknown"  # e.g. ur, hi, en; detected or user-supplied
    symptoms: list[str] = Field(default_factory=list)
    patient: PatientContext = Field(default_factory=PatientContext)
    symptom_clusters: list[str] = Field(default_factory=list)


class RedFlagResult(BaseModel):
    """Output of Stage 2 (Safety Pre-Check)."""

    triggered: bool = False
    flag_class: Optional[str] = None  # MI, stroke, sepsis, ...
    message: str = ""
    matched_symptoms: list[str] = Field(default_factory=list)


class Acuity(str, Enum):
    """Categorical severity derived from a Presentation (routine / priority / urgent)."""

    ROUTINE = "routine"
    PRIORITY = "priority"
    URGENT = "urgent"


class Disposition(str, Enum):
    """Decision of where an Encounter goes next."""

    ESCALATE_TO_CLINICIAN = "escalate_to_clinician"
    STANDARD_QUEUE = "standard_queue"
    PROVIDE_GUIDANCE = "provide_guidance"


class TriageRoute(str, Enum):
    HARD_ESCALATION = "hard_escalation"  # red flag, no LLM
    # Values are retained for API compatibility; inference now uses OpenRouter.
    LOCAL_ONLY = "local_only"  # direct model inference
    LOCAL_WITH_RAG = "local_with_rag"  # model + retrieved evidence
    REMOTE = "remote"  # complex-case model inference
    OUTAGE_FALLBACK = "outage_fallback"  # provider down -> safe fallback
    ESCALATION_BIAS = "escalation_bias"  # low confidence, re-route up


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
    GREEN = "green"  # > 0.80
    YELLOW = "yellow"  # 0.65 - 0.80
    RED = "red"  # < 0.65


class UrgencyLevel(str, Enum):
    EMERGENCY = "emergency"  # immediate care / ED
    URGENT = "urgent"  # same-day clinician
    SOON = "soon"  # within 24–48h
    ROUTINE = "routine"  # primary care / self-care guidance


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
    acuity: Optional[Acuity] = None
    disposition: Optional[Disposition] = None


@dataclass
class IntakeRecord:
    """Structured capture of an Encounter's Presentation from the Intake mode.

    Holds who is contacting, the care recipient, the care need, and the logistics
    (contact, availability, insurance). Domain-neutral: home-health / care-coordination
    generic — no clinical diagnosis.
    """

    contact_name: Optional[str] = None
    care_recipient_name: Optional[str] = None
    phone_or_contact: Optional[str] = None
    care_need_summary: str = ""
    condition_or_issue: Optional[str] = None
    mobility_or_severity_notes: Optional[str] = None
    insurance_or_payment_notes: Optional[str] = None
    preferred_availability: Optional[str] = None
    free_text_summary: str = ""

    def to_dict(self) -> dict:
        return {
            "contact_name": self.contact_name,
            "care_recipient_name": self.care_recipient_name,
            "phone_or_contact": self.phone_or_contact,
            "care_need_summary": self.care_need_summary,
            "condition_or_issue": self.condition_or_issue,
            "mobility_or_severity_notes": self.mobility_or_severity_notes,
            "insurance_or_payment_notes": self.insurance_or_payment_notes,
            "preferred_availability": self.preferred_availability,
            "free_text_summary": self.free_text_summary,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "IntakeRecord":
        return cls(
            contact_name=data.get("contact_name"),
            care_recipient_name=data.get("care_recipient_name"),
            phone_or_contact=data.get("phone_or_contact"),
            care_need_summary=data.get("care_need_summary", ""),
            condition_or_issue=data.get("condition_or_issue"),
            mobility_or_severity_notes=data.get("mobility_or_severity_notes"),
            insurance_or_payment_notes=data.get("insurance_or_payment_notes"),
            preferred_availability=data.get("preferred_availability"),
            free_text_summary=data.get("free_text_summary", ""),
        )


@dataclass
class DistressResult:
    """Output of the intake distress / safeguarding pre-check.

    Parallel to ``RedFlagResult`` but for caller distress, escalation, and
    safeguarding concerns (vulnerable adult/child, domestic abuse, self-neglect).
    Deterministic, pre-LLM, and never a diagnosis.
    """

    triggered: bool = False
    klass: Optional[str] = None
    reason: str = ""
    matched: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "triggered": self.triggered,
            "klass": self.klass,
            "reason": self.reason,
            "matched": self.matched,
        }


@dataclass
class Encounter:
    """A single first-contact event resolved through one Mode over the shared core.

    ``extracted`` holds the IntakeRecord dict for intake mode, or the triage
    result dict for triage mode. ``red_flag`` reuses the shared RedFlagResult so
    both modes share one safety pre-check. ``distress`` carries the parallel
    intake safeguarding pre-check; ``domain`` records which intake variant ran.
    """

    mode: str
    input_mode: str
    input_language: Optional[str]
    raw_transcript: str
    red_flag: Any
    acuity: Optional[Acuity]
    disposition: Disposition
    extracted: dict
    case_id: str
    needs_human_review: bool
    distress: Any = None
    domain: str = "home_health"
