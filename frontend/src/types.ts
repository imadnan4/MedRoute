export type Mode = "intake" | "triage";
export type Domain = "home_health" | "legal";

export interface TriageRequest {
  transcript: string;
  language?: string;
  age_years?: number | null;
  age_months?: number | null;
  pregnancy?: string | null;
}

export interface EncounterRequest {
  transcript: string | null;
  audio_base64?: string | null;
  language?: string | null;
  age_years?: number | null;
  age_months?: number | null;
  pregnancy?: string | null;
  mode: Mode;
  domain?: Domain;
}

/** Data assembled by the input form and handed to the submit handler. */
export interface IntakeSubmitData {
  transcript: string;
  age_years: number | null;
  age_months: number | null;
  pregnancy: string | null;
}

export interface PatientContext {
  age_years?: number | null;
  age_months?: number | null;
  pregnancy: string;
  age_for_display: string;
  duration_days?: number | null;
}

export interface RedFlagResult {
  triggered: boolean;
  flag_class?: string | null;
  message: string;
  matched_symptoms?: string[];
}

export interface TriageResult {
  route: string;
  confidence: number;
  confidence_level: string;
  likely_condition: string;
  differential: string[];
  recommendation: string;
  watch_for: string[];
  rag_evidence: string[];
  red_flag?: RedFlagResult | null;
  reasoning: string;
  patient: PatientContext;
  urgency?: string;
  model_confidence?: number | null;
  scorer_confidence?: number | null;
  cascade_used?: string[];
}

export interface TriageResponse {
  case_id: string;
  result: TriageResult;
  stages?: Record<string, unknown>;
}

export interface PipelineEvent {
  stage: "asr" | "parser" | "safety" | "scorer" | "agent" | "done" | "error";
  status: "running" | "completed" | "error";
  data: Record<string, unknown>;
}

/**
 * Red-flag safety result as serialized by the backend (RedFlagResult.model_dump):
 * `triggered`, `flag_class`, `message`, `matched_symptoms`.
 * NOTE: the task brief described `klass`/`reason`/`matched`; the real backend
 * emits `flag_class`/`message`/`matched_symptoms`, so we match the backend.
 */
export interface RedFlag {
  triggered: boolean;
  flag_class?: string | null;
  message: string;
  matched_symptoms?: string[];
}

/** Intake distress / safeguarding result: `triggered`, `klass`, `reason`, `matched`. */
export interface Distress {
  triggered: boolean;
  klass?: string | null;
  reason: string;
  matched?: string[];
}

export type Disposition =
  | "escalate_to_clinician"
  | "standard_queue"
  | "provide_guidance";

/** IntakeRecord.to_dict() — the structured capture for intake mode. */
export interface ExtractedRecord {
  contact_name?: string | null;
  care_recipient_name?: string | null;
  phone_or_contact?: string | null;
  care_need_summary: string;
  condition_or_issue?: string | null;
  mobility_or_severity_notes?: string | null;
  insurance_or_payment_notes?: string | null;
  preferred_availability?: string | null;
  free_text_summary: string;
}

/** Body returned by POST /encounter in intake mode (Encounter → _encounter_to_dict). */
export interface IntakeResponse {
  mode: string;
  input_mode: string;
  input_language: string | null;
  raw_transcript: string;
  red_flag: RedFlag | null;
  distress: Distress | null;
  acuity: string | null;
  disposition: Disposition;
  extracted: ExtractedRecord;
  case_id: string;
  needs_human_review: boolean;
}

/** Body returned by POST /encounter in triage mode (the persisted store entry). */
export interface TriageEncounterResponse {
  case_id: string;
  request: Record<string, unknown>;
  stages: Record<string, unknown>;
  result: TriageResult;
  mode: string;
}

export type EncounterResponse = IntakeResponse | TriageEncounterResponse;
