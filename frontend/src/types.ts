export interface TriageRequest {
  transcript: string;
  language?: string;
  age_years?: number | null;
  age_months?: number | null;
  pregnancy?: string | null;
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
