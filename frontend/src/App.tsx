import { useState } from "react";
import VoiceRecorder from "./components/VoiceRecorder";
import TriageResultView from "./components/TriageResult";
import ReportDownload from "./components/ReportDownload";
import type { PipelineEvent, TriageResult } from "./types";
import { streamTriage } from "./api";
import "./App.css";

function BrandMark() {
  return (
    <span className="brand-mark" aria-hidden="true">
      <svg viewBox="0 0 28 28" fill="none">
        <path
          d="M14 4v20M4 14h20"
          stroke="currentColor"
          strokeWidth="2.4"
          strokeLinecap="round"
        />
        <circle
          cx="14"
          cy="14"
          r="11"
          stroke="currentColor"
          strokeWidth="1.2"
        />
      </svg>
    </span>
  );
}

interface ProgressState {
  percent: number;
  label: string;
  detail: string;
  completed: string[];
}

const INITIAL_PROGRESS: ProgressState = {
  percent: 1,
  label: "Connecting to MedRoute",
  detail: "Connecting to the clinical pipeline",
  completed: [],
};

const STAGE_PROGRESS: Record<
  string,
  { running: number; completed: number; label: string; detail: string }
> = {
  asr: {
    running: 3,
    completed: 14,
    label: "Preparing patient input",
    detail: "Confirming transcript and language",
  },
  parser: {
    running: 18,
    completed: 32,
    label: "Structuring symptoms",
    detail: "Extracting symptoms, duration, and patient context",
  },
  safety: {
    running: 37,
    completed: 52,
    label: "Checking emergency patterns",
    detail: "Running deterministic red-flag safety rules",
  },
  scorer: {
    running: 57,
    completed: 69,
    label: "Scoring clinical complexity",
    detail: "Selecting the safest assessment route",
  },
  agent: {
    running: 74,
    completed: 96,
    label: "Preparing recommendation",
    detail: "Retrieving evidence and generating the clinical assessment",
  },
  done: {
    running: 99,
    completed: 100,
    label: "Assessment complete",
    detail: "Final result is ready",
  },
};

function App() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<TriageResult | null>(null);
  const [caseId, setCaseId] = useState("");
  const [progress, setProgress] = useState<ProgressState>(INITIAL_PROGRESS);

  async function handleTranscript(text: string) {
    setLoading(true);
    setError("");
    setResult(null);
    setProgress(INITIAL_PROGRESS);
    try {
      const response = await streamTriage(
        { transcript: text, language: "ur" },
        (event: PipelineEvent) => {
          const stage = STAGE_PROGRESS[event.stage];
          if (!stage) return;
          const isCompleted = event.status === "completed";
          setProgress((current) => ({
            percent: isCompleted ? stage.completed : stage.running,
            label: stage.label,
            detail: stage.detail,
            completed:
              isCompleted &&
              event.stage !== "done" &&
              !current.completed.includes(event.stage)
                ? [...current.completed, event.stage]
                : current.completed,
          }));
        },
      );
      setResult(response.result);
      setCaseId(response.case_id);
    } catch (err: unknown) {
      const message =
        err instanceof Error
          ? err.message
          : "Connection failed. Check that the MedRoute backend is available.";
      setError(message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="#main-content" aria-label="MedRoute home">
          <BrandMark />
          <span>MedRoute</span>
        </a>
        <div className="topbar-meta">
          <span className="system-status">
            <i /> Decision support online
          </span>
          <span className="topbar-divider" aria-hidden="true" />
          <span>Clinical workspace</span>
        </div>
      </header>

      <main id="main-content" className="workspace">
        <section className="hero" aria-labelledby="page-title">
          <div>
            <p className="eyebrow">Voice-first clinical intake</p>
            <h1 id="page-title">
              Clear next steps,
              <br />
              when every minute matters.
            </h1>
          </div>
          <p className="hero-copy">
            Describe the patient’s symptoms in Urdu or English. MedRoute checks
            urgent red flags first, then prepares a grounded triage assessment
            for clinician review.
          </p>
        </section>

        <section className="intake-layout" aria-label="Patient intake">
          <div className="intake-card">
            <div className="section-heading">
              <div>
                <p className="eyebrow">New assessment</p>
                <h2>Patient presentation</h2>
              </div>
              <span className="step-index">01</span>
            </div>
            <VoiceRecorder onTranscript={handleTranscript} disabled={loading} />
          </div>

          <aside className="process-panel" aria-label="Assessment process">
            <p className="eyebrow">How it works</p>
            <ol className="process-list">
              <li>
                <span>1</span>
                <div>
                  <strong>Capture</strong>
                  <p>Voice or typed symptoms, age and pregnancy context.</p>
                </div>
              </li>
              <li>
                <span>2</span>
                <div>
                  <strong>Screen</strong>
                  <p>Deterministic emergency rules run before any model.</p>
                </div>
              </li>
              <li>
                <span>3</span>
                <div>
                  <strong>Review</strong>
                  <p>
                    Receive urgency, evidence and an auditable recommendation.
                  </p>
                </div>
              </li>
            </ol>
            <div className="safety-note">
              <svg viewBox="0 0 20 20" fill="none" aria-hidden="true">
                <path
                  d="M10 2.5 16 5v4.3c0 3.7-2.3 6.8-6 8.2-3.7-1.4-6-4.5-6-8.2V5l6-2.5Z"
                  stroke="currentColor"
                  strokeWidth="1.5"
                />
                <path
                  d="m7.4 10 1.7 1.7 3.7-4"
                  stroke="currentColor"
                  strokeWidth="1.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
              <p>
                <strong>Safety comes first.</strong> Emergency patterns bypass
                AI reasoning and trigger immediate escalation.
              </p>
            </div>
          </aside>
        </section>

        {loading && (
          <section
            className="analysis-card"
            aria-live="polite"
            aria-label="Assessment in progress"
          >
            <div className="analysis-header">
              <div>
                <p className="eyebrow">Assessment in progress</p>
                <h2>{progress.label}</h2>
                <p className="progress-detail">{progress.detail}</p>
              </div>
              <strong className="progress-number">
                {progress.percent}
                <small>%</small>
              </strong>
            </div>
            <div
              className="progress-track"
              role="progressbar"
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={progress.percent}
            >
              <span
                style={{ transform: `scaleX(${progress.percent / 100})` }}
              />
            </div>
            <div className="analysis-steps">
              {[
                ["asr", "Input"],
                ["parser", "Symptoms"],
                ["safety", "Safety"],
                ["scorer", "Complexity"],
                ["agent", "Recommendation"],
              ].map(([stage, label]) => (
                <span
                  key={stage}
                  className={
                    progress.completed.includes(stage) ? "complete" : ""
                  }
                >
                  {progress.completed.includes(stage) ? "✓ " : ""}
                  {label}
                </span>
              ))}
            </div>
          </section>
        )}

        {error && (
          <section className="error-section" role="alert">
            <svg viewBox="0 0 20 20" fill="none" aria-hidden="true">
              <circle
                cx="10"
                cy="10"
                r="7.5"
                stroke="currentColor"
                strokeWidth="1.5"
              />
              <path
                d="M10 6.2v4.6M10 13.8v.1"
                stroke="currentColor"
                strokeWidth="1.5"
                strokeLinecap="round"
              />
            </svg>
            <div>
              <h3>Assessment could not be completed</h3>
              <p>{error}</p>
            </div>
          </section>
        )}

        {result && (
          <section className="result-area" aria-labelledby="result-title">
            <div className="result-area-header">
              <div>
                <p className="eyebrow">Assessment complete</p>
                <h2 id="result-title">Triage summary</h2>
              </div>
              <span className="case-reference">
                Case {caseId.slice(0, 8).toUpperCase()}
              </span>
            </div>
            <TriageResultView result={result} />
            <ReportDownload caseId={caseId} />
          </section>
        )}
      </main>

      <footer className="app-footer">
        <div>
          <BrandMark />
          <span>MedRoute</span>
        </div>
        <p>
          Clinical decision support only. Always verify findings with a
          qualified healthcare professional.
        </p>
        <span>Research prototype · 2026</span>
      </footer>
    </div>
  );
}

export default App;
