import { useState } from "react";
import VoiceRecorder from "./components/VoiceRecorder";
import TriageResultView from "./components/TriageResult";
import ReportDownload from "./components/ReportDownload";
import IntakeResult from "./components/IntakeResult";
import type {
  Domain,
  IntakeResponse,
  IntakeSubmitData,
  Mode,
  TriageResult,
} from "./types";
import { postEncounter } from "./api";
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
        <circle cx="14" cy="14" r="11" stroke="currentColor" strokeWidth="1.2" />
      </svg>
    </span>
  );
}

function App() {
  const [mode, setMode] = useState<Mode>("intake");
  const [domain, setDomain] = useState<Domain>("home_health");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [intakeResult, setIntakeResult] = useState<IntakeResponse | null>(null);
  const [triageResult, setTriageResult] = useState<TriageResult | null>(null);
  const [caseId, setCaseId] = useState("");

  async function handleSubmit(data: IntakeSubmitData) {
    setLoading(true);
    setError("");
    setIntakeResult(null);
    setTriageResult(null);

    const request = {
      transcript: data.transcript,
      language: "en",
      age_years: data.age_years,
      age_months: data.age_months,
      pregnancy: data.pregnancy,
      mode,
      domain: mode === "intake" ? domain : undefined,
    };

    try {
      const resp = await postEncounter(request);
      if ("extracted" in resp) {
        setIntakeResult(resp);
        setCaseId(resp.case_id);
      } else {
        setTriageResult(resp.result);
        setCaseId(resp.case_id);
      }
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
            <p className="eyebrow">Voice-first intake &amp; triage</p>
            <h1 id="page-title">
              Capture the story,
              <br />
              route it to the right human.
            </h1>
          </div>
          <p className="hero-copy">
            MedRoute is a clinical first-contact platform that turns a
            caregiver’s voice or typed note into a clean structured record and a
            clear triage routing decision — running deterministic safety rules
            before any model does.
          </p>
        </section>

        <section className="intake-layout" aria-label="First contact">
          <div className="intake-card">
            <div className="section-heading">
              <div>
                <p className="eyebrow">New first contact</p>
                <h2>Patient presentation</h2>
              </div>
              <span className="step-index">01</span>
            </div>

            <div className="controls-row">
              <div
                className="mode-toggle"
                role="group"
                aria-label="Select mode"
              >
                <button
                  type="button"
                  className={mode === "intake" ? "is-active" : ""}
                  aria-pressed={mode === "intake"}
                  onClick={() => setMode("intake")}
                >
                  Intake
                </button>
                <button
                  type="button"
                  className={mode === "triage" ? "is-active" : ""}
                  aria-pressed={mode === "triage"}
                  onClick={() => setMode("triage")}
                >
                  Triage
                </button>
              </div>

              {mode === "intake" && (
                <div className="domain-selector">
                  <label htmlFor="domain">Domain</label>
                  <select
                    id="domain"
                    className="select-field"
                    value={domain}
                    onChange={(event) =>
                      setDomain(event.target.value as Domain)
                    }
                  >
                    <option value="home_health">Home-health</option>
                    <option value="legal">Legal</option>
                  </select>
                </div>
              )}
            </div>

            <VoiceRecorder
              onSubmit={handleSubmit}
              disabled={loading}
              submitLabel={mode === "intake" ? "Run intake" : "Run triage"}
            />
          </div>

          <aside className="process-panel" aria-label="How it works">
            <p className="eyebrow">How it works</p>
            <ol className="process-list">
              <li>
                <span>1</span>
                <div>
                  <strong>Capture</strong>
                  <p>Voice or typed notes, age and pregnancy context.</p>
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
                  <strong>Route</strong>
                  <p>
                    Receive a structured record and a clear routing decision.
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
          <section className="analysis-card" aria-live="polite">
            <div className="analysis-header">
              <div>
                <p className="eyebrow">Processing</p>
                <h2>Preparing the intake</h2>
                <p className="progress-detail">
                  Running the safety screen and structured capture.
                </p>
              </div>
              <strong className="progress-number">
                <svg
                  viewBox="0 0 24 24"
                  width="34"
                  height="34"
                  fill="none"
                  aria-hidden="true"
                >
                  <path
                    d="M12 3a9 9 0 1 0 9 9"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                  />
                </svg>
              </strong>
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
              <h3>Could not complete the request</h3>
              <p>{error}</p>
            </div>
          </section>
        )}

        {intakeResult && (
          <section className="result-area" aria-labelledby="result-title">
            <div className="result-area-header">
              <div>
                <p className="eyebrow">Intake complete</p>
                <h2 id="result-title">Structured intake record</h2>
              </div>
              <span className="case-reference">
                Case {caseId.slice(0, 8).toUpperCase()}
              </span>
            </div>
            <IntakeResult data={intakeResult} />
          </section>
        )}

        {triageResult && (
          <section className="result-area" aria-labelledby="result-title">
            <div className="result-area-header">
              <div>
                <p className="eyebrow">Triage complete</p>
                <h2 id="result-title">Triage summary</h2>
              </div>
              <span className="case-reference">
                Case {caseId.slice(0, 8).toUpperCase()}
              </span>
            </div>
            <TriageResultView result={triageResult} />
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
          Decision support only. Always verify findings with a qualified
          professional.
        </p>
        <span>MedRoute · 2026</span>
      </footer>
    </div>
  );
}

export default App;
