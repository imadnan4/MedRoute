import { useState } from "react";
import VoiceRecorder from "./components/VoiceRecorder";
import TriageResultView from "./components/TriageResult";
import ReportDownload from "./components/ReportDownload";
import type { TriageResult } from "./types";
import { submitTriage } from "./api";
import "./App.css";

function App() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<TriageResult | null>(null);
  const [caseId, setCaseId] = useState("");

  async function handleTranscript(text: string) {
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const response = await submitTriage({ transcript: text });
      setResult(response.result);
      setCaseId(response.case_id);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Connection failed. Check that the backend is running on port 8000.";
      setError(msg);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>MedRoute</h1>
        <p className="subtitle">Medical Triage &amp; Routing Agent</p>
      </header>

      <main id="main-content">
        <section className="section">
          <VoiceRecorder onTranscript={handleTranscript} disabled={loading} />
        </section>

        {loading && (
          <section className="section loading-section">
            <div className="skeleton skeleton-line" />
            <div className="skeleton skeleton-line skeleton-line-short" />
            <div className="skeleton skeleton-line skeleton-line-short" />
            <span className="loading-label">Running triage pipeline</span>
          </section>
        )}

        {error && (
          <section className="error-section">
            <h3>Error</h3>
            <p>{error}</p>
          </section>
        )}

        {result && (
          <>
            <section className="section result-section">
              <TriageResultView result={result} />
            </section>
            <ReportDownload caseId={caseId} />
          </>
        )}
      </main>

      <footer className="app-footer">
        <p>Not a substitute for clinical judgment &middot; AMD Developer Hackathon 2026</p>
      </footer>
    </div>
  );
}

export default App;
