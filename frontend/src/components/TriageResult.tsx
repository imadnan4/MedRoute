import type { TriageResult } from "../types";

interface TriageResultProps {
  result: TriageResult;
}

function badgeClass(level: string) {
  switch (level) {
    case "green": return "badge-green";
    case "yellow": return "badge-yellow";
    default: return "badge-red";
  }
}

export default function TriageResultView({ result }: TriageResultProps) {
  const isRedFlag = result.route === "hard_escalation";

  return (
    <div>
      {isRedFlag && (
        <div className="urgent-banner">
          <svg className="urgent-banner-icon" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M10 2L1 18h18L10 2z" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" />
            <line x1="10" y1="8" x2="10" y2="12" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
            <circle cx="10" cy="15" r="1" fill="currentColor" />
          </svg>
          Hard Red-Flag Override &mdash; Urgent clinician referral required
        </div>
      )}

      <div className={`badge badge-large ${badgeClass(result.confidence_level)}`}>
        Confidence: {result.confidence_level} ({Math.round(result.confidence * 100)}%)
      </div>

      <div className="patient-summary">
        <div className="patient-field">
          <span className="patient-field-label">Age</span>
          <span className="patient-field-value">{result.patient.age_for_display}</span>
        </div>
        <div className="patient-field">
          <span className="patient-field-label">Pregnancy</span>
          <span className="patient-field-value">{result.patient.pregnancy.replace(/_/g, " ")}</span>
        </div>
        <div className="patient-field">
          <span className="patient-field-label">Route</span>
          <span className="patient-field-value">{result.route.replace(/_/g, " ")}</span>
        </div>
        {result.urgency && (
          <div className="patient-field">
            <span className="patient-field-label">Urgency</span>
            <span className="patient-field-value">{result.urgency.replace(/_/g, " ")}</span>
          </div>
        )}
      </div>

      {result.red_flag?.triggered && (
        <div className="red-flag-detail">
          <h3>{result.red_flag.flag_class?.replace(/_/g, " ")}</h3>
          <p>{result.red_flag.message}</p>
        </div>
      )}

      <p className="result-condition">{result.likely_condition}</p>

      {result.differential.length > 0 && (
        <div className="result-block">
          <h3>Differential Diagnosis</h3>
          <ul className="result-list">
            {result.differential.map((d, i) => (
              <li key={i}>{d}</li>
            ))}
          </ul>
        </div>
      )}

      <div className="result-block">
        <h3>Recommendation</h3>
        <p>{result.recommendation}</p>
      </div>

      {result.watch_for.length > 0 && (
        <div className="result-block">
          <h3>Watch For</h3>
          <ul className="result-list">
            {result.watch_for.map((w, i) => (
              <li key={i}>{w}</li>
            ))}
          </ul>
        </div>
      )}

      {result.rag_evidence && result.rag_evidence.length > 0 && (
        <div className="result-block">
          <h3>Guideline Evidence</h3>
          <ul className="result-list">
            {result.rag_evidence.map((e, i) => (
              <li key={i}>{e.length > 220 ? `${e.slice(0, 220)}…` : e}</li>
            ))}
          </ul>
        </div>
      )}

      {result.cascade_used && result.cascade_used.length > 0 && (
        <div className="result-block">
          <h3>Decision Cascade</h3>
          <p className="result-condition" style={{ fontSize: "0.9rem", fontWeight: 400 }}>
            {result.cascade_used.join(" → ")}
          </p>
        </div>
      )}

      {result.reasoning && (
        <div className="result-block">
          <h3>Clinical Reasoning</h3>
          <pre className="result-reasoning">{result.reasoning}</pre>
        </div>
      )}
    </div>
  );
}
