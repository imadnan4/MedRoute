import type { TriageResult } from "../types";

interface TriageResultProps {
  result: TriageResult;
}

function formatValue(value: string) {
  return value.replace(/_/g, " ");
}

function confidenceClass(level: string) {
  if (level === "green") return "status-green";
  if (level === "yellow") return "status-yellow";
  return "status-red";
}

export default function TriageResultView({ result }: TriageResultProps) {
  const isRedFlag = result.route === "hard_escalation";
  const confidencePercent = Math.round(result.confidence * 100);

  return (
    <article className={`triage-result ${isRedFlag ? "has-red-flag" : ""}`}>
      {isRedFlag && (
        <div className="urgent-banner" role="alert">
          <span className="urgent-icon" aria-hidden="true">
            <svg viewBox="0 0 24 24" fill="none">
              <path
                d="M12 3 2.8 20h18.4L12 3Z"
                stroke="currentColor"
                strokeWidth="1.6"
                strokeLinejoin="round"
              />
              <path
                d="M12 9v5M12 17.2v.1"
                stroke="currentColor"
                strokeWidth="1.8"
                strokeLinecap="round"
              />
            </svg>
          </span>
          <div>
            <p className="eyebrow">Immediate action</p>
            <strong>Hard red-flag override</strong>
            <p>
              Urgent clinician referral is required. Do not delay care for
              further automated assessment.
            </p>
          </div>
        </div>
      )}

      <div className="assessment-lead">
        <div>
          <p className="eyebrow">Working assessment</p>
          <h3>{result.likely_condition}</h3>
        </div>
        <div
          className={`confidence-card ${confidenceClass(result.confidence_level)}`}
        >
          <span>Confidence</span>
          <strong>
            {confidencePercent}
            <small>%</small>
          </strong>
          <em>{result.confidence_level}</em>
        </div>
      </div>

      <dl className="patient-summary">
        <div>
          <dt>Age</dt>
          <dd>{result.patient.age_for_display}</dd>
        </div>
        <div>
          <dt>Pregnancy</dt>
          <dd>{formatValue(result.patient.pregnancy)}</dd>
        </div>
        <div>
          <dt>Urgency</dt>
          <dd className={`urgency-${result.urgency || "soon"}`}>
            {formatValue(result.urgency || "soon")}
          </dd>
        </div>
        <div>
          <dt>Assessment route</dt>
          <dd>{formatValue(result.route)}</dd>
        </div>
      </dl>

      {result.red_flag?.triggered && (
        <section className="red-flag-detail">
          <p className="eyebrow">Triggered safety rule</p>
          <h3>
            {formatValue(result.red_flag.flag_class || "Emergency pattern")}
          </h3>
          <p>{result.red_flag.message}</p>
        </section>
      )}

      <div className="clinical-grid">
        <section className="recommendation-block">
          <div className="block-heading">
            <span>01</span>
            <h3>Recommended next step</h3>
          </div>
          <p>{result.recommendation}</p>
        </section>

        {result.watch_for.length > 0 && (
          <section className="watch-block">
            <div className="block-heading">
              <span>02</span>
              <h3>Return immediately if</h3>
            </div>
            <ul>
              {result.watch_for.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          </section>
        )}
      </div>

      {result.differential.length > 0 && (
        <section className="differential-block">
          <div className="block-heading">
            <span>03</span>
            <h3>Conditions to consider</h3>
          </div>
          <div className="differential-list">
            {result.differential.map((condition, index) => (
              <div key={`${condition}-${index}`}>
                <span>{String(index + 1).padStart(2, "0")}</span>
                <p>{condition}</p>
              </div>
            ))}
          </div>
        </section>
      )}

      {result.rag_evidence?.length > 0 && (
        <section className="evidence-block">
          <div className="block-heading">
            <span>04</span>
            <div>
              <h3>Guideline evidence</h3>
              <p>Retrieved context used to ground this assessment.</p>
            </div>
          </div>
          <div className="evidence-list">
            {result.rag_evidence.map((evidence, index) => (
              <blockquote key={`${evidence}-${index}`}>
                <sup>{index + 1}</sup>
                <p>{evidence}</p>
              </blockquote>
            ))}
          </div>
        </section>
      )}

      {(result.reasoning || result.cascade_used?.length) && (
        <details className="audit-details">
          <summary>
            <span>Technical audit trail</span>
            <small>For clinical review</small>
          </summary>
          <div className="audit-content">
            {result.cascade_used && result.cascade_used.length > 0 && (
              <div>
                <h4>Decision cascade</h4>
                <p className="cascade-line">
                  {result.cascade_used.join(" → ")}
                </p>
              </div>
            )}
            {result.reasoning && (
              <div>
                <h4>System reasoning</h4>
                <pre>{result.reasoning}</pre>
              </div>
            )}
          </div>
        </details>
      )}
    </article>
  );
}
