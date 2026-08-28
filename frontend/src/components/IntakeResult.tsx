import type { IntakeResponse } from "../types";
import JsonDownload from "./JsonDownload";

interface IntakeResultProps {
  data: IntakeResponse;
}

const DISPOSITION_META: Record<
  IntakeResponse["disposition"],
  { label: string; className: string }
> = {
  escalate_to_clinician: {
    label: "⚠ Escalate to human now",
    className: "route-escalate",
  },
  standard_queue: { label: "Standard queue", className: "route-standard" },
  provide_guidance: { label: "Provide guidance", className: "route-guidance" },
};

const RECORD_ROWS: { key: keyof IntakeResponse["extracted"]; label: string }[] =
  [
    { key: "contact_name", label: "Contact name" },
    { key: "care_recipient_name", label: "Care recipient" },
    { key: "phone_or_contact", label: "Phone / contact" },
    { key: "care_need_summary", label: "Care need summary" },
    { key: "condition_or_issue", label: "Condition / issue" },
    { key: "mobility_or_severity_notes", label: "Mobility / severity" },
    { key: "insurance_or_payment_notes", label: "Insurance / payment" },
    { key: "preferred_availability", label: "Preferred availability" },
    { key: "free_text_summary", label: "Summary" },
  ];

export default function IntakeResult({ data }: IntakeResultProps) {
  const routing = DISPOSITION_META[data.disposition];
  const redFlag = data.red_flag;
  const distress = data.distress;

  return (
    <article className="intake-result">
      {redFlag?.triggered && (
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
            <p className="eyebrow">Triggered safety rule</p>
            <strong>
              {redFlag.flag_class
                ? redFlag.flag_class.replace(/_/g, " ")
                : "Emergency pattern"}
            </strong>
            <p>{redFlag.message}</p>
            {redFlag.matched_symptoms &&
              redFlag.matched_symptoms.length > 0 && (
                <p className="matched-note">
                  Matched: {redFlag.matched_symptoms.join(", ")}
                </p>
              )}
          </div>
        </div>
      )}

      {distress?.triggered && (
        <div className="urgent-banner distress-banner" role="alert">
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
            <p className="eyebrow">Caller distress / safeguarding</p>
            <strong>
              {distress.klass ? distress.klass.replace(/_/g, " ") : "Distress signal"}
            </strong>
            <p>{distress.reason}</p>
            {distress.matched && distress.matched.length > 0 && (
              <p className="matched-note">Matched: {distress.matched.join(", ")}</p>
            )}
          </div>
        </div>
      )}

      <div className="routing-section">
        <div>
          <p className="eyebrow">Routing decision</p>
          <h3>{routing.label}</h3>
          <p className="routing-sub">
            {data.needs_human_review
              ? "Flagged for human review."
              : "No human review required."}
          </p>
        </div>
        <span className={`routing-badge ${routing.className}`}>
          {data.disposition.replace(/_/g, " ")}
        </span>
      </div>

      <section className="record-block">
        <div className="block-heading">
          <span>01</span>
          <h3>Raw transcript</h3>
        </div>
        <p className="raw-transcript">{data.raw_transcript}</p>
      </section>

      <section className="record-block">
        <div className="block-heading">
          <span>02</span>
          <h3>Structured record</h3>
        </div>
        <dl className="record-grid">
          {RECORD_ROWS.map(({ key, label }) => {
            const value = data.extracted[key];
            if (value === null || value === undefined || value === "")
              return null;
            return (
              <div key={key} className="record-row">
                <dt>{label}</dt>
                <dd>{value}</dd>
              </div>
            );
          })}
        </dl>
      </section>

      <JsonDownload
        text={buildIntakeSheet(data)}
        fileName={`intake_${data.case_id.slice(0, 8)}.txt`}
        title="Intake sheet"
        description="Download a readable intake sheet for the care team or family."
        buttonLabel="Download intake sheet"
      />
    </article>
  );
}

function buildIntakeSheet(data: IntakeResponse): string {
  const lines: string[] = [];
  lines.push("MEDROUTE — INTAKE SHEET");
  lines.push(`Case: ${data.case_id.slice(0, 8).toUpperCase()}`);
  lines.push(`Routing: ${data.disposition.replace(/_/g, " ")}`);
  if (data.red_flag) {
    lines.push(`RED FLAG: ${data.red_flag.message}`);
  }
  if (data.distress && data.distress.matched && data.distress.matched.length) {
    lines.push(`Distress flags: ${data.distress.matched.join(", ")}`);
  }
  lines.push("");
  lines.push("Transcript:");
  lines.push(data.raw_transcript || "(none)");
  lines.push("");
  lines.push("Extracted record:");
  for (const { key, label } of RECORD_ROWS) {
    const value = data.extracted[key];
    if (value === null || value === undefined || value === "") continue;
    lines.push(`- ${label}: ${value}`);
  }
  return lines.join("\n");
}
