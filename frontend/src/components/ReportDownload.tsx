import { useState } from "react";
import { downloadReport } from "../api";

interface ReportDownloadProps {
  caseId: string;
  disabled?: boolean;
}

export default function ReportDownload({
  caseId,
  disabled,
}: ReportDownloadProps) {
  const [downloading, setDownloading] = useState(false);
  const [error, setError] = useState("");

  async function handleDownload() {
    setDownloading(true);
    setError("");
    try {
      await downloadReport(caseId);
    } catch {
      setError("The report could not be downloaded. Please try again.");
    } finally {
      setDownloading(false);
    }
  }

  return (
    <div className="report-download">
      <div>
        <span className="report-icon" aria-hidden="true">
          <svg viewBox="0 0 24 24" fill="none">
            <path
              d="M6 2.8h8l4 4V21H6V2.8Z"
              stroke="currentColor"
              strokeWidth="1.5"
            />
            <path
              d="M14 2.8v4h4M9 12h6M9 15.5h6"
              stroke="currentColor"
              strokeWidth="1.5"
              strokeLinecap="round"
            />
          </svg>
        </span>
        <div>
          <h3>Clinical handoff report</h3>
          <p>
            Download a print-ready PDF with findings, evidence, and the decision
            audit trail.
          </p>
        </div>
      </div>
      <button
        className="btn btn-primary report-button"
        onClick={handleDownload}
        disabled={disabled || !caseId || downloading}
      >
        <svg viewBox="0 0 18 18" fill="none" aria-hidden="true">
          <path
            d="M9 2v10M5 8.5 9 12.5l4-4M3 15.5h12"
            stroke="currentColor"
            strokeWidth="1.6"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
        {downloading ? "Preparing report" : "Download PDF"}
      </button>
      {error && (
        <p className="download-error" role="alert">
          {error}
        </p>
      )}
    </div>
  );
}
