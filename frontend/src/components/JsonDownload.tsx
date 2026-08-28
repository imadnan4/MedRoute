import { useState } from "react";

interface JsonDownloadProps {
  data: unknown;
  fileName?: string;
}

export default function JsonDownload({ data, fileName }: JsonDownloadProps) {
  const [error, setError] = useState("");

  function handleDownload() {
    setError("");
    try {
      const json = JSON.stringify(data, null, 2);
      const blob = new Blob([json], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = fileName ?? "medroute.json";
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch {
      setError("The JSON could not be prepared. Please try again.");
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
          <h3>Structured intake record</h3>
          <p>
            Download the full machine-readable JSON for handoff to your case
            system.
          </p>
        </div>
      </div>
      <button
        className="btn btn-primary report-button"
        onClick={handleDownload}
        type="button"
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
        Download JSON
      </button>
      {error && (
        <p className="download-error" role="alert">
          {error}
        </p>
      )}
    </div>
  );
}
