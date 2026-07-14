import { downloadReport } from "../api";

interface ReportDownloadProps {
  caseId: string;
  disabled?: boolean;
}

export default function ReportDownload({ caseId, disabled }: ReportDownloadProps) {
  async function handleDownload() {
    try {
      await downloadReport(caseId);
    } catch {
      alert("Failed to download report");
    }
  }

  return (
    <div className="report-download">
      <button
        className="btn btn-primary"
        onClick={handleDownload}
        disabled={disabled || !caseId}
      >
        <svg width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M8 1v10M4 7l4 4 4-4M2 13h12" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
        Download PDF Report
      </button>
    </div>
  );
}
