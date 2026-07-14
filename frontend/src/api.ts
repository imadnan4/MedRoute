import type { TriageRequest, TriageResponse } from "./types";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

export async function transcribeAudio(audioB64: string, language?: string): Promise<string> {
  const res = await fetch(`${API_BASE}/transcribe`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    // Clinic: Urdu first, then English only
    body: JSON.stringify({ audio_b64: audioB64, language: language || "ur" }),
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`Transcription failed: ${err}`);
  }
  const data = await res.json();
  return data.text || "";
}

export async function submitTriage(data: TriageRequest): Promise<TriageResponse> {
  const res = await fetch(`${API_BASE}/triage`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`Triage failed: ${err}`);
  }
  return res.json();
}

export async function downloadReport(caseId: string): Promise<void> {
  const res = await fetch(`${API_BASE}/report/${caseId}`);
  if (!res.ok) throw new Error("Report not found");
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `medroute_triage_${caseId.slice(0, 8)}.pdf`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
