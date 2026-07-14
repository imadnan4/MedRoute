import type { PipelineEvent, TriageRequest, TriageResponse } from "./types";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

export async function transcribeAudio(
  audioB64: string,
  language?: string,
): Promise<string> {
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

export async function streamTriage(
  data: TriageRequest,
  onEvent: (event: PipelineEvent) => void,
): Promise<TriageResponse> {
  const params = new URLSearchParams({
    transcript: data.transcript,
    language: data.language || "ur",
  });
  if (data.age_years != null) params.set("age_years", String(data.age_years));
  if (data.age_months != null)
    params.set("age_months", String(data.age_months));
  if (data.pregnancy) params.set("pregnancy", data.pregnancy);

  const response = await fetch(
    `${API_BASE}/triage/stream?${params.toString()}`,
    {
      headers: { Accept: "text/event-stream" },
    },
  );
  if (!response.ok || !response.body) {
    throw new Error(`Triage stream failed: ${await response.text()}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finalResponse: TriageResponse | null = null;

  function processMessage(message: string) {
    const dataLine = message
      .split(/\r?\n/)
      .find((line) => line.startsWith("data:"));
    if (!dataLine) return;

    const event = JSON.parse(dataLine.slice(5).trimStart()) as PipelineEvent;
    onEvent(event);
    if (event.stage === "error") {
      throw new Error(String(event.data.message || "Triage pipeline failed"));
    }
    if (event.stage === "done") {
      finalResponse = {
        case_id: String(event.data.case_id),
        result: event.data.result as unknown as TriageResponse["result"],
      };
    }
  }

  while (true) {
    const { value, done } = await reader.read();
    if (value) buffer += decoder.decode(value, { stream: !done });
    if (done) buffer += decoder.decode();

    const messages = buffer.split(/\r?\n\r?\n/);
    buffer = messages.pop() || "";
    messages.forEach(processMessage);

    if (done) {
      if (buffer.trim()) processMessage(buffer);
      break;
    }
  }

  if (!finalResponse)
    throw new Error("Triage stream ended before returning a result");
  return finalResponse;
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
