import { useRef, useState, type FormEvent } from "react";
import { transcribeAudio } from "../api";

interface VoiceRecorderProps {
  onTranscript: (text: string) => void;
  disabled?: boolean;
}

function encodeWav(samples: Float32Array, sampleRate: number): Blob {
  const blockAlign = 2;
  const dataSize = samples.length * blockAlign;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);
  const writeString = (offset: number, value: string) => {
    for (let i = 0; i < value.length; i++)
      view.setUint8(offset + i, value.charCodeAt(i));
  };

  writeString(0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * blockAlign, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 16, true);
  writeString(36, "data");
  view.setUint32(40, dataSize, true);

  let offset = 44;
  for (const sample of samples) {
    const value = Math.max(-1, Math.min(1, sample));
    view.setInt16(offset, value < 0 ? value * 0x8000 : value * 0x7fff, true);
    offset += 2;
  }
  return new Blob([buffer], { type: "audio/wav" });
}

function blobToBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      const result = reader.result as string;
      const comma = result.indexOf(",");
      resolve(comma >= 0 ? result.slice(comma + 1) : result);
    };
    reader.onerror = () =>
      reject(new Error("The audio file could not be read."));
    reader.readAsDataURL(blob);
  });
}

export default function VoiceRecorder({
  onTranscript,
  disabled,
}: VoiceRecorderProps) {
  const [recording, setRecording] = useState(false);
  const [transcribing, setTranscribing] = useState(false);
  const [textInput, setTextInput] = useState("");
  const [ageInput, setAgeInput] = useState("");
  const [pregnancyInput, setPregnancyInput] = useState("not_pregnant");
  const [feedback, setFeedback] = useState("");

  const streamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const chunksRef = useRef<Float32Array[]>([]);
  const startedAtRef = useRef(0);

  async function startRecording() {
    setFeedback("");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          channelCount: 1,
          sampleRate: 16000,
        },
      });
      streamRef.current = stream;
      const AudioContextConstructor =
        window.AudioContext ||
        (window as unknown as { webkitAudioContext: typeof AudioContext })
          .webkitAudioContext;
      const context = new AudioContextConstructor();
      audioContextRef.current = context;
      if (context.state === "suspended") await context.resume();

      const source = context.createMediaStreamSource(stream);
      const processor = context.createScriptProcessor(4096, 1, 1);
      const mute = context.createGain();
      mute.gain.value = 0;
      sourceRef.current = source;
      processorRef.current = processor;
      chunksRef.current = [];
      startedAtRef.current = performance.now();
      processor.onaudioprocess = (event) => {
        chunksRef.current.push(
          new Float32Array(event.inputBuffer.getChannelData(0)),
        );
      };
      source.connect(processor);
      processor.connect(mute);
      mute.connect(context.destination);
      setRecording(true);
    } catch {
      setFeedback(
        "Microphone access is unavailable. Check browser permissions or use typed input.",
      );
    }
  }

  async function stopRecording() {
    setRecording(false);
    const context = audioContextRef.current;
    const nativeRate = context?.sampleRate || 48000;
    try {
      processorRef.current?.disconnect();
      sourceRef.current?.disconnect();
      streamRef.current?.getTracks().forEach((track) => track.stop());
      await context?.close();
    } catch {
      // Browser audio cleanup can safely fail after capture.
    }
    processorRef.current = null;
    sourceRef.current = null;
    streamRef.current = null;
    audioContextRef.current = null;

    const chunks = chunksRef.current;
    chunksRef.current = [];
    if (!chunks.length || performance.now() - startedAtRef.current < 1500) {
      setFeedback(
        "Recording was too short. Speak clearly for at least two seconds, then stop.",
      );
      return;
    }

    const merged = new Float32Array(
      chunks.reduce((total, chunk) => total + chunk.length, 0),
    );
    let offset = 0;
    for (const chunk of chunks) {
      merged.set(chunk, offset);
      offset += chunk.length;
    }
    const peak = merged.reduce(
      (highest, sample) => Math.max(highest, Math.abs(sample)),
      0,
    );
    if (peak < 0.005) {
      setFeedback(
        "The microphone signal was very quiet. Move closer and check the input volume.",
      );
      return;
    }

    setTranscribing(true);
    setFeedback("Transcribing the recording…");
    try {
      // Preserve the browser's native sample rate; Whisper's decoder performs
      // higher-quality resampling than a simple client-side interpolation.
      const wav = encodeWav(merged, nativeRate);
      const transcript = await transcribeAudio(await blobToBase64(wav), "ur");
      if (!transcript?.trim()) {
        setFeedback(
          "No speech was detected. Try again or type the symptoms below.",
        );
        return;
      }
      setTextInput(transcript);
      setFeedback(
        "Voice transcription is ready. Review it before running triage.",
      );
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Transcription failed.";
      setFeedback(`${message} You can continue with typed input.`);
    } finally {
      setTranscribing(false);
    }
  }

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!textInput.trim()) return;
    const parts = [textInput.trim()];
    if (ageInput) parts.push(`Age: ${ageInput} years`);
    if (pregnancyInput !== "not_pregnant")
      parts.push(`Pregnancy: ${pregnancyInput}`);
    onTranscript(parts.join(". "));
  }

  return (
    <form className="intake-form" onSubmit={handleSubmit}>
      <div className="field-group">
        <div className="field-label-row">
          <label htmlFor="symptoms">Symptoms and context</label>
          <span>Urdu or English</span>
        </div>
        <textarea
          id="symptoms"
          className="textarea"
          value={textInput}
          onChange={(event) => setTextInput(event.target.value)}
          placeholder={
            transcribing
              ? "Transcribing audio…"
              : "Describe the symptoms, when they started, and anything that makes them better or worse."
          }
          rows={6}
          disabled={disabled || transcribing}
        />
      </div>

      <div className="voice-row">
        <button
          type="button"
          onClick={recording ? stopRecording : startRecording}
          disabled={disabled || transcribing}
          className={`voice-button ${recording ? "is-recording" : ""}`}
        >
          <span className="voice-icon" aria-hidden="true">
            {recording ? (
              <i />
            ) : (
              <svg viewBox="0 0 20 20" fill="none">
                <rect
                  x="7"
                  y="2.5"
                  width="6"
                  height="10"
                  rx="3"
                  stroke="currentColor"
                  strokeWidth="1.5"
                />
                <path
                  d="M4.8 9.7a5.2 5.2 0 0 0 10.4 0M10 15v2.5M7.5 17.5h5"
                  stroke="currentColor"
                  strokeWidth="1.5"
                  strokeLinecap="round"
                />
              </svg>
            )}
          </span>
          {recording
            ? "Stop recording"
            : transcribing
              ? "Transcribing"
              : "Add voice note"}
        </button>
        <span className="voice-help">
          {recording
            ? "Listening now — speak clearly"
            : "Recorded securely in this browser session"}
        </span>
      </div>

      {feedback && (
        <p className="form-feedback" role="status">
          {feedback}
        </p>
      )}

      <div className="context-grid">
        <div className="field-group compact">
          <label htmlFor="age">Age</label>
          <div className="input-with-suffix">
            <input
              id="age"
              className="input-field"
              type="number"
              min="0"
              max="120"
              placeholder="e.g. 42"
              value={ageInput}
              onChange={(event) => setAgeInput(event.target.value)}
              disabled={disabled}
            />
            <span>years</span>
          </div>
        </div>
        <div className="field-group compact">
          <label htmlFor="pregnancy">Pregnancy status</label>
          <select
            id="pregnancy"
            className="select-field"
            value={pregnancyInput}
            onChange={(event) => setPregnancyInput(event.target.value)}
            disabled={disabled}
          >
            <option value="not_pregnant">Not pregnant</option>
            <option value="pregnant">Pregnant</option>
            <option value="pregnant_3rd_trimester">
              Pregnant · third trimester
            </option>
          </select>
        </div>
      </div>

      <div className="form-actions">
        <p>Results support—not replace—clinical judgment.</p>
        <button
          className="btn btn-primary btn-large"
          type="submit"
          disabled={disabled || transcribing || !textInput.trim()}
        >
          {disabled ? "Assessing patient" : "Run triage assessment"}
          <svg viewBox="0 0 18 18" fill="none" aria-hidden="true">
            <path
              d="M3 9h12M10.5 4.5 15 9l-4.5 4.5"
              stroke="currentColor"
              strokeWidth="1.6"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </button>
      </div>
    </form>
  );
}
