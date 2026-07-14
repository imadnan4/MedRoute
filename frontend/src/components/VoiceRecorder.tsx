import { useState, useRef } from "react";
import { transcribeAudio } from "../api";

interface VoiceRecorderProps {
  onTranscript: (text: string) => void;
  disabled?: boolean;
}

/** Encode Float32 mono PCM as 16-bit little-endian WAV. */
function encodeWav(samples: Float32Array, sampleRate: number): Blob {
  const numChannels = 1;
  const bitsPerSample = 16;
  const blockAlign = (numChannels * bitsPerSample) / 8;
  const byteRate = sampleRate * blockAlign;
  const dataSize = samples.length * blockAlign;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  const writeStr = (offset: number, s: string) => {
    for (let i = 0; i < s.length; i++) view.setUint8(offset + i, s.charCodeAt(i));
  };

  writeStr(0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true); // PCM chunk size
  view.setUint16(20, 1, true); // PCM format
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bitsPerSample, true);
  writeStr(36, "data");
  view.setUint32(40, dataSize, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    offset += 2;
  }
  return new Blob([buffer], { type: "audio/wav" });
}

/** Downsample Float32 audio to target rate with simple linear interpolation. */
function downsample(buffer: Float32Array, fromRate: number, toRate: number): Float32Array {
  if (fromRate === toRate) return buffer;
  const ratio = fromRate / toRate;
  const newLen = Math.max(1, Math.round(buffer.length / ratio));
  const result = new Float32Array(newLen);
  for (let i = 0; i < newLen; i++) {
    const src = i * ratio;
    const i0 = Math.floor(src);
    const i1 = Math.min(i0 + 1, buffer.length - 1);
    const t = src - i0;
    result[i] = buffer[i0] * (1 - t) + buffer[i1] * t;
  }
  return result;
}

function blobToBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      const result = reader.result as string;
      const comma = result.indexOf(",");
      resolve(comma >= 0 ? result.slice(comma + 1) : result);
    };
    reader.onerror = () => reject(new Error("Failed to read audio"));
    reader.readAsDataURL(blob);
  });
}

export default function VoiceRecorder({ onTranscript, disabled }: VoiceRecorderProps) {
  const [recording, setRecording] = useState(false);
  const [transcribing, setTranscribing] = useState(false);
  const [textInput, setTextInput] = useState("");
  const [ageInput, setAgeInput] = useState("");
  const [pregnancyInput, setPregnancyInput] = useState("not_pregnant");

  const streamRef = useRef<MediaStream | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const chunksRef = useRef<Float32Array[]>([]);
  const startedAtRef = useRef<number>(0);

  async function startRecording() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          channelCount: 1,
          // Prefer 16 kHz when the browser honors it; we resample anyway
          sampleRate: 16000,
        },
      });
      streamRef.current = stream;

      const AudioCtx =
        window.AudioContext ||
        (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      const ctx = new AudioCtx();
      audioCtxRef.current = ctx;
      if (ctx.state === "suspended") {
        await ctx.resume();
      }

      const source = ctx.createMediaStreamSource(stream);
      sourceRef.current = source;
      // ScriptProcessor is deprecated but widely supported; buffer size 4096 is fine for offline capture
      const processor = ctx.createScriptProcessor(4096, 1, 1);
      processorRef.current = processor;
      chunksRef.current = [];
      startedAtRef.current = performance.now();

      processor.onaudioprocess = (e) => {
        const input = e.inputBuffer.getChannelData(0);
        chunksRef.current.push(new Float32Array(input));
      };

      source.connect(processor);
      // Must connect to destination for some browsers to fire onaudioprocess; keep gain 0
      const mute = ctx.createGain();
      mute.gain.value = 0;
      processor.connect(mute);
      mute.connect(ctx.destination);

      setRecording(true);
    } catch {
      alert("Microphone access denied. Use text input instead.");
    }
  }

  async function stopRecording() {
    setRecording(false);

    const ctx = audioCtxRef.current;
    const processor = processorRef.current;
    const source = sourceRef.current;
    const stream = streamRef.current;
    const nativeRate = ctx?.sampleRate || 48000;

    try {
      processor?.disconnect();
      source?.disconnect();
    } catch {
      /* ignore */
    }
    stream?.getTracks().forEach((t) => t.stop());
    try {
      await ctx?.close();
    } catch {
      /* ignore */
    }
    processorRef.current = null;
    sourceRef.current = null;
    streamRef.current = null;
    audioCtxRef.current = null;

    const parts = chunksRef.current;
    chunksRef.current = [];
    const elapsedMs = performance.now() - startedAtRef.current;

    if (!parts.length || elapsedMs < 400) {
      alert("Recording too short. Hold the button and speak for at least 2–3 seconds.");
      return;
    }

    const total = parts.reduce((n, p) => n + p.length, 0);
    const merged = new Float32Array(total);
    let offset = 0;
    for (const p of parts) {
      merged.set(p, offset);
      offset += p.length;
    }

    // Peak check — warn early if mic captured near-silence
    let peak = 0;
    for (let i = 0; i < merged.length; i++) {
      const a = Math.abs(merged[i]);
      if (a > peak) peak = a;
    }
    if (peak < 0.005) {
      alert(
        "Microphone signal is very quiet. Check OS mic permissions/volume, speak closer, then try again."
      );
      return;
    }

    const pcm16k = downsample(merged, nativeRate, 16000);
    const wav = encodeWav(pcm16k, 16000);

    setTranscribing(true);
    try {
      const b64 = await blobToBase64(wav);
      // Backend priority: Urdu → English only
      const transcript = await transcribeAudio(b64, "ur");
      if (transcript && transcript.trim()) {
        setTextInput(transcript);
      } else {
        alert(
          "No speech detected. Speak clearly for 2–3+ seconds in Urdu or English, or type your symptoms."
        );
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Transcription failed";
      alert(`Voice transcription failed: ${msg}. Use text input instead.`);
    } finally {
      setTranscribing(false);
    }
  }

  function handleSubmit() {
    const parts = [textInput];
    if (ageInput) parts.push(`Age: ${ageInput} years`);
    if (pregnancyInput && pregnancyInput !== "not_pregnant") {
      parts.push(`Pregnancy: ${pregnancyInput}`);
    }
    onTranscript(parts.join(". "));
  }

  return (
    <div>
      <textarea
        className="textarea"
        value={textInput}
        onChange={(e) => setTextInput(e.target.value)}
        placeholder={
          transcribing
            ? "Transcribing audio via Nemotron 3.5 ASR..."
            : "Describe symptoms in Urdu or English..."
        }
        rows={3}
        disabled={disabled || transcribing}
      />

      <div className="input-row">
        <input
          className="input-field"
          type="number"
          placeholder="Age (years)"
          value={ageInput}
          onChange={(e) => setAgeInput(e.target.value)}
          disabled={disabled}
        />
        <select
          className="select-field"
          value={pregnancyInput}
          onChange={(e) => setPregnancyInput(e.target.value)}
          disabled={disabled}
        >
          <option value="not_pregnant">Not pregnant</option>
          <option value="pregnant">Pregnant</option>
          <option value="pregnant_3rd_trimester">Pregnant (3rd trimester)</option>
        </select>
        <button
          className="btn btn-primary"
          onClick={handleSubmit}
          disabled={disabled || !textInput.trim()}
        >
          Submit
        </button>
      </div>

      <div style={{ marginTop: "8px" }}>
        <button
          onClick={recording ? stopRecording : startRecording}
          disabled={disabled || transcribing}
          className={`btn ${recording ? "btn-recording" : "btn-outline"}`}
        >
          {recording ? "Stop Recording" : transcribing ? "Transcribing..." : "Record Voice"}
        </button>
        {recording && (
          <span style={{ marginLeft: 10, fontSize: 13, opacity: 0.7 }}>
            Listening… speak clearly, then stop
          </span>
        )}
      </div>
    </div>
  );
}
