import { useEffect, useRef, useState } from "react";
import { api, EMOTIONS, ModelMeta, Prediction } from "../api";
import { MicRecorder, toWav16k } from "../audio";

const MAX_RECORD_SECONDS = 10;
const pct = (x: number) => `${(x * 100).toFixed(1)}%`;

export default function Analyze() {
  const [models, setModels] = useState<ModelMeta[]>([]);
  const [modelId, setModelId] = useState<string>("");
  const [serverMsg, setServerMsg] = useState("Connecting to server...");
  const [audio, setAudio] = useState<{ wav: Blob; url: string; seconds: number; name: string } | null>(null);
  const [recording, setRecording] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [busy, setBusy] = useState(false);
  const [slow, setSlow] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<Prediction | null>(null);
  const rec = useRef(new MicRecorder());
  const timer = useRef<number>();

  useEffect(() => {
    // Render free tier sleeps after 15 min idle; the first request can take ~30-50 s.
    const t = window.setTimeout(() => setServerMsg("Server is waking up (free tier), this can take up to a minute..."), 3000);
    api
      .models()
      .then((r) => {
        setModels(r.models);
        setModelId(r.default ?? r.models[0]?.id ?? "");
        setServerMsg(r.models.length ? "" : "Server is running but no model is loaded yet.");
      })
      .catch(() => setServerMsg("Cannot reach the backend. Check VITE_API_URL or start it locally."))
      .finally(() => clearTimeout(t));
    return () => clearTimeout(t);
  }, []);

  async function useBlob(blob: Blob, name: string) {
    setError("");
    setResult(null);
    try {
      const { wav, seconds } = await toWav16k(blob);
      if (audio) URL.revokeObjectURL(audio.url);
      setAudio({ wav, url: URL.createObjectURL(wav), seconds, name });
    } catch {
      setError("This file could not be decoded by the browser. Try WAV or MP3.");
    }
  }

  async function startRecording() {
    setError("");
    try {
      await rec.current.start();
    } catch {
      setError("Microphone permission was denied.");
      return;
    }
    setRecording(true);
    setElapsed(0);
    const t0 = Date.now();
    timer.current = window.setInterval(() => {
      const s = (Date.now() - t0) / 1000;
      setElapsed(s);
      if (s >= MAX_RECORD_SECONDS) stopRecording();
    }, 100);
  }

  async function stopRecording() {
    clearInterval(timer.current);
    setRecording(false);
    const blob = await rec.current.stop();
    if (blob.size) await useBlob(blob, "Microphone recording");
  }

  async function analyze() {
    if (!audio) return;
    setBusy(true);
    setError("");
    const t = window.setTimeout(() => setSlow(true), 4000);
    try {
      setResult(await api.predict(audio.wav, modelId || undefined));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      clearTimeout(t);
      setSlow(false);
      setBusy(false);
    }
  }

  const current = models.find((m) => m.id === modelId);

  return (
    <div className="grid">
      <section className="card">
        <h2>1. Give some speech</h2>
        <p className="muted">Record up to {MAX_RECORD_SECONDS} seconds, or upload an audio file. Short, clearly spoken sentences work best.</p>

        <div className="row">
          {!recording ? (
            <button className="btn primary" onClick={startRecording} disabled={busy}>
              Record
            </button>
          ) : (
            <button className="btn danger" onClick={stopRecording}>
              Stop ({elapsed.toFixed(1)} s)
            </button>
          )}
          <label className="btn">
            Upload file
            <input
              type="file"
              accept="audio/*"
              hidden
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) useBlob(f, f.name);
                e.target.value = "";
              }}
            />
          </label>
        </div>
        {recording && (
          <div className="meter">
            <div style={{ width: `${(elapsed / MAX_RECORD_SECONDS) * 100}%` }} />
          </div>
        )}

        {audio && (
          <div className="clip">
            <div className="muted small">
              {audio.name} · {audio.seconds.toFixed(1)} s
            </div>
            <audio controls src={audio.url} />
          </div>
        )}

        <h2>2. Choose a model</h2>
        {serverMsg && <p className="notice">{serverMsg}</p>}
        <select value={modelId} onChange={(e) => setModelId(e.target.value)} disabled={!models.length}>
          {models.map((m) => (
            <option key={m.id} value={m.id}>
              {m.model} (trained on {m.train_dataset})
            </option>
          ))}
        </select>
        {current && (
          <p className="muted small">
            {current.description}
            {current.test_uar != null && <> · test UAR {pct(current.test_uar)}</>}
            {current.size_mb != null && <> · {current.size_mb} MB</>}
          </p>
        )}

        <button className="btn primary wide" onClick={analyze} disabled={!audio || busy || !models.length}>
          {busy ? "Analyzing..." : "Detect emotion"}
        </button>
        {slow && <p className="notice">Still working: the server may be waking up.</p>}
        {error && <p className="error">{error}</p>}
      </section>

      <section className="card">
        <h2>Result</h2>
        {!result ? (
          <p className="muted">The predicted emotion and the probability for each of the 6 classes will show here.</p>
        ) : (
          <>
            <div className="verdict">
              <span className={`dot e-${result.emotion}`} />
              <div>
                <div className="verdict-label">{result.emotion}</div>
                <div className="muted">{pct(result.confidence)} confidence</div>
              </div>
            </div>
            <ul className="bars">
              {EMOTIONS.map((e) => (
                <li key={e}>
                  <span className="bar-name">{e}</span>
                  <span className="bar-track">
                    <span className={`bar-fill e-${e}`} style={{ width: pct(result.probabilities[e]) }} />
                  </span>
                  <span className="bar-val">{pct(result.probabilities[e])}</span>
                </li>
              ))}
            </ul>
            <p className="muted small">
              Model {result.model} · {result.duration_seconds} s audio in {result.windows} window(s) · {result.latency_ms} ms on server
            </p>
          </>
        )}
      </section>
    </div>
  );
}
