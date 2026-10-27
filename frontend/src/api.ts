// Thin client for the FastAPI backend.
const BASE = (import.meta.env.VITE_API_URL ?? "").replace(/\/$/, "");

export const EMOTIONS = ["neutral", "happy", "sad", "angry", "fear", "disgust"] as const;
export type Emotion = (typeof EMOTIONS)[number];

export interface ModelMeta {
  id: string;
  model: string;
  description: string;
  train_dataset: string;
  input: string;
  n_params: number | null;
  test_accuracy: number | null;
  test_uar: number | null;
  size_mb?: number;
}

export interface Prediction {
  model: string;
  emotion: Emotion;
  confidence: number;
  probabilities: Record<Emotion, number>;
  duration_seconds: number;
  windows: number;
  latency_ms: number;
}

export interface LeaderboardRow {
  model: string;
  kind: string;
  train_dataset: string;
  test_dataset: string;
  accuracy: number;
  uar: number;
  macro_f1: number;
  n_test: number;
  n_params: number | null;
  train_seconds: number | null;
  ms_per_clip: number | null;
}

export interface Leaderboard {
  rows: LeaderboardRow[];
  metric_notes?: Record<string, string>;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, init);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {
      /* not JSON */
    }
    throw new Error(`${res.status}: ${detail}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => request<{ status: string; models_loaded: number }>("/api/health"),
  models: () => request<{ default: string | null; models: ModelMeta[] }>("/api/models"),
  results: () => request<Leaderboard>("/api/results"),
  predict: (wav: Blob, model?: string) => {
    const form = new FormData();
    form.append("file", wav, "audio.wav");
    const q = model ? `?model=${encodeURIComponent(model)}` : "";
    return request<Prediction>(`/api/predict${q}`, { method: "POST", body: form });
  },
};
