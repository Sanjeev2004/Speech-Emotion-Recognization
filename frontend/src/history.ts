import type { Emotion, Prediction } from "./api";

const KEY = "ser.predictionHistory";
const MAX = 20;

export interface HistoryEntry {
  id: string;
  at: string;
  source: string;
  emotion: Emotion;
  confidence: number;
  model: string;
  latency_ms: number;
  probabilities: Prediction["probabilities"];
}

export function loadHistory(): HistoryEntry[] {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as HistoryEntry[];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

export function pushHistory(entry: Omit<HistoryEntry, "id" | "at">): HistoryEntry[] {
  const next: HistoryEntry = {
    ...entry,
    id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    at: new Date().toISOString(),
  };
  const list = [next, ...loadHistory()].slice(0, MAX);
  localStorage.setItem(KEY, JSON.stringify(list));
  return list;
}

export function clearHistory(): void {
  localStorage.removeItem(KEY);
}

export function exportHistoryCsv(entries: HistoryEntry[]): string {
  const header = "timestamp,source,emotion,confidence,model,latency_ms";
  const lines = entries.map((e) =>
    [e.at, JSON.stringify(e.source), e.emotion, e.confidence.toFixed(4), e.model, e.latency_ms].join(","),
  );
  return [header, ...lines].join("\n");
}
