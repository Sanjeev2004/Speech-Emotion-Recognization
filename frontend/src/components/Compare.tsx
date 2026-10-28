import { useEffect, useMemo, useState } from "react";
import { api, Leaderboard, LeaderboardRow } from "../api";

type Metric = "uar" | "accuracy" | "macro_f1";
const METRICS: { key: Metric; label: string }[] = [
  { key: "uar", label: "UAR" },
  { key: "accuracy", label: "Accuracy" },
  { key: "macro_f1", label: "Macro-F1" },
];
const fmt = (x: number | null | undefined, d = 3) => (x == null ? "-" : x.toFixed(d));
const params = (n: number | null) => (n == null ? "-" : `${(n / 1e6).toFixed(2)}M`);

export default function Compare() {
  const [data, setData] = useState<Leaderboard | null>(null);
  const [error, setError] = useState("");
  const [dataset, setDataset] = useState("");
  const [metric, setMetric] = useState<Metric>("uar");

  useEffect(() => {
    api
      .results()
      .then((d) => {
        setData(d);
        const ds = [...new Set(d.rows.map((r) => r.train_dataset))];
        setDataset(ds.includes("combined") ? "combined" : ds[0] ?? "");
      })
      .catch((e) => setError(e.message));
  }, []);

  const datasets = useMemo(() => [...new Set(data?.rows.map((r) => r.train_dataset) ?? [])].sort(), [data]);
  const rows = useMemo(
    () =>
      (data?.rows ?? [])
        .filter((r) => r.train_dataset === dataset && r.test_dataset === dataset)
        .sort((a, b) => b[metric] - a[metric]),
    [data, dataset, metric],
  );
  const cross = useMemo(() => crossTable(data?.rows ?? []), [data]);

  if (error)
    return (
      <section className="card">
        <h2>Model comparison</h2>
        <p className="muted">No results yet ({error}). Train the models with the Kaggle notebook, then copy <code>leaderboard.json</code> into <code>backend/models/</code>.</p>
      </section>
    );
  if (!data) return <section className="card muted">Loading results...</section>;

  const best = rows[0]?.[metric] ?? 1;

  return (
    <>
      <section className="card">
        <div className="row spread">
          <h2>Model comparison</h2>
          <div className="row">
            <div className="seg">
              {datasets.map((d) => (
                <button key={d} className={d === dataset ? "on" : ""} onClick={() => setDataset(d)}>
                  {d}
                </button>
              ))}
            </div>
            <div className="seg">
              {METRICS.map((m) => (
                <button key={m.key} className={m.key === metric ? "on" : ""} onClick={() => setMetric(m.key)}>
                  {m.label}
                </button>
              ))}
            </div>
          </div>
        </div>
        <p className="muted small">
          Trained and tested on <b>{dataset}</b>. Test speakers are never seen in training. UAR = average recall over the 6
          emotions, the standard metric for this task.
        </p>
        <ul className="bars">
          {rows.map((r) => (
            <li key={r.model}>
              <span className="bar-name plain" title={r.model}>{r.model}</span>
              <span className="bar-track">
                <span className={`bar-fill k-${r.kind}`} style={{ width: `${r[metric] * 100}%` }} />
              </span>
              <span className="bar-val">{fmt(r[metric])}</span>
            </li>
          ))}
        </ul>
        <div className="legend small">
          <span><i className="k-classical" /> classical ML</span>
          <span><i className="k-deep" /> spectrogram deep learning</span>
          <span><i className="k-ssl" /> self-supervised transformer</span>
        </div>
      </section>

      <section className="card">
        <h2>Details</h2>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>#</th><th>Model</th><th>Accuracy</th><th>UAR</th><th>Macro-F1</th><th>Params</th><th>Train time</th><th>ms / clip</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r, i) => (
                <tr key={r.model} className={r[metric] === best ? "best" : ""}>
                  <td>{i + 1}</td>
                  <td>{r.model}</td>
                  <td>{fmt(r.accuracy)}</td>
                  <td>{fmt(r.uar)}</td>
                  <td>{fmt(r.macro_f1)}</td>
                  <td>{params(r.n_params)}</td>
                  <td>{r.train_seconds == null ? "-" : `${Math.round(r.train_seconds / 60)} min`}</td>
                  <td>{fmt(r.ms_per_clip, 1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {cross.cols.length > 0 && (
        <section className="card">
          <h2>Cross-dataset generalisation (UAR)</h2>
          <p className="muted small">Train on one dataset, test on another. A big drop means the model learned dataset-specific recording conditions instead of emotion.</p>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Model</th>
                  {cross.cols.map((c) => (
                    <th key={c}>{c.replace("|", " → ")}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {cross.models.map((m) => (
                  <tr key={m}>
                    <td>{m}</td>
                    {cross.cols.map((c) => {
                      const v = cross.get(m, c);
                      return (
                        <td key={c} className="heat" style={{ ["--v" as string]: v ?? 0 }}>
                          {fmt(v)}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </>
  );
}

function crossTable(rows: LeaderboardRow[]) {
  const pairs = rows.filter((r) => r.train_dataset !== r.test_dataset && r.test_dataset !== "combined" && r.train_dataset !== "combined");
  const cols = [...new Set(pairs.map((r) => `${r.train_dataset}|${r.test_dataset}`))].sort();
  const models = [...new Set(pairs.map((r) => r.model))].sort();
  const map = new Map(pairs.map((r) => [`${r.model}#${r.train_dataset}|${r.test_dataset}`, r.uar]));
  return { cols, models, get: (m: string, c: string) => map.get(`${m}#${c}`) };
}
