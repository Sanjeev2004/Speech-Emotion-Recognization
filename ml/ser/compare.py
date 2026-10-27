"""Collect every runs/*/result.json into one leaderboard + charts.

    python -m ser.compare --runs runs --out reports

Writes:
    reports/leaderboard.csv        one row per (model, train dataset, test dataset)
    reports/leaderboard.md         readable tables for the README
    reports/leaderboard.json       consumed by the backend /api/results and the frontend
    reports/uar_by_model.png       bar chart, same-dataset test UAR
    reports/cross_dataset.png      heat-map: train on X, test on Y
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load_results(runs_dir: str | Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(Path(runs_dir).glob("*/result.json"))]


def to_table(results: list[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        for test_ds, m in r["test"].items():
            rows.append({
                "model": r["model"], "kind": r["kind"], "train_dataset": r["train_dataset"],
                "test_dataset": test_ds, "accuracy": m["accuracy"], "uar": m["uar"],
                "macro_f1": m["macro_f1"], "n_test": m["n"], "n_params": r.get("n_params"),
                "train_seconds": r.get("train_seconds"), "ms_per_clip": m.get("ms_per_clip"),
            })
    return pd.DataFrame(rows)


def build_report(runs_dir: str | Path = "runs", out_dir: str | Path = "reports") -> pd.DataFrame:
    results = load_results(runs_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not results:
        print("[compare] no results yet")
        return pd.DataFrame()
    df = to_table(results)
    df.to_csv(out / "leaderboard.csv", index=False)

    same = df[df["train_dataset"] == df["test_dataset"]].copy()
    md = ["# Model comparison", "",
          "Speaker-independent test sets. UAR = unweighted average recall (main metric).", ""]
    for ds, sub in same.groupby("train_dataset"):
        sub = sub.sort_values("uar", ascending=False)
        md += [f"## Trained and tested on {ds}", "",
               "| Rank | Model | Accuracy | UAR | Macro-F1 | Params |", "|---|---|---|---|---|---|"]
        for i, r in enumerate(sub.itertuples(), 1):
            params = f"{r.n_params / 1e6:.2f}M" if pd.notna(r.n_params) else "-"
            md.append(f"| {i} | {r.model} | {r.accuracy:.3f} | {r.uar:.3f} | {r.macro_f1:.3f} | {params} |")
        md.append("")
    cross = df[df["train_dataset"] != df["test_dataset"]]
    if len(cross):
        piv = cross.pivot_table(index="model", columns=["train_dataset", "test_dataset"], values="uar")
        try:
            table = piv.round(3).to_markdown()
        except ImportError:  # 'tabulate' not installed
            table = "```\n" + piv.round(3).to_string() + "\n```"
        md += ["## Cross-dataset UAR (train -> test)", "", table, ""]
    (out / "leaderboard.md").write_text("\n".join(md))

    payload = {
        "metric_notes": {"accuracy": "share of clips predicted correctly",
                         "uar": "average recall over the 6 emotions (robust to class imbalance)",
                         "macro_f1": "average F1 over the 6 emotions"},
        "rows": json.loads(df.to_json(orient="records")),
        "runs": [{k: r[k] for k in ("model", "kind", "description", "train_dataset", "test", "n_params",
                                     "train_seconds") if k in r} for r in results],
    }
    (out / "leaderboard.json").write_text(json.dumps(payload, indent=2))
    _plots(df, out)
    print(f"[compare] {len(results)} runs -> {out}/leaderboard.(csv|md|json)")
    print(same.sort_values(["train_dataset", "uar"], ascending=[True, False])
          [["train_dataset", "model", "accuracy", "uar", "macro_f1"]].to_string(index=False))
    return df


def _plots(df: pd.DataFrame, out: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    same = df[df["train_dataset"] == df["test_dataset"]]
    piv = same.pivot_table(index="model", columns="train_dataset", values="uar")
    piv = piv.loc[piv.mean(axis=1).sort_values().index]
    ax = piv.plot(kind="barh", figsize=(9, 0.5 * len(piv) + 2), width=0.8)
    ax.set_xlabel("Test UAR (speaker-independent)")
    ax.set_xlim(0, 1)
    ax.set_title("Model comparison")
    plt.tight_layout()
    plt.savefig(out / "uar_by_model.png", dpi=140)
    plt.close()

    cross = df.pivot_table(index=["model", "train_dataset"], columns="test_dataset", values="uar")
    if cross.shape[1] > 1:
        fig, ax = plt.subplots(figsize=(6, 0.35 * len(cross) + 1.5))
        im = ax.imshow(cross.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(cross.shape[1]), cross.columns)
        ax.set_yticks(range(len(cross)), [f"{m} | train {d}" for m, d in cross.index])
        for i in range(cross.shape[0]):
            for j in range(cross.shape[1]):
                v = cross.values[i, j]
                if pd.notna(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if v > 0.6 else "black")
        ax.set_xlabel("test dataset")
        fig.colorbar(im, ax=ax, label="UAR")
        plt.tight_layout()
        plt.savefig(out / "cross_dataset.png", dpi=140)
        plt.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs")
    p.add_argument("--out", default="reports")
    a = p.parse_args()
    build_report(a.runs, a.out)
