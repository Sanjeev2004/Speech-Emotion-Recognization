"""Find audio files, read labels from file names, make speaker-independent splits.

Speaker-independent split = the same actor never appears in both train and test.
The old notebook used a random split, so the model could "memorise voices" and the test
score did not reflect real-world use. Here split is done by actor id.
"""
from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

from .labels import CREMAD_CODES, EMOTION_TO_ID, RAVDESS_CODES

DATASETS = ["ravdess", "cremad"]


def scan_ravdess(root: str | Path) -> pd.DataFrame:
    """RAVDESS name: 03-01-05-01-02-01-12.wav
    modality-channel-EMOTION-intensity-statement-repetition-ACTOR"""
    rows = []
    for p in sorted(Path(root).rglob("*.wav")):
        parts = p.stem.split("-")
        if len(parts) != 7:
            continue
        modality, channel, emo, *_rest, actor = parts
        if modality != "03" or channel != "01":   # audio-only speech
            continue
        if emo not in RAVDESS_CODES:              # drop calm / surprised
            continue
        rows.append({"path": str(p), "emotion": RAVDESS_CODES[emo],
                     "speaker": f"ravdess_{int(actor):02d}", "dataset": "ravdess"})
    return pd.DataFrame(rows)


def scan_cremad(root: str | Path) -> pd.DataFrame:
    """CREMA-D name: 1001_DFA_ANG_XX.wav  -> ACTOR_sentence_EMOTION_level"""
    rows = []
    for p in sorted(Path(root).rglob("*.wav")):
        parts = p.stem.split("_")
        if len(parts) < 3 or parts[2] not in CREMAD_CODES:
            continue
        rows.append({"path": str(p), "emotion": CREMAD_CODES[parts[2]],
                     "speaker": f"cremad_{parts[0]}", "dataset": "cremad"})
    return pd.DataFrame(rows)


SCANNERS = {"ravdess": scan_ravdess, "cremad": scan_cremad}


def speaker_split(df: pd.DataFrame, val_frac: float = 0.15, test_frac: float = 0.15,
                  seed: int = 42) -> pd.DataFrame:
    """Assign split column per speaker (train / val / test), separately per dataset."""
    df = df.copy()
    df["split"] = "train"
    for _, sub in df.groupby("dataset"):
        speakers = sorted(sub["speaker"].unique())
        random.Random(seed).shuffle(speakers)
        n = len(speakers)
        n_test = max(1, round(n * test_frac))
        n_val = max(1, round(n * val_frac))
        test_s = set(speakers[:n_test])
        val_s = set(speakers[n_test:n_test + n_val])
        df.loc[sub.index[sub["speaker"].isin(test_s)], "split"] = "test"
        df.loc[sub.index[sub["speaker"].isin(val_s)], "split"] = "val"
    return df


def build_manifest(ravdess_dir: str | None = None, cremad_dir: str | None = None,
                   seed: int = 42) -> pd.DataFrame:
    """One table of all usable clips: path, emotion, label, speaker, dataset, split."""
    frames = []
    if ravdess_dir:
        frames.append(scan_ravdess(ravdess_dir))
    if cremad_dir:
        frames.append(scan_cremad(cremad_dir))
    frames = [f for f in frames if len(f)]
    if not frames:
        raise FileNotFoundError("No audio found. Check --ravdess-dir / --cremad-dir.")
    df = pd.concat(frames, ignore_index=True)
    df["label"] = df["emotion"].map(EMOTION_TO_ID)
    return speaker_split(df, seed=seed)


def select(df: pd.DataFrame, dataset: str, split: str) -> pd.DataFrame:
    """dataset = 'ravdess' | 'cremad' | 'combined'."""
    out = df[df["split"] == split]
    if dataset != "combined":
        out = out[out["dataset"] == dataset]
    return out.reset_index(drop=True)
