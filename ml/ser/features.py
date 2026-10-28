"""Extract features ONCE for every clip and cache them to disk as .npy files.

Three kinds of model input are cached:
- waves  : (N, 48000)       3 s raw waveform at 16 kHz  -> wav2vec2 / HuBERT / WavLM
- logmel : (N, 1, 64, 301)  log-mel spectrogram          -> CNN, CNN-LSTM, CRNN, BiLSTM
- stats  : (N, D)           hand-crafted summary vector  -> SVM, Random Forest

Why cache? librosa feature extraction is the slowest step. Doing it once means every
model (and every re-run) trains in minutes instead of re-reading 8k wav files.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from joblib import Parallel, delayed

from . import audio_features as af


def load_audio(path: str) -> tuple[np.ndarray, int]:
    try:
        audio, sr = sf.read(path, dtype="float32", always_2d=False)
    except Exception:  # unusual codecs -> fall back to librosa/audioread
        import librosa
        audio, sr = librosa.load(path, sr=None, mono=True)
    return audio, sr


def stats_features(clip: np.ndarray, sr: int = af.SAMPLE_RATE) -> np.ndarray:
    """Classic SER descriptor: mean/std of MFCC, deltas, chroma, spectral shape, energy."""
    import librosa
    mfcc = librosa.feature.mfcc(y=clip, sr=sr, n_mfcc=40)
    delta = librosa.feature.delta(mfcc)
    chroma = librosa.feature.chroma_stft(y=clip, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=clip, sr=sr)
    centroid = librosa.feature.spectral_centroid(y=clip, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=clip, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=clip, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(clip)
    rms = librosa.feature.rms(y=clip)
    parts = [mfcc, delta, chroma, contrast, centroid, bandwidth, rolloff, zcr, rms]
    vec = [np.concatenate([p.mean(axis=1), p.std(axis=1)]) for p in parts]
    return np.concatenate(vec).astype(np.float32)


def _process(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    audio, sr = load_audio(path)
    clip = af.preprocess_waveform(audio, sr)
    return clip, af.logmel_features(clip), stats_features(clip)


def _manifest_key(df: pd.DataFrame) -> str:
    h = hashlib.md5("\n".join(df["path"]).encode()).hexdigest()[:10]
    return f"{len(df)}_{h}"


def build_cache(df: pd.DataFrame, cache_dir: str | Path, n_jobs: int = -1) -> dict[str, np.ndarray]:
    """Returns dict with 'waves', 'logmel', 'stats' arrays aligned with df rows."""
    cache_dir = Path(cache_dir) / _manifest_key(df)
    files = {k: cache_dir / f"{k}.npy" for k in ("waves", "logmel", "stats")}
    if all(f.exists() for f in files.values()):
        print(f"[features] using cache {cache_dir}")
        return {k: np.load(f, mmap_mode="r") for k, f in files.items()}

    print(f"[features] extracting {len(df)} clips -> {cache_dir} (one-time step)")
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = Parallel(n_jobs=n_jobs, verbose=5)(delayed(_process)(p) for p in df["path"])
    waves = np.stack([o[0] for o in out]).astype(np.float16)
    logmel = np.stack([o[1] for o in out]).astype(np.float16)
    stats = np.stack([o[2] for o in out]).astype(np.float32)
    np.save(files["waves"], waves)
    np.save(files["logmel"], logmel)
    np.save(files["stats"], stats)
    df.to_csv(cache_dir / "manifest.csv", index=False)
    (cache_dir / "info.json").write_text(json.dumps(
        {"n": len(df), "sample_rate": af.SAMPLE_RATE, "clip_seconds": af.CLIP_SECONDS,
         "n_mels": af.N_MELS, "n_frames": af.N_FRAMES, "stats_dim": int(stats.shape[1])}, indent=2))
    return {"waves": waves, "logmel": logmel, "stats": stats}
