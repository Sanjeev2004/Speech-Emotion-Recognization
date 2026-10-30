"""Load ONNX models from MODEL_DIR and run predictions."""
from __future__ import annotations

import io
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf

from . import audio_features as af

MODEL_DIR = Path(os.getenv("MODEL_DIR", Path(__file__).resolve().parent.parent / "models"))
MAX_SECONDS = float(os.getenv("MAX_AUDIO_SECONDS", "30"))


@dataclass
class LoadedModel:
    meta: dict
    session: ort.InferenceSession

    def predict(self, clips: list[np.ndarray]) -> np.ndarray:
        if self.meta["input"] == "logmel":
            x = np.stack([af.logmel_features(c) for c in clips]).astype(np.float32)
        else:
            x = np.stack(clips).astype(np.float32)
        probs = self.session.run(None, {self.meta["input_name"]: x})[0]
        return probs.mean(axis=0)     # average over 3 s windows


class Registry:
    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model_dir = Path(model_dir)
        self.models: dict[str, LoadedModel] = {}
        self.default: str | None = os.getenv("DEFAULT_MODEL")

    def load(self) -> None:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(os.getenv("ORT_THREADS", "1"))
        for meta_path in sorted(self.model_dir.glob("*.json")):
            meta = json.loads(meta_path.read_text())
            if "input" not in meta:
                continue          # e.g. leaderboard.json
            onnx_path = meta_path.with_suffix(".onnx")
            if not onnx_path.exists():
                onnx_path = _download(onnx_path.name)
                if onnx_path is None:
                    continue
            sess = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])
            self.models[meta["id"]] = LoadedModel(meta, sess)
        if self.models and self.default not in self.models:
            # default = best test UAR among loaded models
            self.default = max(self.models, key=lambda k: self.models[k].meta.get("test_uar") or 0)

    def leaderboard(self) -> dict | None:
        p = self.model_dir / "leaderboard.json"
        return json.loads(p.read_text()) if p.exists() else None


def _download(filename: str) -> Path | None:
    """Optional: fetch a model file from MODEL_BASE_URL (e.g. the raw GitHub folder) into /tmp.
    Used when the hosting upload does not include the binary .onnx files.

    Prefer shipping .onnx in MODEL_DIR (Docker / Render / Vercel with the file present).
    Set MODEL_BASE_URL to a stable branch, e.g.:
      https://raw.githubusercontent.com/<user>/<repo>/main/backend/models
    """
    base = os.getenv("MODEL_BASE_URL")
    if not base:
        print(f"[models] missing {filename} and MODEL_BASE_URL is unset; skip")
        return None
    import urllib.error
    import urllib.request

    dest = Path(os.getenv("MODEL_CACHE_DIR", "/tmp/ser-models")) / filename
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"{base.rstrip('/')}/{filename}"
    print(f"[models] downloading {url}")
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            data = r.read()
        if not data:
            print(f"[models] download empty: {url}")
            return None
        tmp = dest.with_suffix(dest.suffix + ".partial")
        tmp.write_bytes(data)
        tmp.replace(dest)
    except urllib.error.HTTPError as e:
        print(f"[models] download HTTP {e.code} for {url}")
        return None
    except Exception as e:  # keep the API up; /api/health will show 0 models
        print(f"[models] download failed ({type(e).__name__}): {e}")
        return None
    return dest


def decode_audio(data: bytes) -> tuple[np.ndarray, int]:
    """WAV / FLAC / OGG (and MP3 with libsndfile >= 1.1). The frontend always sends WAV."""
    try:
        audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    except Exception as e:
        raise ValueError(f"Could not decode audio ({e}). Please send a WAV file.") from e
    audio = af.to_mono(audio)
    if len(audio) / sr > MAX_SECONDS:
        audio = audio[: int(MAX_SECONDS * sr)]
    if len(audio) / sr < 0.3:
        raise ValueError("Audio is too short (need at least 0.3 s).")
    return audio, sr


def run_prediction(model: LoadedModel, data: bytes) -> dict:
    t0 = time.perf_counter()
    audio, sr = decode_audio(data)
    clips = af.split_into_clips(audio, sr)
    probs = model.predict(clips)
    emotions = model.meta["emotions"]
    order = np.argsort(-probs)
    return {
        "model": model.meta["id"],
        "emotion": emotions[int(order[0])],
        "confidence": float(probs[order[0]]),
        "probabilities": {emotions[i]: float(probs[i]) for i in range(len(emotions))},
        "duration_seconds": round(len(audio) / sr, 2),
        "windows": len(clips),
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
    }
