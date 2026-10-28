"""Pure-NumPy audio front-end shared by training AND the backend.

Why pure NumPy?
- The backend runs on a small free server (512 MB RAM). librosa pulls in numba/llvmlite,
  which is heavy. NumPy + SciPy is enough for a log-mel spectrogram.
- Training and inference must compute features in EXACTLY the same way, otherwise the
  model sees different numbers at deploy time and accuracy silently drops.
  backend/app/audio_features.py is a byte-for-byte copy of this file
  (checked by backend/tests/test_features_in_sync.py).

Pipeline for one clip:
    waveform (any sr) -> mono -> resample to 16 kHz -> trim silence -> pad/crop to 3 s
    -> STFT -> mel filterbank -> log (dB) -> per-utterance normalisation
"""
from __future__ import annotations

import numpy as np
from scipy.signal import resample_poly

SAMPLE_RATE = 16_000      # all models work at 16 kHz (wav2vec2/HuBERT require it)
CLIP_SECONDS = 3.0        # every clip becomes exactly 3 s
CLIP_SAMPLES = int(SAMPLE_RATE * CLIP_SECONDS)
N_FFT = 512               # FFT size (32 ms window at 16 kHz)
WIN_LENGTH = 400          # 25 ms analysis window
HOP_LENGTH = 160          # 10 ms hop -> 100 frames per second
N_MELS = 64               # number of mel bands
F_MIN, F_MAX = 20.0, 8000.0
TOP_DB = 80.0
N_FRAMES = 1 + CLIP_SAMPLES // HOP_LENGTH   # 301 frames for 3 s


# ---------------------------------------------------------------- waveform utils
def to_mono(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:  # (samples, channels) as returned by soundfile
        audio = audio.mean(axis=1)
    return audio


def resample(audio: np.ndarray, sr: int, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    if sr == target_sr:
        return audio.astype(np.float32)
    g = np.gcd(int(sr), int(target_sr))
    return resample_poly(audio, target_sr // g, sr // g).astype(np.float32)


def trim_silence(audio: np.ndarray, top_db: float = 30.0, frame: int = 512, hop: int = 128) -> np.ndarray:
    """Remove leading/trailing parts quieter than (peak - top_db)."""
    if len(audio) < frame:
        return audio
    n = 1 + (len(audio) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    rms = np.sqrt(np.mean(audio[idx] ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms / (rms.max() + 1e-12) + 1e-12)
    keep = np.where(db > -top_db)[0]
    if len(keep) == 0:
        return audio
    start, end = keep[0] * hop, min(len(audio), keep[-1] * hop + frame)
    return audio[start:end]


def fix_length(audio: np.ndarray, n: int = CLIP_SAMPLES) -> np.ndarray:
    """Centre-crop if too long, zero-pad (both sides) if too short."""
    if len(audio) >= n:
        start = (len(audio) - n) // 2
        return audio[start:start + n]
    pad = n - len(audio)
    return np.pad(audio, (pad // 2, pad - pad // 2))


def peak_normalize(audio: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(audio)) if len(audio) else 0.0
    return audio / peak if peak > 1e-6 else audio


def preprocess_waveform(audio: np.ndarray, sr: int) -> np.ndarray:
    """Any clip -> 3 s, 16 kHz, mono, peak-normalised float32."""
    audio = to_mono(audio)
    audio = resample(audio, sr)
    audio = trim_silence(audio)
    audio = peak_normalize(audio)
    return fix_length(audio).astype(np.float32)


def split_into_clips(audio: np.ndarray, sr: int, max_clips: int = 10) -> list[np.ndarray]:
    """For long recordings at inference: cut into 3 s windows (50% overlap)."""
    audio = peak_normalize(trim_silence(resample(to_mono(audio), sr)))
    if len(audio) <= CLIP_SAMPLES:
        return [fix_length(audio)]
    hop = CLIP_SAMPLES // 2
    starts = list(range(0, len(audio) - CLIP_SAMPLES + 1, hop))
    if starts[-1] + CLIP_SAMPLES < len(audio):      # cover the tail too
        starts.append(len(audio) - CLIP_SAMPLES)
    starts = starts[:max_clips]
    return [audio[s:s + CLIP_SAMPLES].astype(np.float32) for s in starts]


# ---------------------------------------------------------------- spectrograms
def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m) / 2595.0) - 1.0)


def mel_filterbank(sr: int = SAMPLE_RATE, n_fft: int = N_FFT, n_mels: int = N_MELS,
                   fmin: float = F_MIN, fmax: float = F_MAX) -> np.ndarray:
    """Triangular mel filters, shape (n_mels, n_fft//2 + 1), area-normalised."""
    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    mel_pts = np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2)
    hz = _mel_to_hz(mel_pts)
    fb = np.zeros((n_mels, len(freqs)), dtype=np.float32)
    for i in range(n_mels):
        lo, mid, hi = hz[i], hz[i + 1], hz[i + 2]
        up = (freqs - lo) / (mid - lo)
        down = (hi - freqs) / (hi - mid)
        fb[i] = np.maximum(0, np.minimum(up, down))
    fb *= (2.0 / (hz[2:] - hz[:-2]))[:, None]
    return fb


_FB = mel_filterbank()
_WINDOW = np.hanning(WIN_LENGTH + 1)[:-1].astype(np.float32)  # periodic Hann


def stft_power(audio: np.ndarray) -> np.ndarray:
    """Power spectrogram, shape (n_fft//2+1, frames). Centre-padded like librosa."""
    pad = N_FFT // 2
    x = np.pad(audio, (pad, pad), mode="reflect")
    n_frames = 1 + (len(x) - N_FFT) // HOP_LENGTH
    idx = np.arange(N_FFT)[None, :] + HOP_LENGTH * np.arange(n_frames)[:, None]
    win = np.zeros(N_FFT, dtype=np.float32)
    off = (N_FFT - WIN_LENGTH) // 2
    win[off:off + WIN_LENGTH] = _WINDOW
    frames = x[idx] * win
    spec = np.fft.rfft(frames, n=N_FFT, axis=1)
    return (np.abs(spec) ** 2).T.astype(np.float32)


def log_mel(audio: np.ndarray) -> np.ndarray:
    """Log-mel spectrogram in dB, shape (N_MELS, frames)."""
    mel = _FB @ stft_power(audio)
    db = 10.0 * np.log10(np.maximum(mel, 1e-10))
    return np.maximum(db, db.max() - TOP_DB).astype(np.float32)


def normalize(feat: np.ndarray) -> np.ndarray:
    """Per-utterance mean/std normalisation (no dataset statistics needed at deploy)."""
    return ((feat - feat.mean()) / (feat.std() + 1e-6)).astype(np.float32)


def logmel_features(clip_16k: np.ndarray) -> np.ndarray:
    """Model input for CNN/LSTM models: (1, N_MELS, N_FRAMES) float32."""
    return normalize(log_mel(clip_16k))[None, :, :]
