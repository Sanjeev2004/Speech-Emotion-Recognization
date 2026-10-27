"""End-to-end smoke test on a tiny synthetic dataset (no real data needed).

    cd ml && pytest -q
Each emotion gets a different tone, so even 2 epochs should beat random guessing.
"""
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from ser import audio_features as af
from ser import compare, train
from ser.datasets import build_manifest
from ser.labels import CREMAD_CODES, RAVDESS_CODES

FREQ = {"neutral": 200, "happy": 400, "sad": 600, "angry": 900, "fear": 1300, "disgust": 1800}


def _tone(freq, sr, seconds, rng):
    t = np.arange(int(sr * seconds)) / sr
    return (0.5 * np.sin(2 * np.pi * freq * t) + 0.05 * rng.standard_normal(len(t))).astype(np.float32)


@pytest.fixture(scope="session")
def data(tmp_path_factory):
    root = tmp_path_factory.mktemp("data")
    rng = np.random.default_rng(0)
    for actor in range(1, 9):
        d = root / "ravdess" / f"Actor_{actor:02d}"
        d.mkdir(parents=True)
        for code, emo in list(RAVDESS_CODES.items()) + [("02", "calm")]:
            for rep in (1, 2):
                f = FREQ.get(emo, 300)
                sf.write(d / f"03-01-{code}-01-01-{rep:02d}-{actor:02d}.wav", _tone(f, 48000, 3.5, rng), 48000)
    d = root / "cremad"
    d.mkdir()
    for actor in range(1001, 1009):
        for code, emo in CREMAD_CODES.items():
            for sent in ("DFA", "IEO"):
                sf.write(d / f"{actor}_{sent}_{code}_XX.wav", _tone(FREQ[emo], 16000, 2.2, rng), 16000)
    return root


def test_manifest_and_split(data):
    df = build_manifest(data / "ravdess", data / "cremad")
    assert set(df["dataset"]) == {"ravdess", "cremad"}
    assert "calm" not in set(df["emotion"])            # dropped
    assert len(df) == 8 * 6 * 2 * 2
    # no speaker in more than one split
    assert (df.groupby("speaker")["split"].nunique() == 1).all()


def test_logmel_shape():
    clip = af.preprocess_waveform(np.random.randn(70000).astype(np.float32), 22050)
    assert clip.shape == (af.CLIP_SAMPLES,)
    feat = af.logmel_features(clip)
    assert feat.shape == (1, af.N_MELS, af.N_FRAMES)
    assert abs(float(feat.mean())) < 1e-3


@pytest.mark.parametrize("model", ["svm", "random_forest", "cnn2d", "cnn_lstm", "crnn_attention", "bilstm"])
def test_train_each_model(data, tmp_path_factory, model):
    out = tmp_path_factory.getbasetemp() / "runs"
    args = train.parser().parse_args([
        "--model", model, "--dataset", "combined", "--ravdess-dir", str(data / "ravdess"),
        "--cremad-dir", str(data / "cremad"), "--cache-dir", str(tmp_path_factory.getbasetemp() / "cache"),
        "--out", str(out), "--epochs", "3", "--batch-size", "16", "--n-jobs", "2"])
    res = train.run(args)
    assert set(res["test"]) == {"ravdess", "cremad", "combined"}
    assert res["test"]["combined"]["accuracy"] > 1 / 6      # better than chance


def test_compare_and_export(tmp_path_factory):
    base = tmp_path_factory.getbasetemp()
    df = compare.build_report(base / "runs", base / "reports")
    assert len(df) == 6 * 3
    payload = json.loads((base / "reports" / "leaderboard.json").read_text())
    assert payload["rows"]
    from ser.export_onnx import export
    onnx_path = export(base / "runs" / "cnn2d__combined", base / "export")
    assert Path(onnx_path).exists()
    meta = json.loads(Path(str(onnx_path).replace(".onnx", ".json")).read_text())
    assert meta["input"] == "logmel"
