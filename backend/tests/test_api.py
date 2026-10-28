"""API tests with a tiny fake ONNX model (no training needed)."""
import io
import json

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from app import audio_features as af

EMOTIONS = ["neutral", "happy", "sad", "angry", "fear", "disgust"]


def _fake_model(folder):
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    w = numpy_helper.from_array(np.random.randn(af.N_MELS * af.N_FRAMES, 6).astype(np.float32) * 0.01, "W")
    shape = numpy_helper.from_array(np.array([0, -1], dtype=np.int64), "shape")
    graph = helper.make_graph(
        [helper.make_node("Reshape", ["logmel", "shape"], ["flat"]),
         helper.make_node("MatMul", ["flat", "W"], ["logits"]),
         helper.make_node("Softmax", ["logits"], ["probs"], axis=-1)],
        "fake", [helper.make_tensor_value_info("logmel", TensorProto.FLOAT, ["batch", 1, af.N_MELS, af.N_FRAMES])],
        [helper.make_tensor_value_info("probs", TensorProto.FLOAT, ["batch", 6])], [w, shape])
    onnx.save(helper.make_model(graph, ir_version=8, opset_imports=[helper.make_opsetid("", 17)]), folder / "fake__combined.onnx")
    (folder / "fake__combined.json").write_text(json.dumps({
        "id": "fake__combined", "model": "fake", "input": "logmel", "input_name": "logmel",
        "emotions": EMOTIONS, "test_uar": 0.5}))
    (folder / "leaderboard.json").write_text(json.dumps({"rows": [{"model": "fake", "uar": 0.5}]}))


@pytest.fixture()
def client(tmp_path, monkeypatch):
    _fake_model(tmp_path)
    import app.main as main
    from app.inference import Registry
    monkeypatch.setattr(main, "registry", Registry(tmp_path))
    with TestClient(main.app) as c:
        yield c


def _wav(seconds=4.0, sr=44100):
    buf = io.BytesIO()
    t = np.arange(int(sr * seconds)) / sr
    sf.write(buf, (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sr, format="WAV")
    return buf.getvalue()


def test_health_and_models(client):
    assert client.get("/api/health").json()["models_loaded"] == 1
    assert client.get("/api/models").json()["default"] == "fake__combined"
    assert client.get("/api/results").json()["rows"]


def test_predict(client):
    r = client.post("/api/predict", files={"file": ("a.wav", _wav(), "audio/wav")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["emotion"] in EMOTIONS
    assert abs(sum(body["probabilities"].values()) - 1) < 1e-4
    assert body["windows"] >= 2           # 4 s audio -> several 3 s windows


def test_bad_audio(client):
    r = client.post("/api/predict", files={"file": ("a.wav", b"not audio", "audio/wav")})
    assert r.status_code == 400
