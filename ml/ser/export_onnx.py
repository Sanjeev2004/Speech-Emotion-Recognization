"""Export a trained PyTorch model to ONNX for the backend.

    python -m ser.export_onnx --run runs/cnn2d__combined --out ../backend/models

Writes <out>/<model>__<dataset>.onnx and a matching .json with labels, input type and
test metrics. The backend only needs onnxruntime (no PyTorch), which keeps the server
small enough for a free 512 MB instance.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from . import audio_features as af
from .labels import EMOTIONS
from .models import REGISTRY, build_model


def export(run_dir: str | Path, out_dir: str | Path, opset: int = 17) -> Path:
    import torch

    run_dir, out_dir = Path(run_dir), Path(out_dir)
    result = json.loads((run_dir / "result.json").read_text())
    spec = REGISTRY[result["model"]]
    if spec.kind == "classical":
        raise SystemExit("Classical models are only for comparison; export a deep model.")
    model = build_model(spec.name, len(EMOTIONS))
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu"))
    model.eval()

    if spec.input == "logmel":
        dummy = torch.zeros(1, 1, af.N_MELS, af.N_FRAMES)
        input_name = "logmel"
    else:
        dummy = torch.zeros(1, af.CLIP_SAMPLES)
        input_name = "waveform"

    class WithSoftmax(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return torch.softmax(self.m(x), dim=-1)

    name = f"{result['model']}__{result['train_dataset']}"
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / f"{name}.onnx"
    wrapped = WithSoftmax(model).eval()   # eval mode: no dropout, BatchNorm uses running stats
    torch.onnx.export(wrapped, dummy, str(onnx_path), input_names=[input_name],
                      output_names=["probs"], dynamic_axes={input_name: {0: "batch"}, "probs": {0: "batch"}},
                      opset_version=opset, dynamo=False)  # classic exporter: stable for LSTM/GRU

    # sanity check: ONNX output == PyTorch output
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    x = np.random.randn(*dummy.shape).astype(np.float32)
    with torch.no_grad():
        ref = torch.softmax(model(torch.from_numpy(x)), -1).numpy()
    got = sess.run(None, {input_name: x})[0]
    assert np.allclose(ref, got, atol=1e-4), "ONNX output differs from PyTorch"

    own = result["test"].get(result["train_dataset"], {})
    meta = {
        "id": name, "model": result["model"], "description": result["description"],
        "train_dataset": result["train_dataset"], "input": spec.input, "input_name": input_name,
        "emotions": EMOTIONS, "sample_rate": af.SAMPLE_RATE, "clip_seconds": af.CLIP_SECONDS,
        "n_params": result.get("n_params"),
        "test_accuracy": own.get("accuracy"), "test_uar": own.get("uar"),
        "size_mb": round(onnx_path.stat().st_size / 1e6, 2),
    }
    (out_dir / f"{name}.json").write_text(json.dumps(meta, indent=2))
    print(f"[export] {onnx_path} ({meta['size_mb']} MB), ONNX check passed")
    return onnx_path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", nargs="+", required=True, help="one or more runs/<model>__<dataset> folders")
    p.add_argument("--out", default="../backend/models")
    p.add_argument("--leaderboard", default="reports/leaderboard.json",
                   help="also copy the comparison report next to the models")
    a = p.parse_args()
    for r in a.run:
        export(r, a.out)
    if Path(a.leaderboard).exists():
        shutil.copy(a.leaderboard, Path(a.out) / "leaderboard.json")
        print(f"[export] copied {a.leaderboard}")


if __name__ == "__main__":
    main()
