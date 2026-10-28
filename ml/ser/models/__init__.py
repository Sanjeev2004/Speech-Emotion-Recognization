"""Model registry: one place that lists every model the project can train.

kind:   classical (scikit-learn) | deep (PyTorch, spectrogram) | ssl (PyTorch, waveform)
input:  which cached feature the model consumes (see ser/features.py)
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelSpec:
    name: str
    kind: str
    input: str
    description: str
    defaults: dict = field(default_factory=dict)
    deployable: bool = True   # small enough for a 512 MB free server


REGISTRY: dict[str, ModelSpec] = {s.name: s for s in [
    ModelSpec("svm", "classical", "stats", "RBF SVM on MFCC/chroma/spectral statistics", deployable=False),
    ModelSpec("random_forest", "classical", "stats", "Random Forest on the same statistics", deployable=False),
    ModelSpec("bilstm", "deep", "logmel", "2-layer BiLSTM + attention over mel frames",
              {"epochs": 60, "lr": 1e-3, "batch_size": 64}),
    ModelSpec("cnn2d", "deep", "logmel", "4-block 2D CNN on log-mel spectrogram",
              {"epochs": 60, "lr": 1e-3, "batch_size": 64}),
    ModelSpec("cnn_lstm", "deep", "logmel", "CNN + BiLSTM hybrid (fixed version of the original)",
              {"epochs": 60, "lr": 1e-3, "batch_size": 64}),
    ModelSpec("crnn_attention", "deep", "logmel", "CNN + BiGRU + attention pooling",
              {"epochs": 60, "lr": 1e-3, "batch_size": 64}),
    ModelSpec("wav2vec2", "ssl", "waves", "Fine-tuned wav2vec2-base (self-supervised, 95M params)",
              {"epochs": 15, "lr": 3e-5, "head_lr": 1e-3, "batch_size": 16}, deployable=False),
    ModelSpec("hubert", "ssl", "waves", "Fine-tuned HuBERT-base",
              {"epochs": 15, "lr": 3e-5, "head_lr": 1e-3, "batch_size": 16}, deployable=False),
    ModelSpec("wavlm", "ssl", "waves", "Fine-tuned WavLM-base-plus",
              {"epochs": 15, "lr": 3e-5, "head_lr": 1e-3, "batch_size": 16}, deployable=False),
]}


def build_model(name: str, n_classes: int, seed: int = 42, freeze_backbone: bool = False):
    spec = REGISTRY[name]
    if spec.kind == "classical":
        from .classical import build_random_forest, build_svm
        return {"svm": build_svm, "random_forest": build_random_forest}[name](seed)
    if spec.kind == "deep":
        from .deep import CNN2D, CNNLSTM, BiLSTM, CRNNAttention
        cls = {"cnn2d": CNN2D, "cnn_lstm": CNNLSTM, "crnn_attention": CRNNAttention, "bilstm": BiLSTM}[name]
        return cls(n_classes)
    from .ssl import SSLClassifier
    return SSLClassifier(n_classes, backbone=name, freeze_backbone=freeze_backbone)
