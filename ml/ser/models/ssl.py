"""Self-supervised speech models (input: raw 16 kHz waveform, shape (B, 48000)).

wav2vec2 / HuBERT / WavLM were pre-trained on hundreds of hours of unlabelled speech,
so they already "understand" voices. We add a small classifier on top and fine-tune.
These are usually the most accurate models, but also the biggest (~95M parameters).
"""
from __future__ import annotations

import torch
import torch.nn as nn

CHECKPOINTS = {
    "wav2vec2": "facebook/wav2vec2-base",
    "hubert": "facebook/hubert-base-ls960",
    "wavlm": "microsoft/wavlm-base-plus",
}


class SSLClassifier(nn.Module):
    def __init__(self, n_classes: int, backbone: str = "wav2vec2", freeze_backbone: bool = False,
                 dropout: float = 0.2):
        super().__init__()
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained(CHECKPOINTS[backbone])
        # LayerDrop randomly skips layers in training, which changes the number of hidden
        # states; disable it so the weighted layer sum below always has the same size.
        if hasattr(self.backbone.config, "layerdrop"):
            self.backbone.config.layerdrop = 0.0
        # The CNN feature encoder at the bottom is always frozen (standard practice).
        if hasattr(self.backbone, "feature_extractor"):
            for p in self.backbone.feature_extractor.parameters():
                p.requires_grad = False
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        # count hidden states with a tiny dummy pass (differs between model types/versions)
        with torch.no_grad():
            n_layers = len(self.backbone(torch.zeros(1, 16000), output_hidden_states=True).hidden_states)
        hidden = self.backbone.config.hidden_size
        # learnable weighted sum of all transformer layers (emotion info sits in middle layers)
        self.layer_weights = nn.Parameter(torch.zeros(n_layers))
        self.head = nn.Sequential(nn.Linear(hidden, 256), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(256, n_classes))

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        # zero-mean / unit-variance per clip, as the HF feature extractors do
        wave = (wave - wave.mean(dim=1, keepdim=True)) / (wave.std(dim=1, keepdim=True) + 1e-6)
        out = self.backbone(wave, output_hidden_states=True)
        hs = torch.stack(out.hidden_states, dim=0)              # (L, B, T, H)
        w = torch.softmax(self.layer_weights, dim=0)[:, None, None, None]
        x = (w * hs).sum(dim=0).mean(dim=1)                    # (B, H)
        return self.head(x)

    def param_groups(self, lr_backbone: float, lr_head: float):
        bb = [p for p in self.backbone.parameters() if p.requires_grad]
        head = [self.layer_weights, *self.head.parameters()]
        groups = [{"params": head, "lr": lr_head}]
        if bb:
            groups.append({"params": bb, "lr": lr_backbone})
        return groups
