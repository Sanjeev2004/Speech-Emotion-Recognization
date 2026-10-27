"""Spectrogram models (input: log-mel, shape (B, 1, 64 mels, 301 frames))."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(c_in: int, c_out: int, pool: tuple[int, int], dropout: float = 0.2) -> nn.Sequential:
    """Conv -> BatchNorm -> ReLU (x2) -> MaxPool -> Dropout."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.MaxPool2d(pool), nn.Dropout(dropout),
    )


class CNN2D(nn.Module):
    """Plain CNN: treats the spectrogram like an image. Strong, simple baseline."""

    def __init__(self, n_classes: int, width: int = 32, dropout: float = 0.3):
        super().__init__()
        w = width
        self.features = nn.Sequential(
            conv_block(1, w, (2, 2)), conv_block(w, 2 * w, (2, 2)),
            conv_block(2 * w, 4 * w, (2, 2)), conv_block(4 * w, 8 * w, (2, 2)),
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(8 * w, n_classes))

    def forward(self, x):
        x = self.features(x)
        x = x.mean(dim=(2, 3))            # global average pooling
        return self.head(x)


class CNNLSTM(nn.Module):
    """CNN extracts local patterns, LSTM models how they change over time.

    Fix vs. the old Keras version: there the CNN output was flattened and reshaped to
    (40, -1), mixing time and frequency. Here time stays the sequence axis.
    """

    def __init__(self, n_classes: int, width: int = 32, hidden: int = 128, dropout: float = 0.3):
        super().__init__()
        w = width
        # pool frequency more than time so the sequence stays long (301 -> 75 steps)
        self.cnn = nn.Sequential(
            conv_block(1, w, (2, 2)), conv_block(w, 2 * w, (2, 2)), conv_block(2 * w, 4 * w, (4, 1)),
        )
        feat = 4 * w * (64 // 16)          # channels x remaining mel bins
        self.lstm = nn.LSTM(feat, hidden, num_layers=2, batch_first=True,
                            bidirectional=True, dropout=dropout)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * hidden, n_classes))

    def forward(self, x):
        x = self.cnn(x)                    # (B, C, F', T')
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, t, c * f)   # (B, T', C*F')
        x, _ = self.lstm(x)
        return self.head(x.mean(dim=1))


class AttentionPool(nn.Module):
    """Learns which time steps matter most (e.g. the emotional peak of a sentence)."""

    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, dim // 2), nn.Tanh(), nn.Linear(dim // 2, 1))

    def forward(self, x):                  # x: (B, T, D)
        w = torch.softmax(self.score(x), dim=1)
        return (w * x).sum(dim=1)


class CRNNAttention(nn.Module):
    """CNN + bidirectional GRU + attention pooling."""

    def __init__(self, n_classes: int, width: int = 32, hidden: int = 128, dropout: float = 0.3):
        super().__init__()
        w = width
        self.cnn = nn.Sequential(
            conv_block(1, w, (2, 2)), conv_block(w, 2 * w, (2, 2)), conv_block(2 * w, 4 * w, (4, 1)),
        )
        feat = 4 * w * (64 // 16)
        self.proj = nn.Linear(feat, hidden * 2)
        self.gru = nn.GRU(hidden * 2, hidden, batch_first=True, bidirectional=True)
        self.pool = AttentionPool(2 * hidden)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * hidden, n_classes))

    def forward(self, x):
        x = self.cnn(x)
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, t, c * f)
        x, _ = self.gru(F.relu(self.proj(x)))
        return self.head(self.pool(x))


class BiLSTM(nn.Module):
    """No CNN: each spectrogram frame (64 mel values) is one LSTM time step."""

    def __init__(self, n_classes: int, hidden: int = 128, dropout: float = 0.3):
        super().__init__()
        self.norm = nn.LayerNorm(64)
        self.lstm = nn.LSTM(64, hidden, num_layers=2, batch_first=True,
                            bidirectional=True, dropout=dropout)
        self.pool = AttentionPool(2 * hidden)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * hidden, n_classes))

    def forward(self, x):                  # (B, 1, 64, T)
        x = self.norm(x.squeeze(1).transpose(1, 2))   # (B, T, 64)
        x, _ = self.lstm(x)
        return self.head(self.pool(x))


def spec_augment(x: torch.Tensor, freq_mask: int = 8, time_mask: int = 30, n_masks: int = 2,
                 max_shift: int = 30) -> torch.Tensor:
    """Training-only augmentation on spectrograms (cheap, runs on GPU):
    random time shift + hide random frequency bands and time ranges."""
    b, _, f, t = x.shape
    x = x.clone()
    shifts = torch.randint(-max_shift, max_shift + 1, (b,))
    for i in range(b):
        x[i] = torch.roll(x[i], int(shifts[i]), dims=-1)
        for _ in range(n_masks):
            fw = int(torch.randint(0, freq_mask + 1, (1,)))
            f0 = int(torch.randint(0, f - fw + 1, (1,)))
            x[i, :, f0:f0 + fw, :] = 0
            tw = int(torch.randint(0, time_mask + 1, (1,)))
            t0 = int(torch.randint(0, t - tw + 1, (1,)))
            x[i, :, :, t0:t0 + tw] = 0
    return x
