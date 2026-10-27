"""Evaluation metrics.

- accuracy (WA, weighted accuracy): % of all clips predicted correctly.
- uar (UA, unweighted average recall): average of per-class recall. This is the
  standard SER metric because it is not fooled by class imbalance.
- macro_f1: average F1 over classes.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score)

from .labels import EMOTIONS


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    labels = list(range(len(EMOTIONS)))
    per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "uar": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "per_class_f1": {e: float(v) for e, v in zip(EMOTIONS, per_class)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
