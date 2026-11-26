"""Macro F1 for integer labels, as sklearn.metrics.f1_score(average='macro',
zero_division=0), without its input validation.

The federated evaluation calls it twice per client and per round: sklearn's checks
then cost more than the evaluation itself (about 4 ms per call).
"""
from typing import Sequence

import numpy as np


def macro_f1(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Mean over the labels present in y_true or y_pred of 2 TP / (2 TP + FP + FN).
    0.0 for empty inputs."""
    t, p = np.asarray(y_true), np.asarray(y_pred)
    if t.size == 0:
        return 0.0
    labels = np.union1d(t, p)
    tp = np.array([np.sum((t == c) & (p == c)) for c in labels], dtype=np.float64)
    fp = np.array([np.sum((t != c) & (p == c)) for c in labels], dtype=np.float64)
    fn = np.array([np.sum((t == c) & (p != c)) for c in labels], dtype=np.float64)
    return float(np.mean(2 * tp / (2 * tp + fp + fn)))
