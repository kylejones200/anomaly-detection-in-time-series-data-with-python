"""Global z-score anomaly flags."""

from __future__ import annotations

import numpy as np


def detect_anomalies_zscore(values: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    mean = v.mean()
    std = max(v.std(), 1e-12)
    z = np.abs((v - mean) / std)
    return (z > threshold).astype(float)
