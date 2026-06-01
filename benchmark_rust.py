#!/usr/bin/env python3
"""Python vs Rust kernel benchmark."""

from __future__ import annotations

import time
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from compute_kernel import detect_anomalies_zscore  # noqa: E402

def main() -> None:
    v = np.ascontiguousarray(np.sin(np.arange(10000) * 0.01)); th = 3.0
    t0 = time.perf_counter()
    for _ in range(200):
        detect_anomalies_zscore(v, th)
    py_s = time.perf_counter() - t0
    try:
        import anomaly_detection_in_time_series_data_with_python_rs as rs
    except ImportError:
        print("Build: maturin develop --release -m rust/py/Cargo.toml")
        print(f"Python {py_s:.3f}s")
        return
    rs_s = rs.bench_kernel_py(v, th, 500)
    print(f"Python {py_s:.3f}s Rust {rs_s:.3f}s speedup {py_s / max(rs_s, 1e-9):.1f}x")
    np.testing.assert_allclose(detect_anomalies_zscore(v, th), np.asarray(rs.detect_anomalies_zscore_py(v, th)), rtol=1e-10)
    print("Correctness: OK")

if __name__ == "__main__":
    main()
