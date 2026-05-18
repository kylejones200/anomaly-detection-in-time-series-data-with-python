#!/usr/bin/env python3
"""Anomaly detection — Polars + DuckDB rewrite (statistical method)."""

import argparse
import logging
from pathlib import Path

import numpy as np
import polars as pl
import yaml
from core import create_lagged_features, detect_anomalies_statistical, plot_anomalies

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_config(config_path: Path | None = None) -> dict:
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Anomaly detection — Polars + DuckDB")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(config["output"]["figures_dir"])
    )
    output_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(config["data"]["seed"])
    n = config["data"]["n_periods"]
    values = np.sin(np.arange(n) / 20) + rng.normal(0, 0.3, n)
    # inject obvious anomalies
    spike_idx = rng.choice(n, n // 20, replace=False)
    values[spike_idx] += rng.choice([-6, 6], len(spike_idx))
    series = pl.Series("value", values.tolist())
    # ── lagged feature matrix ─────────────────────────────────────────────────
    lag = config["model"]["lag"]
    lagged = create_lagged_features(series, lag=lag)
    logging.info(f"Lag matrix: {lagged.shape}  columns: {lagged.columns}")
    # ── statistical (z-score) anomaly detection ───────────────────────────────
    threshold = config["model"]["statistical_threshold"]
    result = detect_anomalies_statistical(series, threshold=threshold)
    n_anomalies = result["is_anomaly"].sum()
    logging.info(f"\nZ-score threshold: {threshold}")
    logging.info(f"Anomalies detected: {n_anomalies} / {result.height}")
    logging.info(
        f"\nTop anomalies by z-score:\n"
        f"{result.sort('z_score', descending=True).head(5).select(['idx', 'value', 'z_score'])}"
    )
    plot_anomalies(
        result, "Z-Score Anomaly Detection", output_dir / "anomalies_statistical.png"
    )
    logging.info(f"\nDone. Figures saved to {output_dir}")


if __name__ == "__main__":
    main()
