"""Anomaly detection using Polars and DuckDB.

create_lagged_features:       pandas .shift() loop  → DuckDB LAG() window
detect_anomalies_statistical: pandas z-score        → DuckDB (value - AVG) / STDDEV_SAMP
                                                       computed over the full series in one pass

IsolationForest is unchanged (sklearn); the statistical detector is the
DuckDB showcase here.
"""

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import polars as pl


def create_lagged_features(series: pl.Series, lag: int = 5) -> pl.DataFrame:
    """pandas .shift() loop → DuckDB LAG() window functions."""
    pl.DataFrame({"idx": range(len(series)), "value": series})
    lag_exprs = ",\n            ".join(
        f"LAG(value, {i}) OVER (ORDER BY idx) AS lag_{i}" for i in range(1, lag + 1)
    )
    return (
        duckdb.sql(f"""
        SELECT value, {lag_exprs}
        FROM df
        ORDER BY idx
    """)
        .pl()
        .drop_nulls()
    )


def detect_anomalies_statistical(
    series: pl.Series,
    threshold: float = 3.0,
) -> pl.DataFrame:
    """
    Z-score anomaly detection via DuckDB window aggregates.
    Original: (series - series.mean()) / series.std()  (two pandas passes)
    New:      single DuckDB query computes mean, std, z-score, and flag together.
    """
    pl.DataFrame({"idx": range(len(series)), "value": series})
    return duckdb.sql(f"""
        SELECT
            idx,
            value,
            (value - AVG(value) OVER ())         AS deviation,
            ABS(value - AVG(value) OVER ())
                / NULLIF(STDDEV_SAMP(value) OVER (), 0) AS z_score,
            CASE
                WHEN ABS(value - AVG(value) OVER ())
                     / NULLIF(STDDEV_SAMP(value) OVER (), 0) > {threshold}
                THEN 1 ELSE 0
            END AS is_anomaly
        FROM df
        ORDER BY idx
    """).pl()


def plot_anomalies(
    result: pl.DataFrame,
    title: str,
    output_path: Path,
    plot: bool = False,
):
    if not plot:
        return
    indices = result["idx"].to_list()
    values = result["value"].to_list()
    is_anom = result["is_anomaly"].to_list()
    anom_idx = [i for i, a in zip(indices, is_anom) if a == 1]
    anom_val = [v for v, a in zip(values, is_anom) if a == 1]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(indices, values, label="Time Series", color="#4A90A4", linewidth=1.2)
    if anom_idx:
        ax.scatter(
            anom_idx, anom_val, color="#D4A574", s=50, label="Anomalies", zorder=5
        )
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()
