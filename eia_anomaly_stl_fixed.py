import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from statsmodels.tsa.seasonal import STL

np.random.seed(42)
plt.rcParams.update({'font.family': 'serif','axes.spines.top': False,'axes.spines.right': False,'axes.linewidth': 0.8})

def save_fig(path: str):
    plt.tight_layout(); plt.savefig(path, bbox_inches='tight'); plt.close()

@dataclass
class Config:
    csv_path: str = "/Users/k.jones/Downloads/medium-export-e6bf40a8b01915d7380f6f547e0dd25ddd791328d4d9fa3a77513e82e662373c/posts/2001-2025 Net_generation_United_States_all_sectors_monthly.csv"
    freq: str = "MS"
    season: int = 12
    z_thresh: float = 3.0


def load_series(cfg: Config) -> pd.Series:
    p = Path(cfg.csv_path)
    df = pd.read_csv(p, header=None, usecols=[0,1], names=["date","value"], sep=",")
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    s = df.dropna().sort_values("date").set_index("date")["value"].asfreq(cfg.freq)
    return s


def main():
    cfg = Config()
    s = load_series(cfg)
    stl = STL(s, period=cfg.season, robust=True).fit()
    resid = stl.resid
    z = (resid - resid.mean()) / (resid.std(ddof=1) if resid.std(ddof=1) else 1.0)
    anomalies = z.abs() > cfg.z_thresh

    plt.figure(figsize=(9,5))
    plt.plot(s.index, s.values, label="series", alpha=0.7)
    plt.scatter(s.index[anomalies], s.values[anomalies], color='red', s=24, label="anomaly")
    plt.legend()
    save_fig("eia_anomaly_stl.png")

if __name__ == "__main__":
    main()
