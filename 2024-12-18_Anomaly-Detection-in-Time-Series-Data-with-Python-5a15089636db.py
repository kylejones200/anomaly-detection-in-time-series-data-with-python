# Description: Short example for Anomaly Detection in Time Series Data with Python.



from data_io import read_csv
from dataclasses import dataclass
from keras.callbacks import EarlyStopping
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed
from keras.models import Sequential
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import STL
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)



# Simulated Time Series with Anomalies
np.random.seed(42)
normal_data = np.sin(np.linspace(0, 50, 500))  # Sine wave as normal data
anomaly_data = normal_data.copy()
anomaly_data[450:460] += 3  # Inject anomalies

# Reshape and Scale Data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(anomaly_data.reshape(-1, 1))

# Train Isolation Forest Model
iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
iso_forest.fit(scaled_data)

# Compute Anomaly Scores
scores = iso_forest.decision_function(scaled_data)
scores = -scores  # Invert scores so that higher values indicate more anomalous points

# Identify Anomalies: Points with Low Decision Function Values
threshold = np.percentile(scores, 95)  # Top 5% as anomalies
anomalies = np.where(scores > threshold)[0]

# Plot Results
plt.figure(figsize=(12, 6))
plt.plot(anomaly_data, label="Time Series")
plt.scatter(anomalies, anomaly_data[anomalies], color='red', label="Anomalies", zorder=5)
plt.title("Anomaly Detection with Isolation Forest")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.savefig('anomaly_detection_isolation_forest.png')
plt.show()

logger.info(f"Number of anomalies detected: {len(anomalies)}")
logger.info(f"Anomaly indices: {anomalies}")



def generate_data(n_points=1000, anomaly_start=700, anomaly_end=710):
    time = np.arange(n_points)
    data = np.sin(0.02 * time) + np.random.normal(0, 0.1, n_points)
    data[anomaly_start:anomaly_end] += 3  # Inject anomalies
    return data

def prepare_data(data, window_size=20):
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data.reshape(-1, 1))
    X = np.array([data_scaled[i:i+window_size] for i in range(len(data_scaled) - window_size)])
    return X, scaler

def create_model(window_size):
    model = Sequential([
        LSTM(32, activation='relu', input_shape=(window_size, 1), return_sequences=True),
        LSTM(16, activation='relu', return_sequences=False),
        RepeatVector(window_size),
        LSTM(16, activation='relu', return_sequences=True),
        LSTM(32, activation='relu', return_sequences=True),
        TimeDistributed(Dense(1))
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def detect_anomalies(reconstruction_error, threshold=3):
    mean = np.mean(reconstruction_error)
    std = np.std(reconstruction_error)
    return reconstruction_error > mean + threshold * std

def plot_results(data, reconstruction_error, threshold=3, window_size=20):
    plt.figure(figsize=(12, 6))
    
    # Plot the original data
    plt.plot(data[window_size:], label="Time Series")
    
    # Detect anomalies
    mean = np.mean(reconstruction_error)
    std = np.std(reconstruction_error)
    anomalies = reconstruction_error > mean + threshold * std
    
    # Plot anomalies
    anomaly_indices = np.where(anomalies)[0] + window_size
    plt.scatter(anomaly_indices, data[anomaly_indices], color='red', label="Anomalies")
    
    plt.title("Anomaly Detection with LSTM Autoencoder")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.savefig('lstm_anomaly_detection.jpg')
    plt.show()
    
    return anomalies


# Main execution
if __name__ == "__main__":
    # Generate data
    data = generate_data()

    # Prepare data
    X, scaler = prepare_data(data)

    # Create and train model
    model = create_model(window_size=20)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X, X, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=1)

    # Predict and calculate reconstruction error
    X_pred = model.predict(X)
    reconstruction_error = np.mean(np.abs(X - X_pred), axis=(1, 2))

    # Detect anomalies
    anomalies = detect_anomalies(reconstruction_error)

    # Plot results
    plot_results(data, anomalies, window_size=20)

    logger.info(f"Number of anomalies detected: {np.sum(anomalies)}")


np.random.seed(42)
torch.manual_seed(42)
plt.rcParams.update({
    'axes.grid': False,'font.family': 'serif','axes.spines.top': False,'axes.spines.right': False,'axes.linewidth': 0.8})

def save_fig(path: str):
    plt.tight_layout(); plt.savefig(path, bbox_inches='tight'); plt.close()

@dataclass
class Config:
    csv_path: str = "2001-2025 Net_generation_United_States_all_sectors_monthly.csv"
    freq: str = "MS"
    season: int = 12
    window: int = 24
    batch_size: int = 32
    epochs: int = 200
    lr: float = 1e-3
    z_thresh: float = 3.0  # threshold on recon error z-score


def load_series(cfg: Config) -> pd.Series:
    p = Path(cfg.csv_path)
    df = read_csv(p, header=None, usecols=[0,1], names=["date","value"], sep=",")
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    s = df.dropna().sort_values("date").set_index("date")["value"].asfreq(cfg.freq)
    return s.astype(float)


def stl_residuals(s: pd.Series, season: int) -> pd.Series:
    stl = STL(s, period=season, robust=True).fit()
    return stl.resid

class AE(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 16), nn.ReLU(),
            nn.Linear(16, 4)
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 16), nn.ReLU(),
            nn.Linear(16, 64), nn.ReLU(),
            nn.Linear(64, input_dim)
        )
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


def make_windows(x: np.ndarray, win: int) -> np.ndarray:
    if len(x) < win:
        return np.empty((0, win))
    return np.stack([x[i:i+win] for i in range(len(x)-win+1)], axis=0)


def train_autoencoder(X: np.ndarray, cfg: Config) -> Tuple[AE, np.ndarray]:
    device = torch.device('cpu')
    model = AE(X.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    ds = TensorDataset(torch.from_numpy(X).float())
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)

    model.train()
    for _ in range(cfg.epochs):
        for (xb,) in dl:
            xb = xb.to(device)
            opt.zero_grad()
            recon = model(xb)
            loss = loss_fn(recon, xb)
            loss.backward()
            opt.step()

    # Reconstruction errors (per window)
    model.eval()
    with torch.no_grad():
        Xten = torch.from_numpy(X).float().to(device)
        R = model(Xten).cpu().numpy()
    errs = np.mean((R - X)**2, axis=1)
    return model, errs


def main():
    cfg = Config()
    s = load_series(cfg)

    # Physics-informed: learn on STL residual windows (trend/season removed)
    resid = stl_residuals(s, cfg.season).dropna()
    # Standardize residuals for training stability
    mu, sd = resid.mean(), resid.std(ddof=1)
    sd = sd if sd else 1.0
    zres = (resid - mu) / sd

    X = make_windows(zres.values, cfg.window)
    if X.shape[0] == 0:
        raise SystemExit("Series too short for configured window size.")

    # Train on the middle 80% windows to reduce edge effects (still leakage-safe for detection use case)
    n = X.shape[0]
    lo, hi = int(0.1*n), int(0.9*n)
    X_train = X[lo:hi]

    model, errs = train_autoencoder(X_train, cfg)

    # Score all windows using the trained AE
    with torch.no_grad():
        X_all = torch.from_numpy(X).float()
        R_all = model(X_all).cpu().numpy()
        all_errs = np.mean((R_all - X)**2, axis=1)

    # Map window error back to the end timestamp of each window
    err_idx = resid.index[cfg.window-1:]
    err_s = pd.Series(all_errs, index=err_idx)

    # Z-score thresholding on reconstruction error
    e_mu, e_sd = err_s.mean(), err_s.std(ddof=1)
    e_sd = e_sd if e_sd else 1.0
    z = (err_s - e_mu) / e_sd
    anomalies = z > cfg.z_thresh

    # Plot on original series
    plt.figure(figsize=(10,5))
    plt.plot(s.index, s.values, label='EIA series', alpha=0.7)
    if anomalies.any():
        ts_anom = err_s.index[anomalies]
        vals = s.reindex(ts_anom).values
        plt.scatter(ts_anom, vals, color='red', s=24, label='AE anomaly')
    plt.legend()
    save_fig('eia_anomaly_autoencoder.png')

    # Also show error time series
    plt.figure(figsize=(10,3))
    plt.plot(err_s.index, err_s.values, label='Recon error')
    plt.axhline(e_mu + cfg.z_thresh*e_sd, color='red', lw=0.8, linestyle='--', label='threshold')
    plt.legend()
    save_fig('eia_anomaly_autoencoder_error.png')

if __name__ == '__main__':
    main()
