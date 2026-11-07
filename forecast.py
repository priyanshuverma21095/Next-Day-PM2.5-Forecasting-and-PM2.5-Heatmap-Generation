"""CS771 Mini-Project 1: Part 5 Submission forecast.py

Implements forecast(history, issue_time) -> pd.Series
Loads the pre-fitted final model (0.5 x Ridge + 0.5 x Random Forest on 35 features, refitted on
all targets up to 30 Nov) and station locations once at import time. Nothing is refitted here.
Adheres strictly to the starter template API, time limit (<10s), and format.
"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

# Load fitted model and station coordinates once at import time
MODEL_PATH = HERE / "model.pkl"
MODEL = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None

STATIONS_PATH = HERE / "bihar_stations.csv"
if STATIONS_PATH.exists():
    STATIONS = pd.read_csv(STATIONS_PATH, dtype={"sensor_id": str}).set_index("sensor_id")
    _coords_x = (STATIONS["longitude"] - 85.5) * 100.3
    _coords_y = (STATIONS["latitude"] - 25.7) * 110.6
    _X_pos = _coords_x.to_numpy()
    _Y_pos = _coords_y.to_numpy()
    _dist_matrix = np.sqrt((_X_pos[:, None] - _X_pos[None, :]) ** 2 + (_Y_pos[:, None] - _Y_pos[None, :]) ** 2)
    np.fill_diagonal(_dist_matrix, np.inf)
    _NEAREST_10_IDX = np.argsort(_dist_matrix, axis=1)[:, :10]
    _NEAREST_25_IDX = np.argsort(_dist_matrix, axis=1)[:, :25]
else:
    STATIONS = None
    _NEAREST_10_IDX = None
    _NEAREST_25_IDX = None


def forecast(history: pd.DataFrame, issue_time: pd.Timestamp) -> pd.Series:
    """history: wide hourly table (DatetimeIndex 'timestamp', one string column per sensor_id),
    every hour from 1 Jan 2024 00:00 up to and including issue_time.
    issue_time: 12:00 on some day (IST).
    Returns predicted PM2.5 for issue_time + 24 h, indexed by sensor_id (all 468 stations)."""
    # 24-hour historical window ending at issue_time
    window = history.loc[issue_time - pd.Timedelta(hours=23) : issue_time]

    # If model or spatial indices unavailable, fall back to persistence baseline
    if MODEL is None or _NEAREST_10_IDX is None:
        prediction = window.iloc[-1].to_numpy(dtype=float)
        return pd.Series(np.clip(prediction, 0, None), index=history.columns, name="pm25_forecast")

    window_vals = window.to_numpy(dtype=float)  # shape (24, 468)
    readings_24 = window_vals[::-1, :].T        # shape (468, 24) [t, t-1, ..., t-23]
    at_t = window_vals[-1, :]                   # shape (468,)
    mean_24 = window_vals.mean(axis=0)          # shape (468,)

    # Spatial neighborhood features
    n10_t = at_t[_NEAREST_10_IDX].mean(axis=1)[:, None]
    n10_24 = mean_24[_NEAREST_10_IDX].mean(axis=1)[:, None]
    n25_t = at_t[_NEAREST_25_IDX].mean(axis=1)[:, None]
    n25_24 = mean_24[_NEAREST_25_IDX].mean(axis=1)[:, None]

    # Network-wide context
    net_t = np.full((len(history.columns), 1), at_t.mean())
    net_24 = np.full((len(history.columns), 1), mean_24.mean())
    net_std = np.full((len(history.columns), 1), at_t.std())

    # Temporal calendar features
    d = (issue_time - pd.Timestamp("2024-01-01")).days
    sin_d = np.full((len(history.columns), 1), np.sin(2 * np.pi * d / 365.0))
    cos_d = np.full((len(history.columns), 1), np.cos(2 * np.pi * d / 365.0))
    dow = issue_time.dayofweek
    sin_dow = np.full((len(history.columns), 1), np.sin(2 * np.pi * dow / 7.0))
    cos_dow = np.full((len(history.columns), 1), np.cos(2 * np.pi * dow / 7.0))

    # Feature matrix (468, 35)
    X = np.hstack([readings_24, n10_t, n10_24, n25_t, n25_24, net_t, net_24, net_std, sin_d, cos_d, sin_dow, cos_dow])

    w = MODEL["w_ridge"]
    prediction = (w * np.clip(MODEL["ridge"].predict(X), 0, None)
                  + (1 - w) * np.clip(MODEL["rf"].predict(X), 0, None))
    return pd.Series(np.clip(prediction, 0, None), index=history.columns, name="pm25_forecast")
