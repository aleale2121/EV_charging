"""Recursive future station-hour forecasting."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import (
    CATEGORICAL_FEATURES,
    EWM_SESSION_SPANS,
    GLOBAL_ROLLING_WINDOWS,
    GLOBAL_SESSION_LAGS,
    ROLLING_SESSION_WINDOWS,
    SERIES_KEYS,
    STATION_SESSION_LAGS,
    TARGET,
)
from .features import add_calendar_features


def make_forecast_features_from_history(history: pd.DataFrame, forecast_ts: pd.Timestamp) -> pd.DataFrame:
    histories = {
        key: group.set_index("hour_ts")[TARGET].sort_index().astype(float)
        for key, group in history.groupby(SERIES_KEYS, sort=False)
    }
    energy_histories = {
        key: group.set_index("hour_ts")["total_energy_kwh"].sort_index().astype(float)
        for key, group in history.groupby(SERIES_KEYS, sort=False)
    }
    global_series = history.groupby("hour_ts")[TARGET].sum().sort_index().astype(float)

    rows = []
    for station, location in histories:
        rows.append(
            {
                "hour_ts": forecast_ts,
                "station_name": station,
                "location_name": location,
                "session_count": np.nan,
                "session_starts": 0.0,
                "total_energy_kwh": 0.0,
                "occupied_minutes": 0.0,
                "avg_duration_min": 0.0,
                "occupancy_rate_proxy": 0.0,
            }
        )
    future = add_calendar_features(pd.DataFrame(rows))

    feature_rows = []
    for _, row in future.iterrows():
        key = (row["station_name"], row["location_name"])
        series = histories[key]
        energy = energy_histories[key]
        station_history = history[(history["station_name"] == key[0]) & (history["location_name"] == key[1])]
        item = row.to_dict()

        for lag in STATION_SESSION_LAGS:
            item[f"lag_{lag}h_sessions"] = float(series.iloc[-lag]) if len(series) >= lag else np.nan
            item[f"lag_{lag}h_energy"] = float(energy.iloc[-lag]) if len(energy) >= lag else np.nan
        for window in ROLLING_SESSION_WINDOWS:
            item[f"rolling_{window}h_sessions"] = float(series.tail(window).mean()) if len(series) >= 2 else np.nan
            item[f"rolling_{window}h_sessions_std"] = float(series.tail(window).std()) if len(series) >= 3 else np.nan
            item[f"rolling_max_{window}h_sessions"] = float(series.tail(window).max()) if len(series) >= 2 else np.nan
            item[f"rolling_sum_{window}h_sessions"] = float(series.tail(window).sum()) if len(series) >= 2 else np.nan
            item[f"rolling_{window}h_energy"] = float(energy.tail(window).mean()) if len(energy) >= 2 else np.nan
        for span in EWM_SESSION_SPANS:
            item[f"station_ewm_{span}h_sessions"] = float(series.ewm(span=span, adjust=False, min_periods=2).mean().iloc[-1]) if len(series) >= 2 else np.nan
            item[f"station_ewm_{span}h_energy"] = float(energy.ewm(span=span, adjust=False, min_periods=2).mean().iloc[-1]) if len(energy) >= 2 else np.nan

        item["recent_trend_1h_3h"] = item.get("lag_1h_sessions", np.nan) - item.get("lag_3h_sessions", np.nan)
        item["recent_ratio_1h_3h"] = item.get("lag_1h_sessions", np.nan) / (item.get("lag_3h_sessions", np.nan) + 1.0)
        item["recent_acceleration_1h_2h_3h"] = item.get("lag_1h_sessions", np.nan) - 2 * item.get("lag_2h_sessions", np.nan) + item.get("lag_3h_sessions", np.nan)
        item["lag_24h_to_168h_ratio"] = item.get("lag_24h_sessions", np.nan) / (item.get("lag_168h_sessions", np.nan) + 1.0)
        item["lag_167h_168h_169h_mean"] = np.nanmean(
            [item.get("lag_167h_sessions", np.nan), item.get("lag_168h_sessions", np.nan), item.get("lag_169h_sessions", np.nan)]
        )
        item["station_expanding_mean"] = float(series.mean()) if len(series) >= 2 else np.nan
        item["station_dow_hour_expanding_mean"] = float(
            station_history[(station_history["day_of_week"] == forecast_ts.dayofweek) & (station_history["hour"] == forecast_ts.hour)][TARGET].mean()
        )
        for lag in GLOBAL_SESSION_LAGS:
            item[f"global_lag_{lag}h_sessions"] = float(global_series.iloc[-lag]) if len(global_series) >= lag else np.nan
        for window in GLOBAL_ROLLING_WINDOWS:
            item[f"global_rolling_{window}h_sessions"] = float(global_series.tail(window).mean()) if len(global_series) >= 2 else np.nan
        item["station_share_lag_24h"] = item.get("lag_24h_sessions", np.nan) / (item.get("global_lag_24h_sessions", np.nan) + 1.0)
        item["station_share_lag_168h"] = item.get("lag_168h_sessions", np.nan) / (item.get("global_lag_168h_sessions", np.nan) + 1.0)
        feature_rows.append(item)

    return pd.DataFrame(feature_rows)


def recursive_station_hour_forecast(
    model: Any,
    history: pd.DataFrame,
    numeric_features: list[str],
    horizon: int = 168,
) -> pd.DataFrame:
    rolling_history = history.copy().sort_values([*SERIES_KEYS, "hour_ts"]).reset_index(drop=True)
    start_ts = rolling_history["hour_ts"].max() + pd.Timedelta(hours=1)
    forecast_rows = []

    for step in range(1, horizon + 1):
        ts = start_ts + pd.Timedelta(hours=step - 1)
        batch = make_forecast_features_from_history(rolling_history, ts)
        pred = np.maximum(0, model.predict(batch[numeric_features + CATEGORICAL_FEATURES]))
        batch["predicted_session_count"] = pred
        batch["forecast_step"] = step
        forecast_rows.append(batch[["hour_ts", "forecast_step", "station_name", "location_name", "predicted_session_count"]])

        append_rows = batch[["hour_ts", "station_name", "location_name"]].copy()
        append_rows["session_count"] = pred
        append_rows["session_starts"] = 0.0
        append_rows["total_energy_kwh"] = 0.0
        append_rows["occupied_minutes"] = 0.0
        append_rows["avg_duration_min"] = 0.0
        append_rows["occupancy_rate_proxy"] = 0.0
        append_rows = add_calendar_features(append_rows)
        rolling_history = pd.concat([rolling_history, append_rows], ignore_index=True, sort=False)

    return pd.concat(forecast_rows, ignore_index=True).rename(columns={"hour_ts": "forecast_hour_ts"})
