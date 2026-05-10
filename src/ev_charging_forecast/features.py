"""Feature engineering for station-hour EV charging forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    CALENDAR_FEATURES,
    CATEGORICAL_FEATURES,
    ENERGY_TARGET,
    EWM_SESSION_SPANS,
    GLOBAL_ROLLING_WINDOWS,
    GLOBAL_SESSION_LAGS,
    ROLLING_SESSION_WINDOWS,
    SERIES_KEYS,
    STATION_SESSION_LAGS,
    TARGET,
)


def add_calendar_features(df: pd.DataFrame, ts_col: str = "hour_ts") -> pd.DataFrame:
    out = df.copy()
    ts = out[ts_col]
    out["hour"] = ts.dt.hour
    out["day_of_week"] = ts.dt.dayofweek
    out["month"] = ts.dt.month
    out["quarter"] = ts.dt.quarter
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    out["is_peak_hour"] = (out["hour"].between(7, 10) | out["hour"].between(16, 20)).astype(int)
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["dow_sin"] = np.sin(2 * np.pi * out["day_of_week"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["day_of_week"] / 7)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    return out


def add_lag_features(demand: pd.DataFrame) -> pd.DataFrame:
    """Create leakage-safe lags, rolling features, EWM features, and global load features."""
    df = add_calendar_features(demand).sort_values([*SERIES_KEYS, "hour_ts"]).reset_index(drop=True)
    station_group = df.groupby(SERIES_KEYS, group_keys=False)

    for lag in STATION_SESSION_LAGS:
        df[f"lag_{lag}h_sessions"] = station_group[TARGET].shift(lag)
        df[f"lag_{lag}h_energy"] = station_group[ENERGY_TARGET].shift(lag)

    for window in ROLLING_SESSION_WINDOWS:
        min_periods = max(2, min(window, 8))
        df[f"rolling_{window}h_sessions"] = station_group[TARGET].transform(
            lambda s, w=window, m=min_periods: s.shift(1).rolling(w, min_periods=m).mean()
        )
        df[f"rolling_{window}h_sessions_std"] = station_group[TARGET].transform(
            lambda s, w=window: s.shift(1).rolling(w, min_periods=max(3, min(w, 8))).std()
        )
        df[f"rolling_max_{window}h_sessions"] = station_group[TARGET].transform(
            lambda s, w=window, m=min_periods: s.shift(1).rolling(w, min_periods=m).max()
        )
        df[f"rolling_sum_{window}h_sessions"] = station_group[TARGET].transform(
            lambda s, w=window, m=min_periods: s.shift(1).rolling(w, min_periods=m).sum()
        )
        df[f"rolling_{window}h_energy"] = station_group[ENERGY_TARGET].transform(
            lambda s, w=window, m=min_periods: s.shift(1).rolling(w, min_periods=m).mean()
        )

    for span in EWM_SESSION_SPANS:
        df[f"station_ewm_{span}h_sessions"] = station_group[TARGET].transform(
            lambda s, sp=span: s.shift(1).ewm(span=sp, adjust=False, min_periods=2).mean()
        )
        df[f"station_ewm_{span}h_energy"] = station_group[ENERGY_TARGET].transform(
            lambda s, sp=span: s.shift(1).ewm(span=sp, adjust=False, min_periods=2).mean()
        )

    df["recent_trend_1h_3h"] = df["lag_1h_sessions"] - df["lag_3h_sessions"]
    df["recent_ratio_1h_3h"] = df["lag_1h_sessions"] / (df["lag_3h_sessions"] + 1.0)
    df["recent_acceleration_1h_2h_3h"] = df["lag_1h_sessions"] - 2 * df["lag_2h_sessions"] + df["lag_3h_sessions"]
    df["lag_24h_to_168h_ratio"] = df["lag_24h_sessions"] / (df["lag_168h_sessions"] + 1.0)
    df["lag_167h_168h_169h_mean"] = df[["lag_167h_sessions", "lag_168h_sessions", "lag_169h_sessions"]].mean(axis=1)
    df["station_expanding_mean"] = station_group[TARGET].transform(lambda s: s.shift(1).expanding(min_periods=2).mean())
    df["station_dow_hour_expanding_mean"] = df.groupby([*SERIES_KEYS, "day_of_week", "hour"])[TARGET].transform(
        lambda s: s.shift(1).expanding(min_periods=2).mean()
    )

    hourly_total = (
        df.groupby("hour_ts", as_index=False)[TARGET]
        .sum()
        .rename(columns={TARGET: "global_session_count"})
        .sort_values("hour_ts")
    )
    for lag in GLOBAL_SESSION_LAGS:
        hourly_total[f"global_lag_{lag}h_sessions"] = hourly_total["global_session_count"].shift(lag)
    for window in GLOBAL_ROLLING_WINDOWS:
        hourly_total[f"global_rolling_{window}h_sessions"] = hourly_total["global_session_count"].shift(1).rolling(
            window, min_periods=max(2, min(window, 8))
        ).mean()

    df = df.merge(hourly_total.drop(columns="global_session_count"), on="hour_ts", how="left")
    df["station_share_lag_24h"] = df["lag_24h_sessions"] / (df["global_lag_24h_sessions"] + 1.0)
    df["station_share_lag_168h"] = df["lag_168h_sessions"] / (df["global_lag_168h_sessions"] + 1.0)
    return df.sort_values("hour_ts").reset_index(drop=True)


def candidate_numeric_features(df: pd.DataFrame) -> list[str]:
    excluded = set(CATEGORICAL_FEATURES + ["hour_ts", TARGET, ENERGY_TARGET])
    prefixes = ("lag_", "rolling_", "station_", "global_", "recent_")
    engineered = [
        col
        for col in df.columns
        if col not in excluded and (col in CALENDAR_FEATURES or col.startswith(prefixes) or col.endswith("_ratio"))
    ]
    return sorted(dict.fromkeys([*CALENDAR_FEATURES, *engineered]))


def audit_numeric_features(train: pd.DataFrame, features: list[str], min_non_null_fraction: float = 0.35) -> tuple[list[str], pd.DataFrame]:
    rows = []
    kept = []
    for feature in features:
        series = pd.to_numeric(train.get(feature), errors="coerce").replace([np.inf, -np.inf], np.nan)
        non_null = float(series.notna().mean()) if len(series) else 0.0
        unique = int(series.nunique(dropna=True))
        if feature not in train.columns:
            status = "missing"
        elif non_null == 0:
            status = "empty"
        elif non_null < min_non_null_fraction:
            status = "low_coverage"
        elif unique <= 1:
            status = "constant"
        else:
            status = "keep"
            kept.append(feature)
        rows.append({"feature": feature, "non_null_fraction": non_null, "n_unique": unique, "status": status})
    return kept, pd.DataFrame(rows).sort_values(["status", "non_null_fraction"], ascending=[True, False])


def required_history_column(df: pd.DataFrame) -> str:
    if "lag_168h_sessions" in df.columns and df["lag_168h_sessions"].notna().any():
        return "lag_168h_sessions"
    return "lag_24h_sessions"
