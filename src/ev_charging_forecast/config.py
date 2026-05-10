"""Shared configuration for the EV charging forecast project."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_PATH = Path("/Users/alefew/Pictures/dataset.csv")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "time_series_forecast_project"

DEFAULT_RANDOM_STATE = 42
TARGET = "session_count"
ENERGY_TARGET = "total_energy_kwh"
SERIES_KEYS = ["station_name", "location_name"]
CATEGORICAL_FEATURES = ["station_name", "location_name"]

STATION_SESSION_LAGS = [1, 2, 3, 6, 12, 24, 48, 72, 167, 168, 169, 336]
ROLLING_SESSION_WINDOWS = [3, 6, 12, 24, 72, 168]
EWM_SESSION_SPANS = [12, 24, 72]
GLOBAL_SESSION_LAGS = [1, 24, 168]
GLOBAL_ROLLING_WINDOWS = [24, 168]

CALENDAR_FEATURES = [
    "hour",
    "day_of_week",
    "month",
    "quarter",
    "is_weekend",
    "is_peak_hour",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
]
