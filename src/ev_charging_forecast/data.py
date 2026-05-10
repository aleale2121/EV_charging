"""Data loading and active station-hour target construction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import SERIES_KEYS, TARGET


COLUMN_RENAME_MAP = {
    "Date": "date",
    "Station Name": "station_name",
    "Location Name": "location_name",
    "Country": "country",
    "Charge Box ID": "charge_box_id",
    "Connector ID": "connector_id",
    "Driver ID": "driver_id",
    "ID Tag": "id_tag",
    "Connected Time": "connected_time",
    "Disconnected Time": "disconnected_time",
    "Charge Duration (min)": "charge_duration_min",
    "Connected Duration (min)": "connected_duration_min",
    "Energy Provided (kWh)": "energy_provided_kwh",
    "Session Status": "session_status",
    "Invalidity Reason": "invalidity_reason",
}

NUMERIC_SESSION_COLUMNS = ["charge_duration_min", "connected_duration_min", "energy_provided_kwh"]


def load_data(path: Path) -> pd.DataFrame:
    """Load the raw event-level charging-session CSV."""
    return pd.read_csv(path)


def inspect_schema(df: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    """Write a compact schema profile for reproducibility."""
    output_dir.mkdir(parents=True, exist_ok=True)
    profile = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "column_names": df.columns.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_fraction": df.isna().mean().round(5).to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
    }
    (output_dir / "schema_profile.json").write_text(json.dumps(profile, indent=2))
    return profile


def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    clean_names = {col: col.strip().lower().replace(" ", "_") for col in df.columns}
    return df.rename(columns=COLUMN_RENAME_MAP).rename(columns=clean_names)


def parse_session_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Parse Date + Connected/Disconnected Time and fix overnight sessions."""
    out = df.copy()
    date_text = out["date"].astype(str).str.strip()
    date = pd.to_datetime(date_text, format="%m/%d/%Y", errors="coerce")
    missing_date = date.isna()
    if missing_date.any():
        date.loc[missing_date] = pd.to_datetime(date_text.loc[missing_date], errors="coerce")

    start_offset = pd.to_timedelta(out["connected_time"].astype(str).str.strip(), errors="coerce")
    end_offset = pd.to_timedelta(out["disconnected_time"].astype(str).str.strip(), errors="coerce")
    out["session_start"] = date + start_offset
    out["session_end"] = date + end_offset
    overnight = out["session_end"].notna() & out["session_start"].notna() & (out["session_end"] <= out["session_start"])
    out.loc[overnight, "session_end"] = out.loc[overnight, "session_end"] + pd.Timedelta(days=1)
    out["parsed_overnight_session"] = overnight.astype(int)
    return out


def clean_sessions(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean event-level sessions using the same assumptions as the notebook."""
    clean = standardize_columns(df)
    required = ["date", "connected_time", "disconnected_time", "station_name", "location_name", *NUMERIC_SESSION_COLUMNS]
    require_columns(clean, required)

    input_rows = len(clean)
    clean = clean.drop_duplicates()
    duplicates_removed = input_rows - len(clean)

    for col in NUMERIC_SESSION_COLUMNS:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")

    clean = parse_session_timestamps(clean)
    clean["station_name"] = clean["station_name"].astype(str).str.strip().replace({"": np.nan, "nan": np.nan})
    clean["location_name"] = clean["location_name"].astype(str).str.strip().replace({"": np.nan, "nan": np.nan})

    before_missing = len(clean)
    clean = clean.dropna(subset=["session_start", "session_end", "station_name", "location_name", *NUMERIC_SESSION_COLUMNS])
    missing_removed = before_missing - len(clean)

    before_invalid = len(clean)
    clean = clean[clean["session_end"] > clean["session_start"]]
    clean = clean[clean["energy_provided_kwh"] >= 0]
    clean = clean[clean["charge_duration_min"] > 0]
    clean = clean[clean["connected_duration_min"] > 0]
    clean = clean[clean["connected_duration_min"] >= clean["charge_duration_min"]]
    clean["actual_connected_duration_min"] = (clean["session_end"] - clean["session_start"]).dt.total_seconds() / 60
    clean = clean[clean["actual_connected_duration_min"].between(1, 7 * 24 * 60)]
    invalid_removed = before_invalid - len(clean)

    report = {
        "input_rows": int(input_rows),
        "duplicates_removed": int(duplicates_removed),
        "missing_removed": int(missing_removed),
        "invalid_physical_sessions_removed": int(invalid_removed),
        "overnight_sessions": int(clean["parsed_overnight_session"].sum()),
        "overnight_session_fraction": float(clean["parsed_overnight_session"].mean()),
        "clean_rows": int(len(clean)),
        "session_start_min": str(clean["session_start"].min()),
        "session_start_max": str(clean["session_start"].max()),
    }
    return clean.sort_values("session_start").reset_index(drop=True), report


def write_cleaning_report(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "cleaning_report.json").write_text(json.dumps(report, indent=2))


def build_station_hour_demand(sessions: pd.DataFrame) -> pd.DataFrame:
    """Build active station-hour demand by expanding each session across connected hours."""
    required = ["station_name", "location_name", "session_start", "session_end", "energy_provided_kwh", "charge_duration_min"]
    require_columns(sessions, required)

    records: list[dict[str, Any]] = []
    for row in sessions[required].itertuples(index=False):
        station, location, start, end, energy_kwh, charge_duration = row
        if pd.isna(start) or pd.isna(end) or end <= start:
            continue
        first_hour = start.floor("h")
        last_hour = (end - pd.Timedelta(nanoseconds=1)).floor("h")
        connected_minutes = max((end - start).total_seconds() / 60, 1e-9)
        for hour_ts in pd.date_range(first_hour, last_hour, freq="h"):
            hour_start = hour_ts
            hour_end = hour_ts + pd.Timedelta(hours=1)
            overlap_start = max(start, hour_start)
            overlap_end = min(end, hour_end)
            overlap_minutes = max((overlap_end - overlap_start).total_seconds() / 60, 0)
            if overlap_minutes <= 0:
                continue
            records.append(
                {
                    "station_name": station,
                    "location_name": location,
                    "hour_ts": hour_ts,
                    "active_session_count": 1,
                    "session_starts": int(start.floor("h") == hour_ts),
                    "occupied_minutes": overlap_minutes,
                    "total_energy_kwh": float(energy_kwh) * (overlap_minutes / connected_minutes),
                    "avg_duration_min_numer": float(charge_duration),
                }
            )

    expanded = pd.DataFrame.from_records(records)
    if expanded.empty:
        raise ValueError("No expanded station-hour records were created.")

    observed = (
        expanded.groupby([*SERIES_KEYS, "hour_ts"], as_index=False)
        .agg(
            session_count=("active_session_count", "sum"),
            session_starts=("session_starts", "sum"),
            total_energy_kwh=("total_energy_kwh", "sum"),
            occupied_minutes=("occupied_minutes", "sum"),
            avg_duration_min=("avg_duration_min_numer", "mean"),
        )
    )

    frames = []
    for (station, location), group in observed.groupby(SERIES_KEYS, sort=False):
        hours = pd.date_range(group["hour_ts"].min(), group["hour_ts"].max(), freq="h")
        full = pd.DataFrame({"station_name": station, "location_name": location, "hour_ts": hours})
        frames.append(full.merge(group, on=[*SERIES_KEYS, "hour_ts"], how="left"))

    demand = pd.concat(frames, ignore_index=True)
    fill_zero = ["session_count", "session_starts", "total_energy_kwh", "avg_duration_min", "occupied_minutes"]
    demand[fill_zero] = demand[fill_zero].fillna(0)
    demand["occupancy_rate_proxy"] = demand["occupied_minutes"] / 60.0
    return demand.sort_values(["hour_ts", *SERIES_KEYS]).reset_index(drop=True)


def summarize_demand(demand: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(demand)),
        "stations": int(demand["station_name"].nunique()),
        "locations": int(demand["location_name"].nunique()),
        "start": str(demand["hour_ts"].min()),
        "end": str(demand["hour_ts"].max()),
        "target_mean": float(demand[TARGET].mean()),
        "target_zero_fraction": float((demand[TARGET] == 0).mean()),
    }
