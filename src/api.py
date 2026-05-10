"""FastAPI service for EV charging forecast artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel


OUTPUT_DIR = Path("outputs/time_series_forecast_project")
NOTEBOOK_OUTPUT_DIR = Path("outputs/time_series_forecast_notebook")

app = FastAPI(title="EV Charging Active-Session Forecast API", version="2.0")


class FeaturePredictionRequest(BaseModel):
    """A fully engineered feature row for advanced/manual predictions."""

    features: dict[str, Any]


def active_output_dir() -> Path:
    if OUTPUT_DIR.exists():
        return OUTPUT_DIR
    if NOTEBOOK_OUTPUT_DIR.exists():
        return NOTEBOOK_OUTPUT_DIR
    raise HTTPException(status_code=503, detail="No forecast output directory found. Run the training project first.")


def read_csv(name: str) -> pd.DataFrame:
    path = active_output_dir() / name
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Artifact not found: {path}")
    return pd.read_csv(path)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "output_dir": str(active_output_dir())}


@app.get("/model-comparison")
def model_comparison() -> list[dict[str, Any]]:
    path_names = ["demand_model_performance_table.csv", "demand_all_model_comparison.csv"]
    for name in path_names:
        try:
            return read_csv(name).to_dict(orient="records")
        except HTTPException:
            continue
    raise HTTPException(status_code=404, detail="No model comparison artifact found.")


@app.get("/forecast/hourly")
def hourly_forecast(limit: int = Query(default=500, ge=1, le=10000)) -> list[dict[str, Any]]:
    return read_csv("station_hour_forecast.csv").head(limit).to_dict(orient="records")


@app.get("/forecast/daily")
def daily_forecast(limit: int = Query(default=100, ge=1, le=10000)) -> list[dict[str, Any]]:
    return read_csv("station_daily_forecast.csv").head(limit).to_dict(orient="records")


@app.get("/predictions")
def test_predictions(limit: int = Query(default=500, ge=1, le=10000)) -> list[dict[str, Any]]:
    return read_csv("demand_test_predictions.csv").head(limit).to_dict(orient="records")


@app.post("/predict-engineered")
def predict_engineered(payload: FeaturePredictionRequest) -> dict[str, float]:
    artifact_path = OUTPUT_DIR / "demand_model_artifact.joblib"
    if not artifact_path.exists():
        raise HTTPException(status_code=503, detail=f"Model artifact not found: {artifact_path}")
    artifact = joblib.load(artifact_path)
    columns = artifact["numeric_features"] + artifact["categorical_features"]
    row = pd.DataFrame([payload.features])
    missing = [col for col in columns if col not in row.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing engineered features: {missing}")
    pred = max(0.0, float(artifact["model"].predict(row[columns])[0]))
    return {"predicted_session_count": pred}
