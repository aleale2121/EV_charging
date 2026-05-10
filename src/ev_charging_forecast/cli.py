"""Command-line entry point for the notebook-aligned EV forecast project."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .config import DEFAULT_DATASET_PATH, DEFAULT_OUTPUT_DIR, TARGET
from .data import build_station_hour_demand, clean_sessions, inspect_schema, load_data, summarize_demand, write_cleaning_report
from .features import add_lag_features, audit_numeric_features, candidate_numeric_features, required_history_column
from .forecast import recursive_station_hour_forecast
from .modeling import (
    compare_models,
    save_feature_importance,
    save_model_artifact,
    save_predictions,
    temporal_train_test_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EV active-session station-hour forecast models.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--forecast-horizon", type=int, default=168)
    parser.add_argument("--cv-splits", type=int, default=3)
    parser.add_argument("--max-rows", type=int, help="Optional raw CSV row limit for smoke tests.")
    parser.add_argument("--skip-forecast", action="store_true", help="Skip recursive future forecast.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw = load_data(args.input_csv)
    if args.max_rows:
        raw = raw.head(args.max_rows).copy()
    inspect_schema(raw, args.output_dir)

    sessions, cleaning_report = clean_sessions(raw)
    write_cleaning_report(cleaning_report, args.output_dir)

    demand = build_station_hour_demand(sessions)
    feature_df = add_lag_features(demand)
    required_lag = required_history_column(feature_df)
    model_df = feature_df.dropna(subset=[required_lag, "station_name", "location_name", TARGET]).reset_index(drop=True)

    train_df, test_df, cutoff = temporal_train_test_split(model_df, test_size=args.test_size)
    candidates = candidate_numeric_features(model_df)
    numeric_features, feature_audit = audit_numeric_features(train_df, candidates)
    feature_audit.to_csv(args.output_dir / "lag_feature_audit.csv", index=False)

    comparison, best_model, best_pred = compare_models(
        train_df,
        test_df,
        numeric_features,
        target=TARGET,
        cv_splits=args.cv_splits,
    )
    comparison.to_csv(args.output_dir / "demand_model_performance_table.csv", index=False)
    predictions = save_predictions(test_df, best_pred, args.output_dir)
    importance = save_feature_importance(best_model, test_df, numeric_features, args.output_dir)

    forecast_rows = 0
    if not args.skip_forecast:
        forecast = recursive_station_hour_forecast(best_model, model_df, numeric_features, horizon=args.forecast_horizon)
        forecast.to_csv(args.output_dir / "station_hour_forecast.csv", index=False)
        forecast_daily = (
            forecast.assign(forecast_date=pd.to_datetime(forecast["forecast_hour_ts"]).dt.date)
            .groupby(["forecast_date", "station_name", "location_name"], as_index=False)
            .agg(predicted_sessions=("predicted_session_count", "sum"))
            .sort_values(["forecast_date", "predicted_sessions"], ascending=[True, False])
        )
        forecast_daily.to_csv(args.output_dir / "station_daily_forecast.csv", index=False)
        forecast_rows = len(forecast)

    metadata = {
        "input_csv": str(args.input_csv),
        "output_dir": str(args.output_dir),
        "cleaning_report": cleaning_report,
        "demand_summary": summarize_demand(demand),
        "model_rows": int(len(model_df)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "cutoff": str(cutoff),
        "required_history_lag": required_lag,
        "numeric_feature_count": len(numeric_features),
        "best_model": str(comparison.iloc[0]["Model"]),
        "best_metrics": comparison.iloc[0].to_dict(),
        "prediction_rows": int(len(predictions)),
        "forecast_rows": int(forecast_rows),
        "top_features": importance.head(20).to_dict(orient="records"),
    }
    save_model_artifact(best_model, numeric_features, args.output_dir, metadata)
    (args.output_dir / "run_summary.json").write_text(json.dumps(metadata, indent=2, default=str))

    print(json.dumps({"output_dir": str(args.output_dir), "best_model": metadata["best_model"], "best_metrics": metadata["best_metrics"]}, indent=2))


if __name__ == "__main__":
    main()
