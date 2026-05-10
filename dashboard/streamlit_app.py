"""Streamlit dashboard for EV charging forecast outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


DEFAULT_OUTPUT_DIRS = [
    Path("outputs/time_series_forecast_notebook"),
]


def discover_output_dirs() -> list[Path]:
    dirs = [path for path in DEFAULT_OUTPUT_DIRS if path.exists()]
    outputs_root = Path("outputs")
    if outputs_root.exists():
        for path in sorted(outputs_root.iterdir()):
            if path.is_dir() and path not in dirs:
                dirs.append(path)
    return dirs


@st.cache_data(show_spinner=False)
def read_csv_sample(path_text: str, max_rows: int) -> pd.DataFrame:
    path = Path(path_text)
    try:
        return pd.read_csv(path, nrows=max_rows)
    except UnicodeDecodeError:
        return pd.read_csv(path, nrows=max_rows, encoding="latin1")


def read_optional_csv(output_dir: Path, *names: str) -> pd.DataFrame:
    for name in names:
        path = output_dir / name
        if path.exists():
            return read_csv_sample(str(path), 250_000)
    return pd.DataFrame()


def file_inventory(output_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(output_dir.iterdir()):
        if not path.is_file():
            continue
        rows.append(
            {
                "file": path.name,
                "type": path.suffix.lower().lstrip(".") or "file",
                "size_mb": round(path.stat().st_size / 1024 / 1024, 3),
                "modified": pd.to_datetime(path.stat().st_mtime, unit="s"),
            }
        )
    return pd.DataFrame(rows)


def r2_column(df: pd.DataFrame) -> str | None:
    for col in ["R²", "R2", "R² (%)", "CV_R2_mean"]:
        if col in df.columns:
            return col
    return None


def wmape_column(df: pd.DataFrame) -> str | None:
    for col in ["WMAPE (%)", "WMAPE_pct"]:
        if col in df.columns:
            return col
    return None


def best_row(comparison: pd.DataFrame) -> pd.Series:
    if {"RMSE", "MAE"}.issubset(comparison.columns):
        return comparison.sort_values(["RMSE", "MAE"]).iloc[0]
    return comparison.iloc[0]


def show_metrics(comparison: pd.DataFrame, title: str, unit_note: str = "") -> None:
    if comparison.empty:
        st.info(f"No {title.lower()} model table found.")
        return
    best = best_row(comparison)
    r2_col = r2_column(comparison)
    wmape_col = wmape_column(comparison)
    st.subheader(title)
    if unit_note:
        st.caption(unit_note)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Best Model", str(best.get("Model", "unknown")))
    col2.metric("MAE", f"{float(best.get('MAE', 0)):.3f}" if "MAE" in best else "n/a")
    col3.metric("RMSE", f"{float(best.get('RMSE', 0)):.3f}" if "RMSE" in best else "n/a")
    col4.metric("R²", f"{float(best.get(r2_col, 0)):.3f}" if r2_col else "n/a")
    if wmape_col:
        st.caption(f"Best WMAPE: {float(best.get(wmape_col, 0)):.2f}%")
    st.dataframe(comparison, use_container_width=True)


def show_prediction_charts(predictions: pd.DataFrame) -> None:
    if predictions.empty:
        st.info("No prediction CSV found in this output folder.")
        return
    pred_col = "predicted_session_count"
    if pred_col not in predictions.columns:
        st.info("Prediction file exists, but no `predicted_session_count` column was found.")
        st.dataframe(predictions.head(1000), use_container_width=True)
        return
    actual_col = "session_count" if "session_count" in predictions.columns else predictions.columns[3]
    chart_sample = predictions.sample(min(len(predictions), 50_000), random_state=42) if len(predictions) > 50_000 else predictions
    st.scatter_chart(chart_sample, x=actual_col, y=pred_col)
    if "hour_ts" in predictions.columns:
        hourly = predictions.groupby("hour_ts", as_index=False).agg(actual=(actual_col, "sum"), predicted=(pred_col, "sum"))
        st.line_chart(hourly, x="hour_ts", y=["actual", "predicted"])


def show_forecast(hourly_forecast: pd.DataFrame, daily_forecast: pd.DataFrame) -> None:
    if hourly_forecast.empty and daily_forecast.empty:
        st.info("No future forecast CSVs found in this output folder.")
        return
    if not hourly_forecast.empty and "predicted_session_count" in hourly_forecast.columns:
        ts_col = "forecast_hour_ts" if "forecast_hour_ts" in hourly_forecast.columns else "hour_ts"
        system_forecast = hourly_forecast.groupby(ts_col, as_index=False)["predicted_session_count"].sum()
        st.subheader("System Demand Forecast")
        st.line_chart(system_forecast, x=ts_col, y="predicted_session_count")
    if not daily_forecast.empty:
        st.subheader("Daily Station Forecast")
        sort_col = "predicted_sessions" if "predicted_sessions" in daily_forecast.columns else daily_forecast.columns[-1]
        st.dataframe(daily_forecast.sort_values(sort_col, ascending=False).head(100), use_container_width=True)


def sorted_importance(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    value_col = "importance_mean" if "importance_mean" in df.columns else df.columns[-1]
    return df.sort_values(value_col, ascending=False).reset_index(drop=True)


def show_importance(title: str, importance: pd.DataFrame) -> None:
    if importance.empty or "feature" not in importance.columns:
        st.info(f"No {title.lower()} feature-importance file found.")
        return
    importance = sorted_importance(importance)
    value_col = "importance_mean" if "importance_mean" in importance.columns else importance.columns[-1]
    st.subheader(title)
    st.bar_chart(importance.head(30).set_index("feature")[[value_col]])
    st.dataframe(importance, use_container_width=True)


def show_user_behavior_clustering(output_dir: Path) -> None:
    labeled = read_optional_csv(output_dir, "user_behavior_cluster_summary_k6_labeled.csv")
    simple = read_optional_csv(output_dir, "user_behavior_cluster_summary_k6.csv", "user_behavior_cluster_summary.csv")
    silhouette = read_optional_csv(output_dir, "user_behavior_silhouette_scores.csv")
    summary = labeled if not labeled.empty else simple

    st.subheader("User Behavior Clustering")
    if summary.empty:
        st.info("No user-behavior clustering summary was found.")
        return

    user_col = "user_count" if "user_count" in summary.columns else None
    cluster_col = "cluster_label" if "cluster_label" in summary.columns else "user_cluster"
    total_users = int(summary[user_col].sum()) if user_col else len(summary)
    largest = summary.sort_values(user_col, ascending=False).iloc[0] if user_col else summary.iloc[0]

    col1, col2, col3 = st.columns(3)
    col1.metric("Clusters", len(summary))
    col2.metric("Users Clustered", f"{total_users:,}")
    col3.metric("Largest Segment", str(largest.get(cluster_col, largest.get("user_cluster", "n/a"))))

    if not silhouette.empty and {"k", "silhouette"}.issubset(silhouette.columns):
        best_k = silhouette.sort_values("silhouette", ascending=False).iloc[0]
        st.caption(f"Best silhouette in saved search: k={int(best_k['k'])}, score={float(best_k['silhouette']):.3f}")

    preferred_cols = [
        "user_cluster",
        "cluster_label",
        "description",
        "user_count",
        "session_count",
        "avg_energy_kwh",
        "total_energy_kwh",
        "avg_duration_min",
        "avg_idle_min",
        "avg_power_kw",
        "station_count",
        "location_count",
        "weekend_rate",
        "peak_hour_rate",
        "silhouette_k6",
    ]
    cols = [col for col in preferred_cols if col in summary.columns]
    sort_cols = ["user_count"] if "user_count" in summary.columns else ["user_cluster"]
    st.dataframe(summary[cols].sort_values(sort_cols, ascending=False), use_container_width=True)

    cluster_images = [
        "user_behavior_clusters_k6_labeled_clean.png",
        "user_behavior_clusters_k6_labeled_with_descriptions.png",
        "user_behavior_silhouette_comparison.png",
    ]
    existing = [name for name in cluster_images if (output_dir / name).exists()]
    if existing:
        st.subheader("User Behavior Cluster Visuals")
        for name in existing[:2]:
            st.image(str(output_dir / name), caption=name, use_container_width=True)


IMPORTANT_IMAGE_ORDER = [
    "model_comparison_r2_percent_demand_energy.png",
    "model_comparison_mae_demand_energy.png",
    "model_comparison_rmse_demand_energy.png",
    "best_model_r2_improvement_over_baseline.png",
    "demand_forecast_evaluation.png",
    "energy_forecast_evaluation.png",
    "user_behavior_clusters_k6_labeled_clean.png",
    "user_behavior_clusters_k6_labeled_with_descriptions.png",
    "user_behavior_silhouette_comparison.png",
    "demand_feature_importance.png",
    "energy_feature_importance.png",
    "seven_day_system_demand_forecast.png",
    "target_analysis_figures.png",
    "eda_demand_heatmap_day_hour.png",
    "eda_energy_heatmap_day_hour.png",
    "eda_demand_vs_energy_scatter.png",
    "feature_engineering_lag_correlation_demand.png",
    "feature_engineering_group_summary.png",
    "user_behavior_clusters.png",
    "user_behavior_clusters_k6.png",
    "energy_usage_clusters_pca.png",
    "energy_usage_cluster_counts.png",
    "avg_total_energy_by_cluster.png",
    "avg_energy_per_session_by_cluster.png",
    "sessions_over_time.png",
    "temporal_train_test_split.png",
]


def ordered_images(image_files: list[str]) -> list[str]:
    available = set(image_files)
    ordered = [name for name in IMPORTANT_IMAGE_ORDER if name in available]
    ordered.extend(name for name in image_files if name not in set(ordered))
    return ordered


st.set_page_config(page_title="EV Charging Forecast", layout="wide")
st.title("EV Charging Forecast Output Dashboard")

output_dirs = discover_output_dirs()
if not output_dirs:
    st.warning("No output folders found under `outputs/`.")
    st.stop()

preferred = Path("outputs/time_series_forecast_notebook")
default_index = output_dirs.index(preferred) if preferred in output_dirs else 0
output_dir = st.sidebar.selectbox(
    "Output folder",
    output_dirs,
    index=default_index,
    format_func=lambda path: str(path),
)

inventory = file_inventory(output_dir)
st.caption(f"Reading artifacts from `{output_dir}`")

if output_dir.name == "time_series_forecast_notebook":
    st.info("You are viewing notebook outputs. Run the Python project to create `outputs/time_series_forecast_project`.")
elif output_dir.name.endswith("_smoke"):
    st.warning("You are viewing smoke-test outputs, not the full project run.")

tab_overview, tab_tables, tab_importance, tab_images = st.tabs(["Overview", "Tables", "Feature Importance", "Images"])

with tab_overview:
    demand_comparison = read_optional_csv(output_dir, "demand_model_performance_table.csv", "demand_all_model_comparison.csv", "model_comparison.csv")
    energy_comparison = read_optional_csv(output_dir, "energy_model_performance_table.csv", "energy_all_model_comparison.csv")
    combined_comparison = read_optional_csv(output_dir, "combined_model_comparison_table.csv")

    left, right = st.columns(2)
    with left:
        show_metrics(demand_comparison, "Demand Prediction", "Target: active station-hour sessions (`session_count`).")
    with right:
        show_metrics(energy_comparison, "Energy Prediction", "Target: station-hour energy consumption (`total_energy_kwh`).")

    if not combined_comparison.empty:
        st.subheader("Demand vs Energy Summary")
        st.dataframe(combined_comparison, use_container_width=True)

    show_user_behavior_clustering(output_dir)

    predictions = read_optional_csv(output_dir, "demand_test_predictions.csv", "test_predictions.csv")
    st.subheader("Demand: Predicted vs Actual Active Sessions")
    show_prediction_charts(predictions)

    hourly_forecast = read_optional_csv(output_dir, "station_hour_forecast.csv")
    daily_forecast = read_optional_csv(output_dir, "station_daily_forecast.csv")
    show_forecast(hourly_forecast, daily_forecast)

with tab_tables:
    csv_files = inventory[inventory["type"].eq("csv")]["file"].tolist() if not inventory.empty else []
    if not csv_files:
        st.info("No CSV files found.")
    else:
        selected_csv = st.selectbox("CSV artifact", csv_files)
        max_rows = st.slider("Rows to load", min_value=100, max_value=250_000, value=5000, step=1000)
        table = read_csv_sample(str(output_dir / selected_csv), max_rows)
        st.write(f"Showing up to {max_rows:,} rows from `{selected_csv}`.")
        st.dataframe(table, use_container_width=True)

with tab_importance:
    demand_importance = read_optional_csv(output_dir, "demand_feature_importance.csv", "feature_importance.csv")
    energy_importance = read_optional_csv(output_dir, "energy_feature_importance.csv")
    left, right = st.columns(2)
    with left:
        show_importance("Demand Feature Importance", demand_importance)
    with right:
        show_importance("Energy Feature Importance", energy_importance)

with tab_images:
    image_files = inventory[inventory["type"].isin(["png", "jpg", "jpeg"])]["file"].tolist() if not inventory.empty else []
    if not image_files:
        st.info("No image files found.")
    else:
        image_files = ordered_images(image_files)
        default_images = image_files[:12]
        selected_images = st.multiselect("Images to show", image_files, default=default_images)
        for image_name in selected_images:
            st.image(str(output_dir / image_name), caption=image_name, use_container_width=True)
