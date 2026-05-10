"""Model training and evaluation utilities."""

from __future__ import annotations

import json
import copy
from pathlib import Path
from time import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import CATEGORICAL_FEATURES, DEFAULT_RANDOM_STATE, TARGET

try:
    from catboost import CatBoostRegressor
except Exception:  # pragma: no cover - optional dependency
    CatBoostRegressor = None


def temporal_train_test_split(df: pd.DataFrame, test_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    unique_hours = pd.Index(sorted(df["hour_ts"].unique()))
    split_pos = max(1, int(len(unique_hours) * (1 - test_size)))
    split_pos = min(split_pos, len(unique_hours) - 1)
    cutoff = pd.Timestamp(unique_hours[split_pos])
    return df[df["hour_ts"] < cutoff].copy(), df[df["hour_ts"] >= cutoff].copy(), cutoff


def evaluate_predictions(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def build_preprocessor(numeric_features: list[str], categorical_features: list[str] = CATEGORICAL_FEATURES) -> ColumnTransformer:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=3, sparse_output=False)),
        ]
    )
    return ColumnTransformer([("num", numeric, numeric_features), ("cat", categorical, categorical_features)])


def model_candidates(numeric_features: list[str], random_state: int = DEFAULT_RANDOM_STATE) -> dict[str, Any]:
    def pipeline(model: Any) -> Pipeline:
        return Pipeline([("preprocess", build_preprocessor(numeric_features)), ("model", model)])

    candidates: dict[str, Any] = {
        "HistGradientBoosting_poisson": pipeline(
            HistGradientBoostingRegressor(
                loss="poisson",
                max_iter=500,
                learning_rate=0.04,
                max_depth=8,
                max_leaf_nodes=31,
                min_samples_leaf=30,
                l2_regularization=0.1,
                random_state=random_state,
            )
        ),
        "HistGradientBoosting_squared": pipeline(
            HistGradientBoostingRegressor(
                loss="squared_error",
                max_iter=500,
                learning_rate=0.04,
                max_depth=8,
                max_leaf_nodes=31,
                min_samples_leaf=30,
                l2_regularization=0.1,
                random_state=random_state,
            )
        ),
    }

    if CatBoostRegressor is not None:
        candidates["CatBoost_poisson"] = pipeline(
            CatBoostRegressor(
                iterations=350,
                depth=6,
                learning_rate=0.045,
                loss_function="Poisson",
                random_seed=random_state,
                allow_writing_files=False,
                verbose=False,
            )
        )
    else:
        candidates["Ridge_baseline"] = pipeline(Ridge(alpha=15.0))
    return candidates


def compare_models(
    train: pd.DataFrame,
    test: pd.DataFrame,
    numeric_features: list[str],
    target: str = TARGET,
    random_state: int = DEFAULT_RANDOM_STATE,
    cv_splits: int = 3,
) -> tuple[pd.DataFrame, Any, np.ndarray]:
    x_train = train[numeric_features + CATEGORICAL_FEATURES]
    y_train = train[target]
    x_test = test[numeric_features + CATEGORICAL_FEATURES]
    y_test = test[target]

    rows = []
    fitted = {}
    predictions = {}

    baseline_source = "lag_168h_sessions" if "lag_168h_sessions" in test.columns else "lag_24h_sessions"
    baseline_pred = np.maximum(0, test[baseline_source].fillna(train[target].mean()).to_numpy())
    rows.append({"Model": f"Baseline_{baseline_source}", "Train_seconds": 0.0, **evaluate_predictions(y_test, baseline_pred)})

    n_splits = min(cv_splits, max(2, train["hour_ts"].nunique() // 48))
    cv = TimeSeriesSplit(n_splits=n_splits)
    for name, model in model_candidates(numeric_features, random_state=random_state).items():
        started = time()
        cv_scores = []
        error = None
        for train_idx, valid_idx in cv.split(x_train):
            fold_model = copy.deepcopy(model)
            try:
                fold_model.fit(x_train.iloc[train_idx], y_train.iloc[train_idx])
                fold_pred = np.maximum(0, fold_model.predict(x_train.iloc[valid_idx]))
                cv_scores.append(evaluate_predictions(y_train.iloc[valid_idx], fold_pred)["R2"])
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                break
        try:
            model.fit(x_train, y_train)
            pred = np.maximum(0, model.predict(x_test))
            metrics = evaluate_predictions(y_test, pred)
            fitted[name] = model
            predictions[name] = pred
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            pred = np.full(len(y_test), float(y_train.mean()))
            metrics = evaluate_predictions(y_test, pred)
        rows.append(
            {
                "Model": name,
                "CV_R2_mean": float(np.mean(cv_scores)) if cv_scores else np.nan,
                "Train_seconds": float(time() - started),
                "Error": error,
                **metrics,
            }
        )

    comparison = pd.DataFrame(rows).sort_values(["RMSE", "MAE"]).reset_index(drop=True)
    best_name = next(name for name in comparison["Model"] if name in fitted)
    return comparison, fitted[best_name], predictions[best_name]


def save_predictions(test: pd.DataFrame, pred: np.ndarray, output_dir: Path, target: str = TARGET) -> pd.DataFrame:
    predictions = test[["hour_ts", "station_name", "location_name", target]].copy()
    predictions["predicted_session_count"] = pred
    predictions["abs_error"] = (predictions["predicted_session_count"] - predictions[target]).abs()
    predictions["error"] = predictions["predicted_session_count"] - predictions[target]
    predictions.to_csv(output_dir / "demand_test_predictions.csv", index=False)
    return predictions


def save_feature_importance(
    model: Any,
    test: pd.DataFrame,
    numeric_features: list[str],
    output_dir: Path,
    target: str = TARGET,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> pd.DataFrame:
    sample = test.sample(min(4000, len(test)), random_state=random_state) if len(test) else test
    result = permutation_importance(
        model,
        sample[numeric_features + CATEGORICAL_FEATURES],
        sample[target],
        scoring="neg_root_mean_squared_error",
        n_repeats=3,
        random_state=random_state,
        n_jobs=1,
    )
    importance = (
        pd.DataFrame({"feature": numeric_features + CATEGORICAL_FEATURES, "importance_mean": result.importances_mean})
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
    importance.to_csv(output_dir / "demand_feature_importance.csv", index=False)
    return importance


def save_model_artifact(model: Any, numeric_features: list[str], output_dir: Path, metadata: dict[str, Any]) -> None:
    artifact = {
        "model": model,
        "numeric_features": numeric_features,
        "categorical_features": CATEGORICAL_FEATURES,
        "target": TARGET,
        "metadata": metadata,
    }
    joblib.dump(artifact, output_dir / "demand_model_artifact.joblib")
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
