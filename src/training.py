from __future__ import annotations

import os
import json
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from joblib import dump

from . import config
from .preprocessing import (
    ensure_regressors,
    prepare_prophet_frame,
    build_prophet,
    REG_COLS,
)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def train_and_evaluate_for_route(route_df: pd.DataFrame, route_id: str) -> Tuple[object, Dict[str, float]]:
    """Train Prophet for a single route and compute last-month metrics.

    Returns
    -------
    model : Prophet
    metrics : dict with MAE and RMSE
    """
    # Feature engineering
    route_df = ensure_regressors(route_df)
    pf = prepare_prophet_frame(route_df)

    # Train/test split by time
    max_ts = pf["ds"].max()
    cutoff = max_ts - pd.Timedelta(days=config.TEST_DAYS)
    train_df = pf[pf["ds"] < cutoff].copy()
    test_df = pf[pf["ds"] >= cutoff].copy()

    # Build and fit
    m = build_prophet()
    m.fit(train_df[["ds", "y"] + REG_COLS])

    # Evaluate
    forecast_test = m.predict(test_df[["ds"] + REG_COLS])
    y_true = test_df["y"].to_numpy()
    y_pred = forecast_test["yhat"].to_numpy()
    y_pred = np.clip(y_pred, config.Y_MIN, config.Y_MAX)

    metrics = {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "train_rows": float(len(train_df)),
        "test_rows": float(len(test_df)),
    }

    # Persist model
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    model_path = config.MODELS_DIR / f"model_route_{route_id}.joblib"
    dump(m, model_path)

    # Persist simple metadata for inference
    meta = {
        "route_id": str(route_id),
        "last_ds": pf["ds"].max().isoformat(),
        "regressors": REG_COLS,
    }
    meta_path = config.MODELS_DIR / f"metadata_route_{route_id}.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f)

    return m, metrics
