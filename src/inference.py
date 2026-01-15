from __future__ import annotations

import json
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from joblib import load

from . import config
from .preprocessing import make_future_with_regressors


def load_model_and_meta(route_id: str):
    model_path = config.MODELS_DIR / f"model_route_{route_id}.joblib"
    meta_path = config.MODELS_DIR / f"metadata_route_{route_id}.json"
    if not model_path.exists() or not meta_path.exists():
        return None, None
    model = load(model_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return model, meta


def interpret_congestion(level: float) -> str:
    """Map 0-10 congestion score to human-readable status.
    0-3: 'Green - Clear (سالكة)'
    3-6: 'Yellow - Moderate (ماشية)'
    6-8: 'Orange - Heavy (زحمة)'
    8-10: 'Red - Severe (واقفة)'
    """
    try:
        v = float(level)
    except Exception:
        return "Unknown"
    if v < 3:
        return "Green - Clear (سالكة)"
    if v < 6:
        return "Yellow - Moderate (ماشية)"
    if v < 8:
        return "Orange - Heavy (زحمة)"
    return "Red - Severe (واقفة)"


def predict_future(route_id: str, future_hours: int) -> pd.DataFrame:
    model, meta = load_model_and_meta(route_id)
    if model is None or meta is None:
        raise FileNotFoundError("Model or metadata not found for route_id")

    last_ds = pd.Timestamp(meta["last_ds"])
    future_df = make_future_with_regressors(last_ds=last_ds, periods=future_hours, freq="H")
    forecast = model.predict(future_df)
    out = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    # Calibration factor to compensate under-prediction of peaks
    CAL = 1.25
    out["yhat"] = np.clip(out["yhat"] * CAL, config.Y_MIN, config.Y_MAX)
    out["yhat_lower"] = np.clip(out["yhat_lower"] * CAL, config.Y_MIN, config.Y_MAX)
    out["yhat_upper"] = np.clip(out["yhat_upper"] * CAL, config.Y_MIN, config.Y_MAX)
    return out
