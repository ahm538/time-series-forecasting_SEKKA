from __future__ import annotations

import os
from datetime import timedelta
from typing import List

import numpy as np
import pandas as pd

try:
    from prophet import Prophet
except Exception:  # pragma: no cover
    from fbprophet import Prophet  # type: ignore

from . import config


def derive_school_phase(ts: pd.Timestamp) -> str:
    month = ts.month
    if month in (1, 6):
        return "Exam"
    if month in (7, 8, 9):
        return "Holiday"
    if month in (10, 11, 12, 2, 3, 4, 5):
        return "Term"
    return "Term"


def derive_is_summer_peak(ts: pd.Timestamp) -> int:
    return 1 if ts.month in (6, 7, 8, 9) else 0


def derive_is_public_holiday(ts: pd.Timestamp) -> int:
    try:
        import holidays  # type: ignore
        eg_holidays = holidays.country_holidays("EG")
        return 1 if ts.date() in eg_holidays else 0
    except Exception:
        return 0


def ensure_regressors(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["is_public_holiday", "is_summer_peak"]:
        if col in out.columns:
            out[col] = out[col].astype(int)
        else:
            if col == "is_public_holiday":
                out[col] = out["timestamp"].apply(lambda x: derive_is_public_holiday(pd.Timestamp(x))).astype(int)
            if col == "is_summer_peak":
                out[col] = out["timestamp"].apply(lambda x: derive_is_summer_peak(pd.Timestamp(x))).astype(int)

    if "school_term_phase" in out.columns:
        dummies = pd.get_dummies(out["school_term_phase"], prefix="school", dtype=int)
    else:
        phases = out["timestamp"].apply(lambda x: derive_school_phase(pd.Timestamp(x)))
        dummies = pd.get_dummies(phases, prefix="school", dtype=int)

    for needed in ["school_Term", "school_Exam", "school_Holiday"]:
        if needed not in dummies.columns:
            dummies[needed] = 0
    dummies = dummies[["school_Term", "school_Exam", "school_Holiday"]]
    out = pd.concat([out, dummies], axis=1)
    return out


def prepare_prophet_frame(df: pd.DataFrame) -> pd.DataFrame:
    pf = df.copy()
    pf = pf.rename(columns={"timestamp": "ds", "congestion_level": "y"})
    pf["ds"] = pd.to_datetime(pf["ds"], utc=False)
    pf = pf.sort_values("ds").reset_index(drop=True)
    return pf


REG_COLS: List[str] = [
    "is_public_holiday",
    "is_summer_peak",
    "school_Term",
    "school_Exam",
    "school_Holiday",
]


def build_prophet() -> Prophet:
    m = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        seasonality_mode=config.SEASONALITY_MODE,
        changepoint_prior_scale=config.CHANGEPOINT_PRIOR_SCALE,
        seasonality_prior_scale=config.SEASONALITY_PRIOR_SCALE,
    )
    for reg in REG_COLS:
        m.add_regressor(reg)
    try:
        m.add_country_holidays(country_name="EG")
    except Exception:
        pass
    m.add_seasonality(name="daily", period=1, fourier_order=config.DAILY_FOURIER)
    m.add_seasonality(name="weekly", period=7, fourier_order=config.WEEKLY_FOURIER)
    return m


def make_future_with_regressors(last_ds: pd.Timestamp, periods: int, freq: str = "H") -> pd.DataFrame:
    future = pd.date_range(start=last_ds + pd.Timedelta(hours=1), periods=periods, freq=freq)
    fut_df = pd.DataFrame({"ds": future})
    fut_df["is_public_holiday"] = fut_df["ds"].apply(derive_is_public_holiday).astype(int)
    fut_df["is_summer_peak"] = fut_df["ds"].apply(derive_is_summer_peak).astype(int)
    phases = fut_df["ds"].apply(derive_school_phase)
    dummies = pd.get_dummies(phases, prefix="school", dtype=int)
    for needed in ["school_Term", "school_Exam", "school_Holiday"]:
        if needed not in dummies.columns:
            dummies[needed] = 0
    dummies = dummies[["school_Term", "school_Exam", "school_Holiday"]]
    fut_df = pd.concat([fut_df, dummies], axis=1)
    return fut_df
