"""
Feature engineering shared by training and inference.

Every feature is built by a function in this module, and both backend/ml/train.py
and the FastAPI inference path call build_feature_frame(). There is deliberately
no second copy of this logic anywhere else in the project -- if training and
serving computed features differently, the served predictions would silently be
wrong in a way that is very hard to detect.

Leakage rule enforced here
--------------------------
Every lag and rolling window is shifted back by FORECAST_HORIZON_DAYS before it
is used. A row dated D only ever sees sales from D - FORECAST_HORIZON_DAYS or
earlier. So a forecast made on day D for any day up to D + FORECAST_HORIZON_DAYS
depends only on sales that had already happened when the forecast was made.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data.clean import MONTH_ABBREVIATIONS  # noqa: E402
from ml.config import (  # noqa: E402
    FEATURE_COLUMNS,
    FORECAST_HORIZON_DAYS,
    LAG_DAYS,
    ROLLING_WINDOWS,
    TARGET_COLUMN,
)


def add_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Split the date into the seasonal components a tree model can split on."""
    result = frame.copy()
    result["Month"] = result["Date"].dt.month
    result["Year"] = result["Date"].dt.year
    result["DayOfMonth"] = result["Date"].dt.day
    result["WeekOfYear"] = result["Date"].dt.isocalendar().week.astype(int)
    return result


def add_competition_features(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Months elapsed since a nearby competitor opened.

    Negative values (competitor opens later) are clipped to 0. Stores with no
    known competitor opening date get 0, and HasCompetitionInfo already tells
    the model which case it is looking at.
    """
    result = frame.copy()
    months_open = (
        12 * (result["Year"] - result["CompetitionOpenSinceYear"])
        + (result["Month"] - result["CompetitionOpenSinceMonth"])
    )
    result["CompetitionOpenMonths"] = months_open.clip(lower=0).fillna(0).astype(float)
    return result


def add_promo2_active_flag(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Whether the store's rolling Promo2 campaign was live on this date.

    A store is in an active Promo2 month when it joined Promo2, the date is on
    or after its Promo2 start date, and the current month appears in its
    PromoInterval string.
    """
    result = frame.copy()
    month_abbreviation = result["Month"].map(MONTH_ABBREVIATIONS)

    joined_promo2 = result["Promo2"].fillna(0) == 1
    started = result["Promo2StartDate"].notna() & (
        result["Date"] >= result["Promo2StartDate"]
    )
    in_promo_month = [
        abbreviation in interval.split(",") if interval else False
        for abbreviation, interval in zip(
            month_abbreviation, result["PromoInterval"].fillna("")
        )
    ]

    result["Promo2Active"] = (
        joined_promo2 & started & pd.Series(in_promo_month, index=result.index)
    ).astype(int)
    return result


def add_lag_features(
    frame: pd.DataFrame, horizon: int = FORECAST_HORIZON_DAYS
) -> pd.DataFrame:
    """
    Sales N days ago, per store.

    Lags are multiples of 7 so each one lands on the same weekday as the row
    being predicted, which matters a lot in retail. The smallest lag is the
    forecast horizon itself.
    """
    result = frame.sort_values(["Store", "Date"]).copy()
    for lag in LAG_DAYS:
        if lag < horizon:
            raise ValueError(
                f"Lag {lag} is shorter than the forecast horizon {horizon}; "
                "that value would not be known at prediction time."
            )
        result[f"SalesLag{lag}"] = result.groupby("Store")[TARGET_COLUMN].shift(lag)
    return result


def add_rolling_features(
    frame: pd.DataFrame, horizon: int = FORECAST_HORIZON_DAYS
) -> pd.DataFrame:
    """
    Rolling mean and standard deviation of sales, per store.

    The series is shifted by the forecast horizon *before* rolling, so no window
    can include a day the forecaster would not have seen yet.
    """
    result = frame.sort_values(["Store", "Date"]).copy()
    shifted_sales = result.groupby("Store")[TARGET_COLUMN].shift(horizon)
    shifted_by_store = shifted_sales.groupby(result["Store"])

    for window in ROLLING_WINDOWS:
        result[f"SalesRollMean{window}"] = (
            shifted_by_store.rolling(window).mean().reset_index(level=0, drop=True)
        )
        result[f"SalesRollStd{window}"] = (
            shifted_by_store.rolling(window).std().reset_index(level=0, drop=True)
        )
    return result


def build_feature_frame(
    panel: pd.DataFrame, horizon: int = FORECAST_HORIZON_DAYS
) -> pd.DataFrame:
    """
    Run the full feature pipeline on a clean daily panel.

    Expects the output of data.clean.build_clean_dataset(), or a frame with the
    same columns. Used identically by training and inference.
    """
    frame = add_calendar_features(panel)
    frame = add_competition_features(frame)
    frame = add_promo2_active_flag(frame)
    frame = add_lag_features(frame, horizon=horizon)
    frame = add_rolling_features(frame, horizon=horizon)
    return frame


def select_modelling_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only rows that are valid to train or evaluate on.

    Dropped, and why:
      - gap-filled rows: never actually observed
      - closed days: sales are structurally 0, not a demand signal
      - open days with 0 sales: a handful of records, almost certainly errors
      - rows with incomplete history: the first ~6 weeks of each store, where
        the lag and rolling windows have not filled up yet
    """
    usable = frame[
        (frame["IsObserved"] == 1)
        & (frame["Open"] == 1)
        & (frame[TARGET_COLUMN] > 0)
    ]
    return usable.dropna(subset=FEATURE_COLUMNS).copy()


def get_feature_matrix(frame: pd.DataFrame):
    """Return X in the exact column order the model was trained on."""
    return frame[FEATURE_COLUMNS]
