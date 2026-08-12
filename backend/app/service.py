"""
Data and model access for the API.

All pandas and scikit-learn work lives here so the route handlers in main.py
stay thin. Feature engineering is imported from backend/ml/features.py -- the
exact same functions training used, never a reimplementation.
"""

import sys
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data.clean import build_clean_dataset  # noqa: E402
from ml.config import (  # noqa: E402
    FEATURE_COLUMNS,
    FORECAST_HORIZON_DAYS,
    MIN_HISTORY_DAYS,
    MODEL_PATH,
)
from ml.features import build_feature_frame  # noqa: E402

STORE_TYPE_LABELS = {0: "a", 1: "b", 2: "c", 3: "d"}
ASSORTMENT_LABELS = {0: "a", 1: "b", 2: "c"}

# A store is treated as closed on a weekday when it was open on fewer than this
# fraction of those weekdays historically. Most Rossmann stores close Sundays.
OPEN_RATE_CLOSED_THRESHOLD = 0.1

PANEL_COLUMNS = [
    "Store", "Date", "Sales", "Open", "Promo", "SchoolHoliday",
    "StateHolidayCode", "IsObserved", "DayOfWeek", "StoreTypeCode",
    "AssortmentCode", "CompetitionDistance", "HasCompetitionInfo",
    "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear",
    "Promo2", "Promo2StartDate", "PromoInterval",
]


class ServiceError(Exception):
    """Raised for expected failure cases the API turns into clean HTTP errors."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class ForecastService:
    """
    Holds the loaded panel and model in memory for the life of the process.

    Data and model are loaded independently. If the model file is missing or
    unreadable the service still starts and still serves history -- only the
    forecast endpoint degrades, and it does so with a clear message rather than
    a crash.
    """

    def __init__(self):
        self.panel: Optional[pd.DataFrame] = None
        self.artifact: Optional[dict] = None
        self.data_error: Optional[str] = None
        self.model_error: Optional[str] = None
        self._weekday_open_rate: Optional[pd.DataFrame] = None
        self._store_summary: Optional[pd.DataFrame] = None

    # --- startup ----------------------------------------------------------

    def load(self) -> None:
        self._load_data()
        self._load_model()

    def _load_data(self) -> None:
        try:
            panel = build_clean_dataset()[PANEL_COLUMNS]
            self.panel = panel.sort_values(["Store", "Date"]).reset_index(drop=True)
            self._precompute_store_lookups()
            self.data_error = None
        except Exception as exc:
            self.panel = None
            self.data_error = f"Could not load data: {exc}"

    def _load_model(self) -> None:
        if not MODEL_PATH.exists():
            self.artifact = None
            self.model_error = (
                f"Model file not found at {MODEL_PATH}. "
                "Run 'python backend/ml/train.py' to create it."
            )
            return
        try:
            artifact = joblib.load(MODEL_PATH)
            if "model" not in artifact or "feature_columns" not in artifact:
                raise ValueError("artifact is missing 'model' or 'feature_columns'")
            self.artifact = artifact
            self.model_error = None
        except Exception as exc:
            self.artifact = None
            self.model_error = (
                f"Model file at {MODEL_PATH} could not be loaded: {exc}. "
                "Retrain with 'python backend/ml/train.py'."
            )

    def _precompute_store_lookups(self) -> None:
        observed = self.panel[self.panel["IsObserved"] == 1]

        self._weekday_open_rate = (
            observed.groupby(["Store", "DayOfWeek"])["Open"].mean().rename("open_rate")
        )

        open_days = observed[observed["Open"] == 1]
        self._store_summary = (
            open_days.groupby("Store")
            .agg(
                first_date=("Date", "min"),
                last_date=("Date", "max"),
                observed_days=("Date", "count"),
                average_daily_sales=("Sales", "mean"),
                store_type_code=("StoreTypeCode", "first"),
                assortment_code=("AssortmentCode", "first"),
            )
            .reset_index()
        )

    # --- state ------------------------------------------------------------

    @property
    def is_data_ready(self) -> bool:
        return self.panel is not None

    @property
    def is_model_ready(self) -> bool:
        return self.artifact is not None

    def require_data(self) -> None:
        if not self.is_data_ready:
            raise ServiceError(self.data_error or "Data not loaded", status_code=503)

    def require_model(self) -> None:
        self.require_data()
        if not self.is_model_ready:
            raise ServiceError(self.model_error or "Model not loaded", status_code=503)

    def require_store(self, store_id: int) -> None:
        self.require_data()
        if store_id not in set(self._store_summary["Store"]):
            raise ServiceError(
                f"Store {store_id} does not exist. "
                f"Valid store ids run from {int(self._store_summary['Store'].min())} "
                f"to {int(self._store_summary['Store'].max())}.",
                status_code=404,
            )

    # --- reads ------------------------------------------------------------

    def list_stores(self) -> list:
        self.require_data()
        return [
            {
                "store_id": int(row.Store),
                "store_type": STORE_TYPE_LABELS.get(int(row.store_type_code), "?"),
                "assortment": ASSORTMENT_LABELS.get(int(row.assortment_code), "?"),
                "first_date": row.first_date.date(),
                "last_date": row.last_date.date(),
                "observed_days": int(row.observed_days),
                "average_daily_sales": round(float(row.average_daily_sales), 2),
            }
            for row in self._store_summary.itertuples()
        ]

    def get_history(self, store_id: int, days: int) -> dict:
        """Most recent `days` calendar days of observed sales for one store."""
        self.require_store(store_id)

        store_rows = self.panel[
            (self.panel["Store"] == store_id) & (self.panel["IsObserved"] == 1)
        ]
        if store_rows.empty:
            raise ServiceError(
                f"Store {store_id} has no observed sales records.", status_code=404
            )

        last_date = store_rows["Date"].max()
        window_start = last_date - pd.Timedelta(days=days - 1)
        window = store_rows[store_rows["Date"] >= window_start]

        if window.empty:
            raise ServiceError(
                f"Store {store_id} has no data in the requested window "
                f"({window_start.date()} to {last_date.date()}).",
                status_code=404,
            )

        return {
            "store_id": store_id,
            "start_date": window["Date"].min().date(),
            "end_date": window["Date"].max().date(),
            "points": [
                {
                    "date": row.Date.date(),
                    "sales": float(row.Sales),
                    "promo": bool(row.Promo),
                    "is_open": bool(row.Open),
                }
                for row in window.itertuples()
            ],
        }

    # --- forecasting ------------------------------------------------------

    def _build_future_rows(
        self,
        store_history: pd.DataFrame,
        last_date: pd.Timestamp,
        days: int,
        assume_promo: bool,
        assume_school_holiday: bool,
    ) -> pd.DataFrame:
        """
        Create placeholder rows for the dates being forecast.

        Sales is left as NaN: these are the rows we are predicting. Store
        attributes carry forward from the most recent observed row, since they
        describe the store rather than the day.
        """
        future_dates = pd.date_range(
            last_date + pd.Timedelta(days=1), periods=days, freq="D"
        )
        latest_row = store_history.iloc[-1]

        future = pd.DataFrame({"Date": future_dates})
        future["Store"] = int(latest_row["Store"])
        future["Sales"] = np.nan
        future["Open"] = 1
        future["Promo"] = int(assume_promo)
        future["SchoolHoliday"] = int(assume_school_holiday)
        future["StateHolidayCode"] = 0
        future["IsObserved"] = 0
        future["DayOfWeek"] = future["Date"].dt.dayofweek + 1

        for column in [
            "StoreTypeCode", "AssortmentCode", "CompetitionDistance",
            "HasCompetitionInfo", "CompetitionOpenSinceMonth",
            "CompetitionOpenSinceYear", "Promo2", "Promo2StartDate",
            "PromoInterval",
        ]:
            future[column] = latest_row[column]

        return future[PANEL_COLUMNS]

    def forecast(
        self,
        store_id: int,
        days: int,
        assume_promo: bool = False,
        assume_school_holiday: bool = False,
    ) -> dict:
        self.require_model()
        self.require_store(store_id)

        horizon = int(self.artifact.get("forecast_horizon_days", FORECAST_HORIZON_DAYS))
        if days < 1 or days > horizon:
            raise ServiceError(
                f"days must be between 1 and {horizon}. The model's features are "
                f"shifted back {horizon} days so that no prediction depends on "
                "sales that had not happened yet; forecasting further out would "
                "require feeding predictions back in as inputs, which this "
                "project does not do.",
                status_code=422,
            )

        store_history = self.panel[
            (self.panel["Store"] == store_id) & (self.panel["IsObserved"] == 1)
        ].copy()

        if store_history.empty:
            raise ServiceError(
                f"Store {store_id} has no observed sales records.", status_code=404
            )

        if len(store_history) < MIN_HISTORY_DAYS:
            raise ServiceError(
                f"Store {store_id} has only {len(store_history)} days of history. "
                f"At least {MIN_HISTORY_DAYS} are needed to fill the lag and "
                "rolling-average features.",
                status_code=422,
            )

        last_date = store_history["Date"].max()
        future_rows = self._build_future_rows(
            store_history, last_date, days, assume_promo, assume_school_holiday
        )

        combined = pd.concat([store_history, future_rows], ignore_index=True)
        featured = build_feature_frame(combined)
        future_featured = featured[featured["Date"] > last_date].sort_values("Date")

        feature_matrix = future_featured[FEATURE_COLUMNS]
        if feature_matrix.isna().any().any():
            incomplete = feature_matrix.columns[
                feature_matrix.isna().any()
            ].tolist()
            raise ServiceError(
                f"Store {store_id} does not have enough continuous recent history "
                f"to compute these features: {', '.join(incomplete)}.",
                status_code=422,
            )

        predictions = self.artifact["model"].predict(feature_matrix)

        points = []
        for row, predicted in zip(future_featured.itertuples(), predictions):
            open_rate = self._lookup_open_rate(store_id, int(row.DayOfWeek))
            expected_closed = open_rate < OPEN_RATE_CLOSED_THRESHOLD
            points.append(
                {
                    "date": row.Date.date(),
                    # A store that is essentially always shut on this weekday
                    # will sell nothing, regardless of what the model says.
                    "predicted_sales": 0.0 if expected_closed
                    else round(float(max(predicted, 0.0)), 2),
                    "day_of_week": int(row.DayOfWeek),
                    "is_expected_closed": bool(expected_closed),
                }
            )

        return {
            "store_id": store_id,
            "generated_from_date": last_date.date(),
            "horizon_days": days,
            "assumptions": {
                "promo": assume_promo,
                "school_holiday": assume_school_holiday,
                "note": (
                    "Future promotion and school-holiday schedules are not part of "
                    "the dataset, so they are supplied as inputs rather than "
                    "predicted. Change them with the promo and school_holiday "
                    "query parameters."
                ),
            },
            "points": points,
        }

    def _lookup_open_rate(self, store_id: int, day_of_week: int) -> float:
        try:
            return float(self._weekday_open_rate.loc[(store_id, day_of_week)])
        except KeyError:
            return 1.0

    # --- model metadata ---------------------------------------------------

    def get_model_info(self) -> dict:
        self.require_model()
        artifact = self.artifact
        return {
            "model_type": artifact["model_type"],
            "trained_at": artifact["trained_at"],
            "train_start_date": artifact["train_start_date"],
            "train_end_date": artifact["train_end_date"],
            "test_start_date": artifact["test_start_date"],
            "test_end_date": artifact["test_end_date"],
            "n_train_rows": artifact["n_train_rows"],
            "n_test_rows": artifact["n_test_rows"],
            "forecast_horizon_days": artifact["forecast_horizon_days"],
            "feature_columns": artifact["feature_columns"],
            "metrics": artifact["metrics"],
            "baseline_metrics": artifact["baseline_metrics"],
        }

    def get_health(self) -> dict:
        problems = [
            message
            for message in (self.data_error, self.model_error)
            if message is not None
        ]
        return {
            "status": "ok" if not problems else "degraded",
            "model_loaded": self.is_model_ready,
            "data_loaded": self.is_data_ready,
            "model_type": self.artifact["model_type"] if self.is_model_ready else None,
            "trained_at": self.artifact["trained_at"] if self.is_model_ready else None,
            "last_observed_date": (
                self.panel["Date"].max().date() if self.is_data_ready else None
            ),
            "forecast_horizon_days": (
                self.artifact["forecast_horizon_days"] if self.is_model_ready else None
            ),
            "detail": " | ".join(problems) if problems else None,
        }


service = ForecastService()
