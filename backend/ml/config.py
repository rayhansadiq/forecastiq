"""
Shared configuration for the ForecastIQ data pipeline, model training and
inference. Kept in one place so training and serving can never drift apart.
"""

from pathlib import Path

# --- Paths -----------------------------------------------------------------

BACKEND_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = BACKEND_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BACKEND_DIR / "data" / "processed"
MODEL_DIR = BACKEND_DIR / "ml"

MODEL_PATH = MODEL_DIR / "model.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"

TRAIN_CSV = RAW_DATA_DIR / "train.csv"
STORE_CSV = RAW_DATA_DIR / "store.csv"


# --- Forecasting setup -----------------------------------------------------

# How many days ahead the model is designed to predict.
#
# This value drives the minimum lag used in feature engineering. Every lag and
# rolling window is shifted back by at least FORECAST_HORIZON_DAYS, which means
# a prediction for any day within the horizon only ever depends on sales that
# were already observed before the forecast was made. This is what makes the
# forecast leakage-free without needing recursive prediction.
FORECAST_HORIZON_DAYS = 14

# Lags are multiples of 7 so each one lands on the same weekday as the target.
LAG_DAYS = [14, 21, 28]

# Rolling windows are computed on the series already shifted by the horizon.
ROLLING_WINDOWS = [7, 14, 28]

# Days of history a store needs before any feature row is complete:
# shift(14) + rolling(28) reaches 14 + 28 - 1 = 41 days back.
MIN_HISTORY_DAYS = 42

# Length of the held-out evaluation period at the end of the timeline.
TEST_PERIOD_DAYS = 42


# --- Feature columns -------------------------------------------------------

CALENDAR_FEATURES = [
    "DayOfWeek",
    "Month",
    "Year",
    "DayOfMonth",
    "WeekOfYear",
]

FLAG_FEATURES = [
    "Promo",
    "SchoolHoliday",
    "StateHolidayCode",
    "Promo2Active",
]

STORE_FEATURES = [
    "StoreTypeCode",
    "AssortmentCode",
    "CompetitionDistance",
    "CompetitionOpenMonths",
    "HasCompetitionInfo",
]

LAG_FEATURES = [f"SalesLag{lag}" for lag in LAG_DAYS]

ROLLING_FEATURES = (
    [f"SalesRollMean{window}" for window in ROLLING_WINDOWS]
    + [f"SalesRollStd{window}" for window in ROLLING_WINDOWS]
)

FEATURE_COLUMNS = (
    CALENDAR_FEATURES
    + FLAG_FEATURES
    + STORE_FEATURES
    + LAG_FEATURES
    + ROLLING_FEATURES
)

TARGET_COLUMN = "Sales"

# Deliberately excluded from FEATURE_COLUMNS:
#
#   Customers  - recorded on the same day as Sales. It is not known at forecast
#                time, so using it would leak information the model would never
#                have in production.
#   Open       - rows where the store is closed are removed before training, so
#                every training row has Open == 1 and the column carries no
#                signal.
EXCLUDED_WITH_REASON = {
    "Customers": "same-day measurement, unavailable at forecast time",
    "Open": "constant after filtering to open days",
}
