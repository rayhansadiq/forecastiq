"""
Unit tests for feature engineering.

The important ones here are the leakage tests. A forecasting model that
accidentally sees the future will report excellent metrics and then fail in
production, and the failure is silent -- nothing crashes. These tests assert
the property directly on synthetic data where the correct answer is known.
"""

import numpy as np
import pandas as pd
import pytest

from ml.config import FEATURE_COLUMNS, FORECAST_HORIZON_DAYS, LAG_DAYS
from ml.features import (
    add_calendar_features,
    add_lag_features,
    add_rolling_features,
    build_feature_frame,
)


def make_synthetic_panel(n_days: int = 120, n_stores: int = 2) -> pd.DataFrame:
    """
    Two stores with sales equal to a simple function of the day index, so the
    expected value of any lag is arithmetic we can assert on exactly.
    """
    dates = pd.date_range("2014-01-01", periods=n_days, freq="D")
    rows = []
    for store in range(1, n_stores + 1):
        for day_index, date in enumerate(dates):
            rows.append(
                {
                    "Store": store,
                    "Date": date,
                    "Sales": float(store * 1000 + day_index),
                    "Open": 1,
                    "Promo": day_index % 2,
                    "SchoolHoliday": 0,
                    "StateHolidayCode": 0,
                    "IsObserved": 1,
                    "DayOfWeek": date.dayofweek + 1,
                    "StoreTypeCode": 0,
                    "AssortmentCode": 0,
                    "CompetitionDistance": 500.0,
                    "HasCompetitionInfo": 1,
                    "CompetitionOpenSinceMonth": 1.0,
                    "CompetitionOpenSinceYear": 2010.0,
                    "Promo2": 0,
                    "Promo2StartDate": pd.NaT,
                    "PromoInterval": "",
                }
            )
    return pd.DataFrame(rows)


def test_lag_features_match_sales_n_days_earlier():
    """SalesLag14 on day D must equal Sales on day D-14, per store."""
    panel = make_synthetic_panel()
    featured = add_lag_features(panel)

    for store in panel["Store"].unique():
        store_rows = featured[featured["Store"] == store].sort_values("Date")
        for lag in LAG_DAYS:
            expected = store_rows["Sales"].shift(lag)
            actual = store_rows[f"SalesLag{lag}"]
            pd.testing.assert_series_equal(
                actual, expected, check_names=False
            )


def test_lags_do_not_cross_store_boundaries():
    """Store 2's earliest lag values must be NaN, not leaked from store 1."""
    panel = make_synthetic_panel()
    featured = add_lag_features(panel)

    store_two = featured[featured["Store"] == 2].sort_values("Date")
    earliest = store_two.head(max(LAG_DAYS))
    for lag in LAG_DAYS:
        assert earliest[f"SalesLag{lag}"].head(lag).isna().all()


def test_rolling_windows_never_include_the_target_day():
    """
    The rolling mean on day D must be computable from sales strictly before
    D - FORECAST_HORIZON_DAYS + 1. Verified by recomputing it by hand.
    """
    panel = make_synthetic_panel()
    featured = add_rolling_features(panel)

    store_rows = featured[featured["Store"] == 1].sort_values("Date").reset_index(
        drop=True
    )
    window = 7
    target_index = 80

    expected_source = store_rows["Sales"].iloc[
        target_index - FORECAST_HORIZON_DAYS - window + 1 :
        target_index - FORECAST_HORIZON_DAYS + 1
    ]
    assert np.isclose(
        store_rows[f"SalesRollMean{window}"].iloc[target_index],
        expected_source.mean(),
    )


def test_changing_future_sales_cannot_change_past_features():
    """
    The end-to-end leakage guarantee: rewrite the last FORECAST_HORIZON_DAYS of
    sales to nonsense and every feature on earlier rows must be unchanged.
    """
    panel = make_synthetic_panel()
    baseline = build_feature_frame(panel)

    tampered_panel = panel.copy()
    cutoff = tampered_panel["Date"].max() - pd.Timedelta(
        days=FORECAST_HORIZON_DAYS - 1
    )
    tampered_panel.loc[tampered_panel["Date"] >= cutoff, "Sales"] = 999_999.0
    tampered = build_feature_frame(tampered_panel)

    unchanged_rows = baseline["Date"] < cutoff
    pd.testing.assert_frame_equal(
        baseline.loc[unchanged_rows, FEATURE_COLUMNS].reset_index(drop=True),
        tampered.loc[unchanged_rows, FEATURE_COLUMNS].reset_index(drop=True),
    )


def test_lag_shorter_than_horizon_is_rejected():
    """A lag inside the horizon would not be known at prediction time."""
    panel = make_synthetic_panel()
    with pytest.raises(ValueError, match="shorter than the forecast horizon"):
        add_lag_features(panel, horizon=max(LAG_DAYS) + 1)


def test_calendar_features_are_derived_from_the_date():
    panel = make_synthetic_panel(n_days=40, n_stores=1)
    featured = add_calendar_features(panel)

    first = featured.iloc[0]
    assert first["Year"] == 2014
    assert first["Month"] == 1
    assert first["DayOfMonth"] == 1
