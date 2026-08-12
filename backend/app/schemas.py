"""
Pydantic request/response models for the ForecastIQ API.

These define the contract the frontend codes against, and give FastAPI enough
information to generate the interactive docs at /docs.
"""

from datetime import date
from typing import List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(description="'ok' when the API can serve predictions")
    model_loaded: bool
    data_loaded: bool
    model_type: Optional[str] = None
    trained_at: Optional[str] = None
    last_observed_date: Optional[date] = None
    forecast_horizon_days: Optional[int] = None
    detail: Optional[str] = Field(
        default=None, description="Populated when status is not 'ok'"
    )


class ModelMetrics(BaseModel):
    """Held-out test-set performance, read from the saved model artifact."""

    mae: float
    rmse: float
    rmspe: float
    mae_as_pct_of_mean: float
    mean_actual_sales: float


class ModelInfoResponse(BaseModel):
    model_type: str
    trained_at: str
    train_start_date: date
    train_end_date: date
    test_start_date: date
    test_end_date: date
    n_train_rows: int
    n_test_rows: int
    forecast_horizon_days: int
    feature_columns: List[str]
    metrics: ModelMetrics
    baseline_metrics: ModelMetrics


class StoreSummary(BaseModel):
    store_id: int
    store_type: str
    assortment: str
    first_date: date
    last_date: date
    observed_days: int
    average_daily_sales: float


class HistoryPoint(BaseModel):
    date: date
    sales: float
    promo: bool
    is_open: bool


class HistoryResponse(BaseModel):
    store_id: int
    start_date: date
    end_date: date
    points: List[HistoryPoint]


class ForecastPoint(BaseModel):
    date: date
    predicted_sales: float
    day_of_week: int
    is_expected_closed: bool = Field(
        description="True when this store is historically closed on this weekday"
    )


class ForecastAssumptions(BaseModel):
    """
    Future promo and holiday schedules are not in the dataset, so the caller
    supplies them. Returned alongside the forecast so results are never
    interpreted without the assumptions that produced them.
    """

    promo: bool
    school_holiday: bool
    note: str


class ForecastResponse(BaseModel):
    store_id: int
    generated_from_date: date = Field(
        description="Last date with observed sales; the forecast starts the day after"
    )
    horizon_days: int
    assumptions: ForecastAssumptions
    points: List[ForecastPoint]


class ErrorResponse(BaseModel):
    detail: str
