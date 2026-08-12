"""
ForecastIQ API.

Route handlers only: every one of them validates input, calls the service
layer, and shapes a response. All data and model logic lives in service.py.

Run from the project root:
    uvicorn backend.app.main:app --reload

Interactive docs: http://127.0.0.1:8000/docs
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Path as PathParam, Query
from fastapi.middleware.cors import CORSMiddleware

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.schemas import (  # noqa: E402
    ErrorResponse,
    ForecastResponse,
    HealthResponse,
    HistoryResponse,
    ModelInfoResponse,
    StoreSummary,
)
from app.service import ServiceError, service  # noqa: E402

# Vite's dev server. Kept explicit rather than "*" so the allowed origin is
# an intentional decision rather than an accident.
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the dataset and model once at startup.

    Failures are captured on the service rather than raised, so a missing or
    corrupt model file degrades the API instead of preventing it from starting.
    /health reports exactly what went wrong.
    """
    print("Loading dataset and model ...")
    service.load()
    health = service.get_health()
    print(f"Startup status: {health['status']}")
    if health["detail"]:
        print(f"  {health['detail']}")
    yield


app = FastAPI(
    title="ForecastIQ API",
    description=(
        "Sales demand forecasting for retail stores, served from a gradient "
        "boosted regression model trained on the public Rossmann Store Sales "
        "dataset."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)


def _handle(callable_, *args, **kwargs):
    """Translate ServiceError into the matching HTTP response."""
    try:
        return callable_(*args, **kwargs)
    except ServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message)


@app.get("/api/health", response_model=HealthResponse, tags=["meta"])
def get_health():
    """Report whether the data and model loaded. Never fails."""
    return service.get_health()


@app.get(
    "/api/model",
    response_model=ModelInfoResponse,
    responses={503: {"model": ErrorResponse}},
    tags=["meta"],
)
def get_model_info():
    """Training window, feature list and held-out test metrics."""
    return _handle(service.get_model_info)


@app.get(
    "/api/stores",
    response_model=list[StoreSummary],
    responses={503: {"model": ErrorResponse}},
    tags=["stores"],
)
def list_stores():
    """Every store with data, plus its date range and average daily sales."""
    return _handle(service.list_stores)


@app.get(
    "/api/stores/{store_id}/history",
    response_model=HistoryResponse,
    responses={404: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    tags=["stores"],
)
def get_store_history(
    store_id: int = PathParam(ge=1, description="Store id"),
    days: int = Query(
        default=90, ge=1, le=1000, description="How many recent days to return"
    ),
):
    """Observed daily sales for one store, most recent `days` calendar days."""
    return _handle(service.get_history, store_id, days)


@app.get(
    "/api/stores/{store_id}/forecast",
    response_model=ForecastResponse,
    responses={
        404: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
    tags=["forecast"],
)
def get_store_forecast(
    store_id: int = PathParam(ge=1, description="Store id"),
    days: int = Query(default=14, ge=1, description="Days ahead to forecast"),
    promo: bool = Query(
        default=False, description="Assume a promotion runs on the forecast days"
    ),
    school_holiday: bool = Query(
        default=False, description="Assume the forecast days are school holidays"
    ),
):
    """
    Predict daily sales for the days immediately after the last observed date.

    Capped at the model's forecast horizon. Promotion and school-holiday
    schedules are inputs, not predictions -- the dataset contains no future
    calendar, so the caller states the scenario and it is echoed back in the
    response.
    """
    return _handle(service.forecast, store_id, days, promo, school_holiday)


@app.get("/", tags=["meta"])
def root():
    return {
        "name": "ForecastIQ API",
        "docs": "/docs",
        "endpoints": [
            "/api/health",
            "/api/model",
            "/api/stores",
            "/api/stores/{store_id}/history?days=90",
            "/api/stores/{store_id}/forecast?days=14&promo=false",
        ],
    }
