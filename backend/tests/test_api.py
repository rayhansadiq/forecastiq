"""
API tests, including the failure paths.

Loading the dataset takes a few seconds, so the TestClient is created once and
shared. Forecast tests skip rather than fail when no trained model is present,
so a fresh clone can run the suite before running train.py.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.service import service


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def model_required(client):
    if not service.is_model_ready:
        pytest.skip("No trained model; run 'python backend/ml/train.py' first.")


# --- meta -----------------------------------------------------------------


def test_health_always_responds(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"ok", "degraded"}
    assert "data_loaded" in body


def test_root_lists_endpoints(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "/api/stores" in response.json()["endpoints"]


# --- stores ---------------------------------------------------------------


def test_stores_returns_the_full_catalogue(client):
    response = client.get("/api/stores")
    assert response.status_code == 200
    stores = response.json()
    assert len(stores) == 1115
    assert {"store_id", "store_type", "average_daily_sales"} <= set(stores[0])


def test_history_returns_requested_window(client):
    response = client.get("/api/stores/1/history?days=30")
    assert response.status_code == 200
    body = response.json()
    assert body["store_id"] == 1
    assert 0 < len(body["points"]) <= 30
    assert {"date", "sales", "promo", "is_open"} <= set(body["points"][0])


def test_history_is_chronological(client):
    points = client.get("/api/stores/1/history?days=60").json()["points"]
    dates = [point["date"] for point in points]
    assert dates == sorted(dates)


# --- forecasting ----------------------------------------------------------


def test_forecast_returns_one_point_per_day(client, model_required):
    response = client.get("/api/stores/1/forecast?days=14")
    assert response.status_code == 200
    body = response.json()
    assert len(body["points"]) == 14
    assert body["horizon_days"] == 14


def test_forecast_starts_after_last_observed_day(client, model_required):
    body = client.get("/api/stores/1/forecast?days=7").json()
    assert body["points"][0]["date"] > body["generated_from_date"]


def test_forecast_values_are_non_negative(client, model_required):
    body = client.get("/api/stores/1/forecast?days=14").json()
    assert all(point["predicted_sales"] >= 0 for point in body["points"])


def test_forecast_echoes_its_assumptions(client, model_required):
    body = client.get("/api/stores/1/forecast?days=5&promo=true").json()
    assert body["assumptions"]["promo"] is True


def test_promotion_increases_the_forecast(client, model_required):
    """Promo is the strongest feature, so turning it on should raise predictions."""
    without = client.get("/api/stores/1/forecast?days=14").json()
    with_promo = client.get("/api/stores/1/forecast?days=14&promo=true").json()

    total_without = sum(point["predicted_sales"] for point in without["points"])
    total_with = sum(point["predicted_sales"] for point in with_promo["points"])
    assert total_with > total_without


def test_closed_weekdays_forecast_zero(client, model_required):
    body = client.get("/api/stores/1/forecast?days=14").json()
    closed = [point for point in body["points"] if point["is_expected_closed"]]
    assert closed, "store 1 is expected to close at least one weekday"
    assert all(point["predicted_sales"] == 0 for point in closed)


# --- failure paths --------------------------------------------------------


def test_unknown_store_returns_404(client, model_required):
    response = client.get("/api/stores/9999/forecast?days=5")
    assert response.status_code == 404
    assert "does not exist" in response.json()["detail"]


def test_unknown_store_history_returns_404(client):
    assert client.get("/api/stores/9999/history?days=10").status_code == 404


def test_horizon_beyond_the_model_returns_422(client, model_required):
    response = client.get("/api/stores/1/forecast?days=60")
    assert response.status_code == 422
    assert "between 1 and" in response.json()["detail"]


def test_zero_days_is_rejected(client):
    assert client.get("/api/stores/1/forecast?days=0").status_code == 422


def test_negative_store_id_is_rejected(client):
    assert client.get("/api/stores/-5/history?days=10").status_code == 422
