import { useEffect, useMemo, useState } from "react";

import {
  fetchForecast,
  fetchHealth,
  fetchHistory,
  fetchModelInfo,
  fetchStores,
} from "./api/client.js";
import ForecastControls from "./components/ForecastControls.jsx";
import ModelSummary from "./components/ModelSummary.jsx";
import SalesChart, { buildChartData } from "./components/SalesChart.jsx";
import StatusBanner from "./components/StatusBanner.jsx";
import StoreSelector from "./components/StoreSelector.jsx";
import { formatSales, formatLongDate } from "./utils/format.js";

const DEFAULT_HISTORY_DAYS = 90;
const DEFAULT_FORECAST_DAYS = 14;

export default function App() {
  const [stores, setStores] = useState([]);
  const [modelInfo, setModelInfo] = useState(null);
  const [health, setHealth] = useState(null);

  const [selectedStoreId, setSelectedStoreId] = useState(null);
  const [historyDays, setHistoryDays] = useState(DEFAULT_HISTORY_DAYS);
  const [forecastDays, setForecastDays] = useState(DEFAULT_FORECAST_DAYS);
  const [promo, setPromo] = useState(false);
  const [schoolHoliday, setSchoolHoliday] = useState(false);

  const [history, setHistory] = useState(null);
  const [forecast, setForecast] = useState(null);

  const [isLoadingSetup, setIsLoadingSetup] = useState(true);
  const [isLoadingSeries, setIsLoadingSeries] = useState(false);
  const [setupError, setSetupError] = useState(null);
  const [seriesError, setSeriesError] = useState(null);

  // One-time setup: health, store list and model metadata.
  useEffect(() => {
    let cancelled = false;

    async function loadSetup() {
      setIsLoadingSetup(true);
      setSetupError(null);
      try {
        const healthResponse = await fetchHealth();
        if (cancelled) return;
        setHealth(healthResponse);

        const storeList = await fetchStores();
        if (cancelled) return;
        setStores(storeList);
        if (storeList.length > 0) {
          setSelectedStoreId(storeList[0].store_id);
        }

        if (healthResponse.model_loaded) {
          const info = await fetchModelInfo();
          if (cancelled) return;
          setModelInfo(info);
        }
      } catch (error) {
        if (!cancelled) setSetupError(error.message);
      } finally {
        if (!cancelled) setIsLoadingSetup(false);
      }
    }

    loadSetup();
    return () => {
      cancelled = true;
    };
  }, []);

  // Refetch the series whenever the store or any scenario input changes.
  useEffect(() => {
    if (selectedStoreId === null) return;
    let cancelled = false;

    async function loadSeries() {
      setIsLoadingSeries(true);
      setSeriesError(null);
      try {
        const historyResponse = await fetchHistory(selectedStoreId, historyDays);
        if (cancelled) return;
        setHistory(historyResponse);

        if (health && health.model_loaded) {
          const forecastResponse = await fetchForecast(selectedStoreId, {
            days: forecastDays,
            promo,
            schoolHoliday,
          });
          if (cancelled) return;
          setForecast(forecastResponse);
        } else {
          setForecast(null);
        }
      } catch (error) {
        if (!cancelled) {
          setSeriesError(error.message);
          setForecast(null);
        }
      } finally {
        if (!cancelled) setIsLoadingSeries(false);
      }
    }

    loadSeries();
    return () => {
      cancelled = true;
    };
  }, [selectedStoreId, historyDays, forecastDays, promo, schoolHoliday, health]);

  const chartData = useMemo(
    () => buildChartData(history, forecast),
    [history, forecast]
  );

  const selectedStore = stores.find((store) => store.store_id === selectedStoreId);
  const maxForecastDays = modelInfo ? modelInfo.forecast_horizon_days : 14;

  const forecastTotal = forecast
    ? forecast.points.reduce((sum, point) => sum + point.predicted_sales, 0)
    : null;
  const openForecastDays = forecast
    ? forecast.points.filter((point) => !point.is_expected_closed).length
    : 0;

  return (
    <div className="app">
      <header className="app-header">
        <div>
          <h1>
            Forecast<span className="brand-accent">IQ</span>
          </h1>
          <p className="tagline">
            Machine-learned daily sales forecasting for retail stores
          </p>
        </div>
        {health && (
          <span className={`status-pill status-${health.status}`}>
            API {health.status}
            {health.last_observed_date &&
              ` · data through ${health.last_observed_date}`}
          </span>
        )}
      </header>

      <StatusBanner kind="error" message={setupError} />
      {health && health.status !== "ok" && (
        <StatusBanner kind="warning" message={health.detail} />
      )}

      {isLoadingSetup ? (
        <p className="loading">Loading stores and model…</p>
      ) : (
        <div className="layout">
          <aside className="sidebar">
            <section className="card">
              <header className="card-header">
                <h2>Controls</h2>
              </header>
              <StoreSelector
                stores={stores}
                selectedStoreId={selectedStoreId}
                onChange={setSelectedStoreId}
                disabled={isLoadingSeries}
              />
              <ForecastControls
                historyDays={historyDays}
                onHistoryDaysChange={setHistoryDays}
                forecastDays={forecastDays}
                onForecastDaysChange={setForecastDays}
                maxForecastDays={maxForecastDays}
                promo={promo}
                onPromoChange={setPromo}
                schoolHoliday={schoolHoliday}
                onSchoolHolidayChange={setSchoolHoliday}
                disabled={isLoadingSeries}
              />
            </section>

            <ModelSummary modelInfo={modelInfo} />
          </aside>

          <main className="content">
            <section className="card">
              <header className="card-header">
                <h2>
                  {selectedStore
                    ? `Store ${selectedStore.store_id}`
                    : "Sales"}{" "}
                  : actual vs forecast
                </h2>
                {isLoadingSeries && <span className="badge">updating…</span>}
              </header>

              <StatusBanner kind="error" message={seriesError} />

              <SalesChart
                data={chartData}
                forecastStartDate={
                  forecast && forecast.points.length > 0
                    ? forecast.points[0].date
                    : null
                }
              />

              {forecast && (
                <p className="card-note">
                  Forecast generated from{" "}
                  {formatLongDate(forecast.generated_from_date)}, the last day
                  with observed sales. Promotion and school-holiday flags are
                  scenario inputs, not predictions. The dataset contains no
                  future calendar.
                </p>
              )}
            </section>

            {forecast && (
              <div className="stat-grid">
                <div className="stat">
                  <span className="stat-label">Forecast horizon</span>
                  <span className="stat-value">{forecast.horizon_days} days</span>
                </div>
                <div className="stat">
                  <span className="stat-label">Predicted total</span>
                  <span className="stat-value">{formatSales(forecastTotal)}</span>
                </div>
                <div className="stat">
                  <span className="stat-label">Trading days</span>
                  <span className="stat-value">
                    {openForecastDays} of {forecast.points.length}
                  </span>
                </div>
                <div className="stat">
                  <span className="stat-label">Scenario</span>
                  <span className="stat-value">
                    {forecast.assumptions.promo ? "Promo" : "No promo"}
                    {forecast.assumptions.school_holiday && " · holiday"}
                  </span>
                </div>
              </div>
            )}

            {forecast && (
              <section className="card">
                <header className="card-header">
                  <h2>Forecast detail</h2>
                </header>
                <table className="forecast-table">
                  <thead>
                    <tr>
                      <th>Date</th>
                      <th>Predicted sales</th>
                      <th>Note</th>
                    </tr>
                  </thead>
                  <tbody>
                    {forecast.points.map((point) => (
                      <tr
                        key={point.date}
                        className={point.is_expected_closed ? "row-muted" : ""}
                      >
                        <td>{formatLongDate(point.date)}</td>
                        <td className="numeric">
                          {formatSales(point.predicted_sales)}
                        </td>
                        <td className="note-cell">
                          {point.is_expected_closed
                            ? "Store historically closed on this weekday"
                            : ""}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </section>
            )}
          </main>
        </div>
      )}

      <footer className="app-footer">
        <p>
          Built on the public{" "}
          <a
            href="https://www.kaggle.com/competitions/rossmann-store-sales"
            target="_blank"
            rel="noreferrer"
          >
            Rossmann Store Sales
          </a>{" "}
          dataset. Forecasts are model estimates, not guarantees.
        </p>
      </footer>
    </div>
  );
}
