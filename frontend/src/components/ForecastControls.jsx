const HISTORY_WINDOW_OPTIONS = [30, 60, 90, 180, 365];

// Controls for the forecast scenario.
//
// Promotion and school-holiday flags are inputs rather than predictions: the
// dataset has no future calendar, so the user states the scenario and the API
// echoes it back. maxForecastDays comes from the model artifact, not a
// hardcoded constant, so the UI can never offer a horizon the model cannot
// actually serve.
export default function ForecastControls({
  historyDays,
  onHistoryDaysChange,
  forecastDays,
  onForecastDaysChange,
  maxForecastDays,
  promo,
  onPromoChange,
  schoolHoliday,
  onSchoolHolidayChange,
  disabled,
}) {
  return (
    <>
      <div className="control">
        <label className="control-label" htmlFor="history-select">
          History window
        </label>
        <select
          id="history-select"
          className="control-input"
          value={historyDays}
          disabled={disabled}
          onChange={(event) => onHistoryDaysChange(Number(event.target.value))}
        >
          {HISTORY_WINDOW_OPTIONS.map((days) => (
            <option key={days} value={days}>
              Last {days} days
            </option>
          ))}
        </select>
      </div>

      <div className="control">
        <label className="control-label" htmlFor="forecast-range">
          Forecast horizon
          <span className="control-value">{forecastDays} days</span>
        </label>
        <input
          id="forecast-range"
          className="control-range"
          type="range"
          min="1"
          max={maxForecastDays}
          value={forecastDays}
          disabled={disabled}
          onChange={(event) => onForecastDaysChange(Number(event.target.value))}
        />
        <p className="control-hint">
          Capped at {maxForecastDays} days. The model's features are shifted
          back {maxForecastDays} days so no prediction depends on sales that had
          not happened yet.
        </p>
      </div>

      <div className="control">
        <span className="control-label">Scenario assumptions</span>
        <label className="checkbox-row">
          <input
            type="checkbox"
            checked={promo}
            disabled={disabled}
            onChange={(event) => onPromoChange(event.target.checked)}
          />
          <span>Promotion running</span>
        </label>
        <label className="checkbox-row">
          <input
            type="checkbox"
            checked={schoolHoliday}
            disabled={disabled}
            onChange={(event) => onSchoolHolidayChange(event.target.checked)}
          />
          <span>School holiday</span>
        </label>
      </div>
    </>
  );
}
