import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { formatSales, formatShortDate, weekdayName } from "../utils/format.js";

// Merge observed history and predicted future into one series.
//
// The two are kept in separate keys so they can be drawn with different styles.
//
// Days the store was closed are plotted as null rather than 0. A closed shop
// has no demand signal at all -- drawing it as a zero implies "we opened and
// sold nothing", and on a weekly-closing store it buries the actual trend
// under a row of deep spikes. Null with connectNulls={false} breaks the line
// instead, so each trading week reads as its own segment.
export function buildChartData(history, forecast) {
  if (!history) return [];

  const rows = history.points.map((point) => ({
    date: point.date,
    actual: point.is_open ? point.sales : null,
    promo: point.promo,
    isOpen: point.is_open,
  }));

  if (!forecast || forecast.points.length === 0) return rows;

  // Join the two lines at the most recent day that actually has a value,
  // which may not be the final row if the store was shut on it.
  for (let index = rows.length - 1; index >= 0; index -= 1) {
    if (rows[index].actual !== null) {
      rows[index].forecast = rows[index].actual;
      break;
    }
  }

  forecast.points.forEach((point) => {
    rows.push({
      date: point.date,
      forecast: point.is_expected_closed ? null : point.predicted_sales,
      predictedSales: point.predicted_sales,
      dayOfWeek: point.day_of_week,
      isExpectedClosed: point.is_expected_closed,
    });
  });

  return rows;
}

function ChartTooltip({ active, payload, label }) {
  if (!active || !payload || payload.length === 0) return null;

  const row = payload[0].payload;
  const isForecastRow = row.predictedSales !== undefined;

  return (
    <div className="tooltip">
      <p className="tooltip-date">{formatShortDate(label)}</p>
      {row.actual !== null && row.actual !== undefined && (
        <p className="tooltip-row">
          <span className="dot dot-actual" />
          Actual <strong>{formatSales(row.actual)}</strong>
        </p>
      )}
      {isForecastRow && !row.isExpectedClosed && (
        <p className="tooltip-row">
          <span className="dot dot-forecast" />
          Forecast <strong>{formatSales(row.predictedSales)}</strong>
        </p>
      )}
      {row.promo && <p className="tooltip-note">Promotion was running</p>}
      {row.isOpen === false && <p className="tooltip-note">Store was closed</p>}
      {row.isExpectedClosed && (
        <p className="tooltip-note">
          Usually closed on {weekdayName(row.dayOfWeek)}. No forecast issued.
        </p>
      )}
    </div>
  );
}

export default function SalesChart({ data, forecastStartDate }) {
  if (data.length === 0) {
    return <div className="chart-empty">No data to display.</div>;
  }

  return (
    <div className="chart-wrapper">
      <ResponsiveContainer width="100%" height={380}>
        <LineChart data={data} margin={{ top: 8, right: 16, bottom: 8, left: 8 }}>
          <CartesianGrid stroke="var(--grid)" strokeDasharray="3 3" vertical={false} />
          <XAxis
            dataKey="date"
            tickFormatter={formatShortDate}
            tick={{ fontSize: 12, fill: "var(--text-muted)" }}
            stroke="var(--border)"
            minTickGap={28}
          />
          <YAxis
            tickFormatter={formatSales}
            tick={{ fontSize: 12, fill: "var(--text-muted)" }}
            stroke="var(--border)"
            width={64}
          />
          <Tooltip content={<ChartTooltip />} />
          <Legend
            wrapperStyle={{ fontSize: 13, paddingTop: 8 }}
            iconType="plainline"
          />
          {forecastStartDate && (
            <ReferenceLine
              x={forecastStartDate}
              stroke="var(--accent-forecast)"
              strokeDasharray="4 4"
              label={{
                value: "forecast starts",
                position: "insideTopRight",
                fill: "var(--text-muted)",
                fontSize: 11,
              }}
            />
          )}
          <Line
            type="monotone"
            dataKey="actual"
            name="Actual sales"
            stroke="var(--accent-actual)"
            strokeWidth={2}
            dot={false}
            connectNulls={false}
          />
          <Line
            type="monotone"
            dataKey="forecast"
            name="Forecast"
            stroke="var(--accent-forecast)"
            strokeWidth={2}
            strokeDasharray="5 4"
            dot={{ r: 2.5 }}
            connectNulls
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
