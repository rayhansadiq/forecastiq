import { formatDecimal, formatPercent, formatSales } from "../utils/format.js";

// Held-out test performance, read straight from the model artifact.
//
// Shown next to the seasonal-naive baseline on purpose: an error figure with
// nothing to compare it against tells the reader nothing about whether the
// model is actually doing useful work.
export default function ModelSummary({ modelInfo }) {
  if (!modelInfo) return null;

  const { metrics, baseline_metrics: baseline } = modelInfo;
  const maeImprovement =
    ((baseline.mae - metrics.mae) / baseline.mae) * 100;

  return (
    <section className="card">
      <header className="card-header">
        <h2>Model performance</h2>
        <span className="badge">held-out test set</span>
      </header>

      <table className="metrics-table">
        <thead>
          <tr>
            <th />
            <th>Seasonal naive</th>
            <th>ForecastIQ</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <th>MAE</th>
            <td>{formatSales(baseline.mae)}</td>
            <td className="highlight">{formatSales(metrics.mae)}</td>
          </tr>
          <tr>
            <th>RMSE</th>
            <td>{formatSales(baseline.rmse)}</td>
            <td className="highlight">{formatSales(metrics.rmse)}</td>
          </tr>
          <tr>
            <th>RMSPE</th>
            <td>{formatDecimal(baseline.rmspe)}</td>
            <td className="highlight">{formatDecimal(metrics.rmspe)}</td>
          </tr>
        </tbody>
      </table>

      <dl className="facts">
        <div>
          <dt>Average error</dt>
          <dd>{formatPercent(metrics.mae_as_pct_of_mean)} of mean daily sales</dd>
        </div>
        <div>
          <dt>Improvement over baseline</dt>
          <dd>{formatPercent(maeImprovement)} lower MAE</dd>
        </div>
        <div>
          <dt>Trained on</dt>
          <dd>
            {modelInfo.train_start_date} to {modelInfo.train_end_date} (
            {formatSales(modelInfo.n_train_rows)} rows)
          </dd>
        </div>
        <div>
          <dt>Evaluated on</dt>
          <dd>
            {modelInfo.test_start_date} to {modelInfo.test_end_date} (
            {formatSales(modelInfo.n_test_rows)} rows)
          </dd>
        </div>
      </dl>

      <p className="card-note">
        The test period comes strictly after every date the model trained on.
        Nothing was shuffled, so these figures reflect predicting genuinely
        unseen future days rather than interpolating within known ones.
      </p>
    </section>
  );
}
