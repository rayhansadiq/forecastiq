"""
Train and honestly evaluate the ForecastIQ demand model.

Run from the project root:
    python backend/ml/train.py
    python backend/ml/train.py --model rf
    python backend/ml/train.py --sample-stores 200      (faster iteration)

Methodology
-----------
Split      : time-based. The final TEST_PERIOD_DAYS days of the timeline are
             held out. Nothing is shuffled, so the model is always evaluated on
             dates strictly later than everything it trained on.
Leakage    : all lag and rolling features are shifted back by the forecast
             horizon (see backend/ml/features.py). Customers is excluded because
             it is measured on the same day as the target.
Baseline   : seasonal naive -- average sales on the same weekday over the three
             most recent comparable weeks. The model has to beat this to be
             worth anything, and the comparison is reported either way.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data.clean import build_clean_dataset, describe_cleaning  # noqa: E402
from ml.config import (  # noqa: E402
    FEATURE_COLUMNS,
    FORECAST_HORIZON_DAYS,
    LAG_DAYS,
    METRICS_PATH,
    MODEL_PATH,
    TARGET_COLUMN,
    TEST_PERIOD_DAYS,
)
from ml.features import build_feature_frame, select_modelling_rows  # noqa: E402

RANDOM_SEED = 42


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train the ForecastIQ model.")
    parser.add_argument(
        "--model",
        choices=["hgb", "rf"],
        default="hgb",
        help=(
            "hgb = HistGradientBoostingRegressor (default, fast at this scale). "
            "rf = RandomForestRegressor (slower, more memory)."
        ),
    )
    parser.add_argument(
        "--sample-stores",
        type=int,
        default=None,
        help="Train on a random subset of stores for faster iteration.",
    )
    return parser.parse_args()


def split_by_date(frame: pd.DataFrame, test_period_days: int):
    """
    Hold out the final `test_period_days` of the timeline.

    Deliberately not train_test_split: a random split would let the model learn
    from days that come after the days it is scored on, which inflates the
    metrics and would not be reproducible in production.
    """
    last_date = frame["Date"].max()
    test_start_date = last_date - pd.Timedelta(days=test_period_days - 1)

    train_frame = frame[frame["Date"] < test_start_date]
    test_frame = frame[frame["Date"] >= test_start_date]
    return train_frame, test_frame, test_start_date, last_date


def build_model(model_choice: str):
    if model_choice == "rf":
        return RandomForestRegressor(
            n_estimators=80,
            min_samples_leaf=5,
            max_features=0.6,
            n_jobs=-1,
            random_state=RANDOM_SEED,
        )
    return HistGradientBoostingRegressor(
        max_iter=400,
        learning_rate=0.08,
        min_samples_leaf=20,
        early_stopping=False,
        random_state=RANDOM_SEED,
    )


def seasonal_naive_prediction(frame: pd.DataFrame) -> np.ndarray:
    """
    Baseline: average of the same weekday over the three most recent
    comparable weeks. All three lags are multiples of 7, so they land on the
    same weekday as the target row.
    """
    return frame[[f"SalesLag{lag}" for lag in LAG_DAYS]].mean(axis=1).to_numpy()


def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """MAE, RMSE and RMSPE. RMSPE is the metric the original competition used."""
    mae = mean_absolute_error(actual, predicted)
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    rmspe = float(np.sqrt(np.mean(((actual - predicted) / actual) ** 2)))
    return {
        "mae": float(mae),
        "rmse": rmse,
        "rmspe": rmspe,
        "mean_actual_sales": float(np.mean(actual)),
        "mae_as_pct_of_mean": float(mae / np.mean(actual) * 100),
    }


def report_feature_importance(
    model, x_test, y_test, feature_names, top_n: int = 12, sample_size: int = 10_000
) -> list:
    """
    Rank features by importance.

    Tree ensembles like RandomForest expose feature_importances_ directly.
    HistGradientBoostingRegressor does not, so fall back to permutation
    importance measured on a subsample of the held-out test set: shuffle one
    column at a time and see how much the error gets worse.
    """
    importances = getattr(model, "feature_importances_", None)
    method = "impurity"

    if importances is None:
        from sklearn.inspection import permutation_importance

        method = "permutation"
        sample = x_test.sample(
            min(sample_size, len(x_test)), random_state=RANDOM_SEED
        )
        result = permutation_importance(
            model,
            sample,
            y_test[x_test.index.get_indexer(sample.index)],
            n_repeats=3,
            random_state=RANDOM_SEED,
            scoring="neg_mean_absolute_error",
        )
        importances = result.importances_mean

    ranked = sorted(
        zip(feature_names, importances), key=lambda pair: pair[1], reverse=True
    )
    print(f"\nTop {top_n} features ({method} importance):")
    for name, score in ranked[:top_n]:
        print(f"  {name:<24} {score:,.4f}")
    return [
        {"feature": name, "importance": float(score), "method": method}
        for name, score in ranked
    ]


def main():
    arguments = parse_arguments()

    print("Loading and cleaning raw data ...")
    panel = build_clean_dataset()
    print(describe_cleaning(panel))

    if arguments.sample_stores:
        sampled = (
            pd.Series(panel["Store"].unique())
            .sample(arguments.sample_stores, random_state=RANDOM_SEED)
            .to_numpy()
        )
        panel = panel[panel["Store"].isin(sampled)]
        print(f"\nSubsampled to {len(sampled)} stores for faster training.")

    print("\nBuilding features ...")
    featured = build_feature_frame(panel)
    modelling_frame = select_modelling_rows(featured)
    print(f"Usable modelling rows : {len(modelling_frame):,}")

    train_frame, test_frame, test_start, last_date = split_by_date(
        modelling_frame, TEST_PERIOD_DAYS
    )

    print("\nTime-based split (no shuffling):")
    print(
        f"  train : {train_frame['Date'].min().date()} -> "
        f"{train_frame['Date'].max().date()}  ({len(train_frame):,} rows)"
    )
    print(
        f"  test  : {test_frame['Date'].min().date()} -> "
        f"{test_frame['Date'].max().date()}  ({len(test_frame):,} rows)"
    )

    x_train = train_frame[FEATURE_COLUMNS]
    y_train = train_frame[TARGET_COLUMN].to_numpy()
    x_test = test_frame[FEATURE_COLUMNS]
    y_test = test_frame[TARGET_COLUMN].to_numpy()

    baseline_metrics = compute_metrics(y_test, seasonal_naive_prediction(test_frame))

    print(f"\nTraining {arguments.model} on {len(x_train):,} rows ...")
    model = build_model(arguments.model)
    model.fit(x_train, y_train)

    model_metrics = compute_metrics(y_test, model.predict(x_test))
    importance = report_feature_importance(model, x_test, y_test, FEATURE_COLUMNS)

    improvement_pct = (
        (baseline_metrics["mae"] - model_metrics["mae"])
        / baseline_metrics["mae"]
        * 100
    )

    print("\n" + "=" * 62)
    print("HELD-OUT TEST RESULTS")
    print("=" * 62)
    print(f"{'':<22}{'seasonal naive':>18}{'model':>18}")
    print(f"{'MAE':<22}{baseline_metrics['mae']:>18,.1f}{model_metrics['mae']:>18,.1f}")
    print(
        f"{'RMSE':<22}{baseline_metrics['rmse']:>18,.1f}"
        f"{model_metrics['rmse']:>18,.1f}"
    )
    print(
        f"{'RMSPE':<22}{baseline_metrics['rmspe']:>18.4f}"
        f"{model_metrics['rmspe']:>18.4f}"
    )
    print(f"\nMean actual sales on test set : {model_metrics['mean_actual_sales']:,.1f}")
    print(f"Model MAE as % of mean sales  : {model_metrics['mae_as_pct_of_mean']:.2f}%")
    print(f"MAE improvement over baseline : {improvement_pct:.1f}%")
    print("=" * 62)

    artifact = {
        "model": model,
        "model_type": arguments.model,
        "feature_columns": FEATURE_COLUMNS,
        "forecast_horizon_days": FORECAST_HORIZON_DAYS,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "train_start_date": str(train_frame["Date"].min().date()),
        "train_end_date": str(train_frame["Date"].max().date()),
        "test_start_date": str(test_start.date()),
        "test_end_date": str(last_date.date()),
        "last_observed_date": str(last_date.date()),
        "n_train_rows": int(len(train_frame)),
        "n_test_rows": int(len(test_frame)),
        "sklearn_version": sklearn.__version__,
        "metrics": model_metrics,
        "baseline_metrics": baseline_metrics,
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)
    print(f"\nSaved model to {MODEL_PATH}")
    print(f"Model file size: {MODEL_PATH.stat().st_size / (1024 * 1024):.1f} MB")

    metrics_record = {
        key: value for key, value in artifact.items() if key != "model"
    }
    metrics_record["feature_importance"] = importance[:20]
    METRICS_PATH.write_text(json.dumps(metrics_record, indent=2))
    print(f"Saved metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()
