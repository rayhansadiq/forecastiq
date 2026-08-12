"""
Loading and cleaning of the raw Rossmann CSV files.

The output of build_clean_dataset() is a tidy, gap-free daily panel:
one row per (Store, Date) for every store across the full timeline, with
store metadata joined on. Feature engineering happens separately in
backend/ml/features.py.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from ml.config import STORE_CSV, TRAIN_CSV  # noqa: E402

# Rossmann encodes September as "Sept", not the usual "Sep".
MONTH_ABBREVIATIONS = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sept", 10: "Oct", 11: "Nov", 12: "Dec",
}

STORE_TYPE_CODES = {"a": 0, "b": 1, "c": 2, "d": 3}
ASSORTMENT_CODES = {"a": 0, "b": 1, "c": 2}
STATE_HOLIDAY_CODES = {"0": 0, "a": 1, "b": 2, "c": 3}


def _require_file(path: Path) -> None:
    if not path.exists():
        raise SystemExit(
            f"ERROR: {path} not found.\n"
            "Run 'python backend/data/download.py' first."
        )


def load_raw_sales() -> pd.DataFrame:
    """Read train.csv with correct dtypes."""
    _require_file(TRAIN_CSV)
    frame = pd.read_csv(TRAIN_CSV, parse_dates=["Date"], low_memory=False)
    # StateHoliday arrives as a mix of int 0 and str 'a'/'b'/'c'.
    frame["StateHoliday"] = frame["StateHoliday"].astype(str)
    return frame


def load_raw_stores() -> pd.DataFrame:
    """Read store.csv."""
    _require_file(STORE_CSV)
    return pd.read_csv(STORE_CSV)


def clean_store_metadata(stores: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the per-store metadata table.

    Missing values here are meaningful rather than random:
      - CompetitionDistance missing  -> no known competitor. Filled with the
        median distance and flagged, so the model can tell imputed from real.
      - CompetitionOpenSince* missing -> opening date unknown.
      - Promo2Since* / PromoInterval missing -> store never joined Promo2.
    """
    cleaned = stores.copy()

    cleaned["HasCompetitionInfo"] = cleaned["CompetitionDistance"].notna().astype(int)
    median_distance = cleaned["CompetitionDistance"].median()
    cleaned["CompetitionDistance"] = cleaned["CompetitionDistance"].fillna(
        median_distance
    )

    cleaned["StoreTypeCode"] = cleaned["StoreType"].map(STORE_TYPE_CODES).astype(int)
    cleaned["AssortmentCode"] = cleaned["Assortment"].map(ASSORTMENT_CODES).astype(int)

    cleaned["PromoInterval"] = cleaned["PromoInterval"].fillna("")

    # Convert the ISO year+week that Promo2 started into an actual date so we
    # can test whether Promo2 was live on any given day.
    promo2_year = cleaned["Promo2SinceYear"].astype("Int64").astype(str)
    promo2_week = cleaned["Promo2SinceWeek"].astype("Int64").astype(str)
    cleaned["Promo2StartDate"] = pd.to_datetime(
        promo2_year + "-" + promo2_week + "-1",
        format="%Y-%W-%w",
        errors="coerce",
    )

    return cleaned[
        [
            "Store",
            "StoreTypeCode",
            "AssortmentCode",
            "CompetitionDistance",
            "HasCompetitionInfo",
            "CompetitionOpenSinceMonth",
            "CompetitionOpenSinceYear",
            "Promo2",
            "Promo2StartDate",
            "PromoInterval",
        ]
    ]


def build_daily_panel(sales: pd.DataFrame) -> pd.DataFrame:
    """
    Expand the sales table to a complete (Store x Date) grid.

    Roughly 180 stores are missing about six months of 2014 in the raw data.
    Shifting by row position across those gaps would silently produce lag
    features that are not actually N calendar days old. Reindexing onto a
    dense date grid first makes every lag a true calendar lag.

    Rows created by this step are marked IsObserved = 0 and are excluded from
    training and evaluation. They exist only so that lag arithmetic is correct.
    """
    panel = sales.copy()
    panel["IsObserved"] = 1

    all_stores = np.sort(panel["Store"].unique())
    all_dates = pd.date_range(panel["Date"].min(), panel["Date"].max(), freq="D")
    complete_grid = pd.MultiIndex.from_product(
        [all_stores, all_dates], names=["Store", "Date"]
    )

    panel = (
        panel.set_index(["Store", "Date"])
        .reindex(complete_grid)
        .reset_index()
        .sort_values(["Store", "Date"])
        .reset_index(drop=True)
    )

    panel["IsObserved"] = panel["IsObserved"].fillna(0).astype(int)
    panel["Sales"] = panel["Sales"].fillna(0.0)
    panel["Customers"] = panel["Customers"].fillna(0.0)
    panel["Open"] = panel["Open"].fillna(0).astype(int)
    panel["Promo"] = panel["Promo"].fillna(0).astype(int)
    panel["SchoolHoliday"] = panel["SchoolHoliday"].fillna(0).astype(int)
    panel["StateHoliday"] = panel["StateHoliday"].fillna("0")

    # DayOfWeek is missing on filled-in rows; derive it from the date instead
    # of trusting the raw column. Pandas uses Monday=0, Rossmann uses Monday=1.
    panel["DayOfWeek"] = panel["Date"].dt.dayofweek + 1

    panel["StateHolidayCode"] = (
        panel["StateHoliday"].map(STATE_HOLIDAY_CODES).fillna(0).astype(int)
    )

    return panel


def build_clean_dataset() -> pd.DataFrame:
    """Load, clean and join everything into one daily panel."""
    sales = load_raw_sales()
    stores = clean_store_metadata(load_raw_stores())
    panel = build_daily_panel(sales)
    return panel.merge(stores, on="Store", how="left")


def describe_cleaning(panel: pd.DataFrame) -> str:
    """Human-readable summary of what cleaning did, for logs and the README."""
    observed = int(panel["IsObserved"].sum())
    imputed = int(len(panel) - observed)
    open_rows = int(((panel["IsObserved"] == 1) & (panel["Open"] == 1)).sum())
    closed_rows = int(((panel["IsObserved"] == 1) & (panel["Open"] == 0)).sum())
    open_zero_sales = int(
        (
            (panel["IsObserved"] == 1)
            & (panel["Open"] == 1)
            & (panel["Sales"] == 0)
        ).sum()
    )

    return "\n".join(
        [
            f"Panel rows            : {len(panel):,}",
            f"  observed            : {observed:,}",
            f"  gap-filled          : {imputed:,} (excluded from training)",
            f"Observed open days    : {open_rows:,}",
            f"Observed closed days  : {closed_rows:,} (excluded from training)",
            f"Open days w/ 0 sales  : {open_zero_sales:,} (excluded from training)",
            f"Stores                : {panel['Store'].nunique():,}",
            f"Date range            : {panel['Date'].min().date()}"
            f" -> {panel['Date'].max().date()}",
        ]
    )


if __name__ == "__main__":
    dataset = build_clean_dataset()
    print(describe_cleaning(dataset))
