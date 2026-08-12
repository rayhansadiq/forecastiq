"""
Print a factual summary of the raw Rossmann data so we know exactly what
we are modeling before writing any cleaning or feature code.

Run from the project root:
    python backend\\scripts\\inspect_data.py
"""

from pathlib import Path

import pandas as pd

RAW_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)


def load_raw_frame(filename: str) -> pd.DataFrame:
    path = RAW_DATA_DIR / filename
    if not path.exists():
        raise SystemExit(
            f"ERROR: {path} not found.\n"
            "Run 'python backend\\data\\download.py' first."
        )
    # StateHoliday mixes '0' (int in some rows, str in others) with 'a'/'b'/'c'.
    return pd.read_csv(path, low_memory=False)


def summarize(name: str, frame: pd.DataFrame) -> None:
    print("=" * 70)
    print(f"{name}  shape={frame.shape}")
    print("=" * 70)
    print("\n-- dtypes --")
    print(frame.dtypes)
    print("\n-- head --")
    print(frame.head())
    print("\n-- missing values per column --")
    missing = frame.isna().sum()
    print(missing[missing > 0] if missing.any() else "(none)")
    print()


def main():
    train = load_raw_frame("train.csv")
    store = load_raw_frame("store.csv")

    summarize("train.csv", train)
    summarize("store.csv", store)

    print("=" * 70)
    print("KEY FACTS")
    print("=" * 70)
    train["Date"] = pd.to_datetime(train["Date"])
    print(f"Date range        : {train['Date'].min().date()} -> {train['Date'].max().date()}")
    print(f"Unique stores     : {train['Store'].nunique()}")
    print(f"Total rows        : {len(train):,}")
    print(f"Rows with Open==0 : {(train['Open'] == 0).sum():,}")
    print(f"Rows with Sales==0: {(train['Sales'] == 0).sum():,}")
    print(f"Sales mean        : {train['Sales'].mean():,.2f}")
    print(f"Sales median      : {train['Sales'].median():,.2f}")
    print(f"Sales max         : {train['Sales'].max():,.2f}")

    print("\nStateHoliday value counts (raw, mixed types):")
    print(train["StateHoliday"].astype(str).value_counts())

    print("\nRows per store - smallest 5 (edge case: sparse-history stores):")
    print(train.groupby("Store").size().sort_values().head())

    print("\nStore metadata - StoreType / Assortment distribution:")
    print(store["StoreType"].value_counts())
    print(store["Assortment"].value_counts())


if __name__ == "__main__":
    main()
