"""
Download the Kaggle 'Rossmann Store Sales' dataset into backend/data/raw/.

Prerequisites:
  1. Free Kaggle account
  2. Legacy API key saved at C:\\Users\\<you>\\.kaggle\\kaggle.json
  3. Competition terms accepted at:
     https://www.kaggle.com/competitions/rossmann-store-sales/rules

Run from the project root:
    python backend\\data\\download.py
"""

import os
import sys
import zipfile
from pathlib import Path

COMPETITION_SLUG = "rossmann-store-sales"
EXPECTED_FILES = ["train.csv", "store.csv", "test.csv"]

# backend/data/download.py -> backend/data/raw
RAW_DATA_DIR = Path(__file__).resolve().parent / "raw"


def get_authenticated_api():
    """Authenticate against the Kaggle API, failing with a clear message."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        raise SystemExit(
            "ERROR: the 'kaggle' package is not installed.\n"
            "Fix: python -m pip install -r backend/requirements.txt"
        )

    token_path = Path.home() / ".kaggle" / "kaggle.json"
    has_env_credentials = os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY")

    if not token_path.exists() and not has_env_credentials:
        raise SystemExit(
            f"ERROR: Kaggle API credentials not found at {token_path}\n"
            "Fix:\n"
            "  1. https://www.kaggle.com/settings\n"
            "  2. Legacy API Credentials -> 'Create Legacy API Key'\n"
            f"  3. Move the downloaded kaggle.json to {token_path}"
        )

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as exc:
        raise SystemExit(
            f"ERROR: Kaggle authentication failed: {exc}\n"
            "Your kaggle.json may be malformed or the key may have been expired.\n"
            "Fix: create a new legacy key at https://www.kaggle.com/settings"
        )

    return api


def download_competition_data(api):
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading '{COMPETITION_SLUG}' into {RAW_DATA_DIR} ...")

    try:
        api.competition_download_files(
            COMPETITION_SLUG, path=str(RAW_DATA_DIR), quiet=False
        )
    except Exception as exc:
        message = str(exc)
        if "403" in message or "Forbidden" in message:
            raise SystemExit(
                "ERROR: Kaggle returned 403 Forbidden.\n"
                "This means the competition terms have not been accepted.\n"
                "Fix: open https://www.kaggle.com/competitions/"
                f"{COMPETITION_SLUG}/rules, click 'Late Submission' to accept "
                "the terms, then rerun this script."
            )
        raise SystemExit(f"ERROR: download failed: {exc}")


def extract_archives():
    """Unzip archives Kaggle produced, including nested per-file zips."""
    archives = sorted(RAW_DATA_DIR.glob("*.zip"))
    if not archives:
        print("No .zip archives found (files may already be extracted).")
        return

    for archive in archives:
        print(f"Extracting {archive.name} ...")
        with zipfile.ZipFile(archive, "r") as zip_file:
            zip_file.extractall(RAW_DATA_DIR)
        archive.unlink()

    # Kaggle sometimes nests individual .csv.zip files inside the bundle.
    for nested in sorted(RAW_DATA_DIR.glob("*.zip")):
        print(f"Extracting nested {nested.name} ...")
        with zipfile.ZipFile(nested, "r") as zip_file:
            zip_file.extractall(RAW_DATA_DIR)
        nested.unlink()


def verify_expected_files():
    missing = [name for name in EXPECTED_FILES if not (RAW_DATA_DIR / name).exists()]
    if missing:
        print(f"WARNING: expected files not found: {', '.join(missing)}")
        print(f"Contents of {RAW_DATA_DIR}:")
        for item in sorted(RAW_DATA_DIR.iterdir()):
            print(f"  - {item.name}")
        return False

    print("\nDownload complete. Files in backend/data/raw:")
    for path in sorted(RAW_DATA_DIR.iterdir()):
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"  {path.name:<28} {size_mb:>7.2f} MB")
    return True


def main():
    api = get_authenticated_api()
    download_competition_data(api)
    extract_archives()
    if not verify_expected_files():
        sys.exit(1)


if __name__ == "__main__":
    main()
