
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import yaml
from fredapi import Fred

# ---------------------------
# Config
# ---------------------------
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Map of FRED series -> friendly name
SERIES_MAP = {
    "CPALTT01INM657N": "CPI_AllItems_IN",
    "INDIRLTLT01STM": "Gsec10Y_IN_pct",
    "NGDPRNSAXDCINQ": "RealGDP_IN_quarterly"
}

FORWARD_FILL_QUARTERLY_TO_MONTHS = True

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("fetch_macro_fred_india")


# ---------------------------
# Helpers
# ---------------------------
def load_api_key(config_file="config.yaml"):
    """Load API key from config.yaml"""
    with open(config_file, "r") as f:
        cfg = yaml.safe_load(f)
    if cfg is None or "fred_api_key" not in cfg:
        raise RuntimeError("config.yaml missing 'fred_api_key'")
    return cfg["fred_api_key"]


def fetch_series(fred, series_id):
    """Fetch a series from FRED into a pandas Series"""
    logger.info(f"Fetching {series_id}...")
    s = fred.get_series(series_id)
    if s is None or len(s) == 0:
        logger.warning(f"{series_id} returned empty.")
        return pd.Series(dtype=float)

    s.index = pd.to_datetime(s.index)
    s = s.sort_index()
    return s


def detect_series_type(s):
    """Heuristic to guess if the series is a level or growth rate."""
    if s.empty:
        return "unknown"

    med = float(np.nanmedian(np.abs(s.dropna())))
    if med < 5:
        return "pct_or_growth"
    return "level"


def resample_month_end(s):
    """Convert any frequency series to month-end."""
    if s.empty:
        return s
    return s.resample("ME").last()


# ---------------------------
# Main pipeline
# ---------------------------
def fetch_and_process_all(config_file="config.yaml"):
    api_key = load_api_key(config_file)
    fred = Fred(api_key=api_key)

    fetched = {}
    meta = {}

    for sid, name in SERIES_MAP.items():
        s = fetch_series(fred, sid)
        if s.empty:
            continue

        series_type = detect_series_type(s)
        logger.info(f"{name}: detected type = {series_type}")

        s_m = resample_month_end(s)
        fetched[name] = s_m

        meta[name] = {
            "fred_id": sid,
            "points": len(s_m),
            "start": str(s_m.index.min().date()),
            "end": str(s_m.index.max().date()),
            "detected_type": series_type
        }

    # Combine into one DataFrame
    df = pd.concat(fetched.values(), axis=1, keys=fetched.keys())
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(0)
    df = df.sort_index()

    # Forward-fill quarterly GDP
    if "RealGDP_IN_quarterly" in df.columns and FORWARD_FILL_QUARTERLY_TO_MONTHS:
        df["RealGDP_IN_quarterly"] = df["RealGDP_IN_quarterly"].resample("M").ffill()

    # Compute MoM and YoY
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[f"{col}_MoM"] = df[col].pct_change(1)
            df[f"{col}_YoY"] = df[col].pct_change(12)

    # Save combined dataset
    out_path = OUT_DIR / "macro_fred_india.parquet"
    df.to_parquet(out_path)
    logger.info(f"Saved: {out_path}")

    # Save metadata
    meta_path = OUT_DIR / "macro_fred_india_metadata.yaml"
    with open(meta_path, "w") as f:
        yaml.safe_dump(meta, f)
    logger.info(f"Saved metadata to {meta_path}")

    return df


if __name__ == "__main__":
    logger.info("=== Running fetch_macro_fred_india.py===")
    df = fetch_and_process_all()
    print(df.tail())
