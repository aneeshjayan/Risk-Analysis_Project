"""
macro_features.py — FRED macroeconomic feature integration.

Fetches 5 key economic indicators from the Federal Reserve (FRED API),
merges them with loan data on issue_d (year-month), and exposes a
helper for real-time inference (current macro snapshot).

Owner: Aneesh Jayan Prabhu

Required:
    pip install fredapi
    FRED_API_KEY in .env  (free key at https://fred.stlouisfed.org/docs/api/api_key.html)
"""

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ── FRED series to fetch ───────────────────────────────────────────────────────
# Each key becomes a column name in the merged dataset.
FRED_SERIES = {
    "unemployment_rate":   "UNRATE",    # Monthly US unemployment %
    "fed_funds_rate":      "FEDFUNDS",  # Federal funds effective rate %
    "inflation_cpi":       "CPIAUCSL",  # Consumer Price Index (all urban)
    "real_disposable_inc": "DSPIC96",   # Real disposable personal income (billions $)
    "recession_flag":      "USREC",     # NBER recession indicator: 1=recession, 0=expansion
}

# Columns added after feature derivation (fed into the scaler / model)
MACRO_NUMERICAL_COLS = [
    "unemployment_rate",
    "fed_funds_rate",
    "inflation_cpi",
    "real_disposable_inc",
    "unemp_3m_change",     # momentum: 3-month change in unemployment
]

# Binary — not scaled
MACRO_BINARY_COLS = ["recession_flag"]

# Fallback values used when FRED key is absent at inference time
# (approximate 2024 long-run averages / current values)
MACRO_FALLBACK = {
    "unemployment_rate":   4.0,
    "fed_funds_rate":      5.33,
    "inflation_cpi":       314.0,
    "real_disposable_inc": 15_800.0,
    "unemp_3m_change":     0.0,
    "recession_flag":      0,
}

CACHE_PATH = Path("data/processed/macro_features.parquet")


# ── Fetch & cache ──────────────────────────────────────────────────────────────
def fetch_macro_data(
    api_key: str,
    start: str = "2007-01-01",
    end:   str = "2019-12-31",
) -> pd.DataFrame:
    """
    Download FRED series and return a monthly DataFrame indexed by Period.

    Args:
        api_key: FRED API key (free at fred.stlouisfed.org)
        start:   Start date string (YYYY-MM-DD)
        end:     End date string (YYYY-MM-DD)

    Returns:
        DataFrame with one row per month, macro indicator columns.
    """
    try:
        from fredapi import Fred
    except ImportError:
        raise ImportError(
            "fredapi not installed. Run:  pip install fredapi"
        )

    fred   = Fred(api_key=api_key)
    frames = {}

    # Alternative series IDs tried if primary fails (FRED sometimes returns 500)
    FALLBACK_IDS = {
        "CPIAUCSL": "CPILFESL",   # Core CPI (ex food & energy) — same monthly frequency
        "DSPIC96":  "PI",         # Personal Income (nominal) if real not available
        "USREC":    None,         # No clean alternative — will use fallback value
    }

    for col_name, series_id in FRED_SERIES.items():
        fetched = False
        for sid in [series_id, FALLBACK_IDS.get(series_id)]:
            if sid is None:
                break
            try:
                s = fred.get_series(
                    sid,
                    observation_start=start,
                    observation_end=end,
                )
                # Convert to monthly period for clean joining
                s.index = pd.PeriodIndex(s.index, freq="M")
                frames[col_name] = s
                label = f"{sid}" if sid == series_id else f"{sid} (fallback for {series_id})"
                print(f"  ✓ FRED {label} ({col_name}): {len(s)} obs")
                fetched = True
                break
            except Exception as e:
                print(f"  ✗ FRED {sid} failed: {e}")

        if not fetched:
            # Use fallback constant so missing series doesn't break the merge
            fallback_val = MACRO_FALLBACK.get(col_name, 0)
            print(f"  ⚠ Using fallback constant for {col_name}: {fallback_val}")
            frames[col_name] = pd.Series(dtype=float)   # filled below

    macro_df = pd.DataFrame(frames)
    macro_df.index.name = "year_month"

    # Fill any columns that are entirely empty with their fallback value
    for col_name, fallback_val in MACRO_FALLBACK.items():
        if col_name in macro_df.columns and macro_df[col_name].isna().all():
            macro_df[col_name] = fallback_val
            print(f"  ⚠ {col_name}: all NaN — filled with fallback {fallback_val}")

    # Forward-fill quarterly series to monthly (pandas 2.x: use .ffill() not fillna(method=))
    macro_df = macro_df.ffill()

    # ── Derived features ───────────────────────────────────────────────────────
    # Unemployment momentum: worsening economy → higher default risk
    macro_df["unemp_3m_change"] = macro_df["unemployment_rate"].diff(3).round(4)

    # Fill NaN at edges from diff (pandas 2.x compatible)
    macro_df = macro_df.bfill().fillna(0)

    print(f"  Macro DataFrame shape: {macro_df.shape}")
    return macro_df


def load_or_fetch_macro(
    api_key: str,
    start: str = "2007-01-01",
    end:   str = "2019-12-31",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Return macro DataFrame from cache if available, else fetch from FRED.

    Args:
        api_key:       FRED API key
        start/end:     Date range (only used on first fetch)
        force_refresh: Re-download even if cache exists

    Returns:
        Monthly macro DataFrame.
    """
    if CACHE_PATH.exists() and not force_refresh:
        print(f"  Loading macro features from cache: {CACHE_PATH}")
        df = pd.read_parquet(CACHE_PATH)
        # Restore PeriodIndex (parquet converts it to datetime)
        if not isinstance(df.index, pd.PeriodIndex):
            df.index = pd.PeriodIndex(df.index, freq="M")
        return df

    print("  Fetching macro features from FRED …")
    df = fetch_macro_data(api_key, start=start, end=end)
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Convert to timestamp for parquet serialisation
    df_save = df.copy()
    df_save.index = df_save.index.to_timestamp()
    df_save.to_parquet(CACHE_PATH)
    print(f"  Cached → {CACHE_PATH}")
    return df


def merge_macro_features(
    loans_df: pd.DataFrame,
    macro_df: pd.DataFrame,
    issue_col: str = "issue_d",
) -> pd.DataFrame:
    """
    Left-join macro features to loans_df on year-month of issue_d.

    Args:
        loans_df:  Loan DataFrame — must contain issue_col
        macro_df:  Monthly macro DataFrame (PeriodIndex)
        issue_col: Column name of loan issue date

    Returns:
        loans_df with macro columns appended; issue_col dropped after join.
    """
    if issue_col not in loans_df.columns:
        print(f"  WARNING: '{issue_col}' not found — skipping macro merge.")
        # Return with safe fallback values so training still runs
        for col, val in MACRO_FALLBACK.items():
            loans_df[col] = val
        return loans_df

    loans = loans_df.copy()

    # Parse issue_d to monthly Period
    loans["_ym"] = (
        pd.to_datetime(loans[issue_col], format="mixed", errors="coerce")
        .dt.to_period("M")
    )

    # Prepare macro for merge
    macro_reset = macro_df.reset_index()   # year_month column
    macro_reset = macro_reset.rename(columns={"year_month": "_ym"})

    merged = loans.merge(macro_reset, on="_ym", how="left")

    # Fill any loans outside FRED date range with fallback
    for col, val in MACRO_FALLBACK.items():
        if col in merged.columns:
            merged[col] = merged[col].fillna(val)

    # Drop helper columns
    merged = merged.drop(columns=["_ym", issue_col], errors="ignore")

    missing = merged[list(FRED_SERIES.keys()) + ["unemp_3m_change"]].isnull().sum().sum()
    if missing > 0:
        print(f"  WARNING: {missing} remaining NaN after macro merge — filling with fallback")
        for col, val in MACRO_FALLBACK.items():
            if col in merged.columns:
                merged[col] = merged[col].fillna(val)

    print(f"  Macro merge complete. Shape: {merged.shape}")
    return merged


# ── Real-time inference helper ─────────────────────────────────────────────────
def get_current_macro(api_key: str = "") -> dict:
    """
    Fetch the most recent FRED values for real-time inference.

    Called by the FastAPI backend at /predict time — the loan hasn't been
    issued yet, so we use current economic conditions as the macro context.

    Returns dict with all MACRO_NUMERICAL_COLS + MACRO_BINARY_COLS values.
    Falls back to MACRO_FALLBACK if API key is absent or call fails.
    """
    if not api_key:
        print("  No FRED_API_KEY — using fallback macro values for inference.")
        return dict(MACRO_FALLBACK)

    try:
        from fredapi import Fred
        fred = Fred(api_key=api_key)

        result = {}
        for col_name, series_id in FRED_SERIES.items():
            try:
                s = fred.get_series(series_id)
                result[col_name] = float(s.dropna().iloc[-1])
            except Exception:
                result[col_name] = MACRO_FALLBACK[col_name]

        # Derive momentum (need last 4 obs of unemployment)
        try:
            unemp = fred.get_series("UNRATE").dropna()
            if len(unemp) >= 4:
                result["unemp_3m_change"] = float(unemp.iloc[-1] - unemp.iloc[-4])
            else:
                result["unemp_3m_change"] = 0.0
        except Exception:
            result["unemp_3m_change"] = MACRO_FALLBACK["unemp_3m_change"]

        print(f"  Current macro: unemployment={result['unemployment_rate']:.1f}% "
              f"fed_funds={result['fed_funds_rate']:.2f}% "
              f"recession={result['recession_flag']:.0f}")
        return result

    except Exception as e:
        print(f"  FRED fetch failed ({e}) — using fallback macro values.")
        return dict(MACRO_FALLBACK)
