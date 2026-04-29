"""
Data ingestion module.
Loads raw LendingClub datasets, filters to completed loans, builds binary target.
Owner: Jayasurya Sakthivel
"""

import pandas as pd
from pathlib import Path


RAW_DATA_PATH = "data/raw/Lending_club_data.csv"
TARGET_COL    = "default"

COMPLETED_STATUS = [
    "Fully Paid",
    "Charged Off",
    "Default",
    "Does not meet the credit policy. Status:Fully Paid",
    "Does not meet the credit policy. Status:Charged Off",
]

DEFAULT_STATUS = [
    "Charged Off",
    "Default",
    "Does not meet the credit policy. Status:Charged Off",
]

# Post-loan leakage columns (known only after outcome — drop after building target)
LEAKAGE_COLS = ["total_pymnt", "recoveries", "grade", "sub_grade", "loan_status"]


def load_raw_data(file_path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load CSV or Parquet from disk."""
    path = Path(file_path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def validate_schema(df: pd.DataFrame, required_cols: list[str]) -> None:
    """Raise ValueError if any required column is missing."""
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Schema validation failed. Missing columns: {missing}")


def filter_completed_loans(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only fully resolved loans — drop 'Current', 'In Grace Period', etc."""
    if "loan_status" not in df.columns:
        raise ValueError("'loan_status' column not found — cannot filter completed loans.")
    before = len(df)
    df = df[df["loan_status"].isin(COMPLETED_STATUS)].copy()
    print(f"  Filtered to completed loans: {before:,} → {len(df):,} rows")
    return df


def build_target(df: pd.DataFrame) -> pd.DataFrame:
    """Create binary default indicator from loan_status (1=default, 0=paid)."""
    if "loan_status" not in df.columns:
        raise ValueError("'loan_status' column required to build target.")
    df[TARGET_COL] = df["loan_status"].isin(DEFAULT_STATUS).astype(int)
    print(f"  Default rate: {df[TARGET_COL].mean():.3f}")
    return df


def remove_leakage_columns(df: pd.DataFrame,
                            leakage_cols: list[str] = LEAKAGE_COLS) -> pd.DataFrame:
    """Drop post-loan leakage variables. Call AFTER build_target."""
    cols_to_drop = [c for c in leakage_cols if c in df.columns]
    return df.drop(columns=cols_to_drop)


def is_preprocessed(df: pd.DataFrame) -> bool:
    """Return True if the CSV is already feature-engineered and encoded."""
    return "loan_status" not in df.columns and "default" in df.columns


def run_ingestion(file_path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load data. If already preprocessed, return as-is. Otherwise run full pipeline."""
    df = load_raw_data(file_path)
    print(f"Loaded data: {df.shape}")

    if is_preprocessed(df):
        print("  Detected pre-processed dataset — skipping filter/target/leakage steps.")
        print(f"Ingestion complete. Shape: {df.shape}")
        return df

    # Raw LendingClub data path
    df = filter_completed_loans(df)
    df = build_target(df)
    df = remove_leakage_columns(df)
    print(f"Ingestion complete. Shape: {df.shape}")
    return df
