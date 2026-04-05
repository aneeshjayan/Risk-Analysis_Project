"""
Data ingestion module.
Loads raw LendingClub datasets, validates schema, and merges into a single DataFrame.
Owner: Jayasurya Sakthivel
"""

import pandas as pd
from pathlib import Path


RAW_DATA_PATH = "data/raw/Lending_club_data.csv"
TARGET_COL = "default"
LEAKAGE_COLS = ["loan_status", "total_pymnt", "recoveries", "grade", "sub_grade"]


def load_raw_data(file_path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load a single CSV/parquet dataset from disk."""
    path = Path(file_path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def validate_schema(df: pd.DataFrame, required_cols: list[str]) -> None:
    """Raise ValueError if any required column is missing."""
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Schema validation failed. Missing columns: {missing}")


def remove_leakage_columns(df: pd.DataFrame, leakage_cols: list[str] = LEAKAGE_COLS) -> pd.DataFrame:
    """Drop post-loan leakage variables (payments, recoveries, grades)."""
    cols_to_drop = [c for c in leakage_cols if c in df.columns]
    return df.drop(columns=cols_to_drop)


def build_target(df: pd.DataFrame, target_col: str = TARGET_COL) -> pd.DataFrame:
    """Ensure binary PD target exists (1 = default, 0 = non-default)."""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")
    df[target_col] = df[target_col].astype(int)
    return df


def run_ingestion(file_path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Full ingestion pipeline: load -> remove leakage -> build target."""
    df = load_raw_data(file_path)
    df = remove_leakage_columns(df)
    df = build_target(df)
    print(f"Ingestion complete. Shape: {df.shape}")
    return df
