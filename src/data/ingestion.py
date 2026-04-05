"""
Data ingestion module.
Loads raw LendingClub datasets, validates schema, and merges into a single DataFrame.
Owner: Jayasurya Sakthivel
"""

import pandas as pd
import yaml
from pathlib import Path


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_raw_data(file_path: str) -> pd.DataFrame:
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


def remove_leakage_columns(df: pd.DataFrame, leakage_cols: list[str]) -> pd.DataFrame:
    """Drop post-loan leakage variables (payments, recoveries, grades)."""
    cols_to_drop = [c for c in leakage_cols if c in df.columns]
    return df.drop(columns=cols_to_drop)


def build_target(df: pd.DataFrame, target_col: str = "default") -> pd.DataFrame:
    """Ensure binary PD target exists (1 = default, 0 = non-default)."""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")
    df[target_col] = df[target_col].astype(int)
    return df


def run_ingestion(config_path: str = "config/config.yaml") -> pd.DataFrame:
    """Full ingestion pipeline: load -> validate -> remove leakage -> build target."""
    config = load_config(config_path)
    df = load_raw_data(config["data"]["raw_path"])
    remove_leakage_columns(df, config["preprocessing"]["drop_leakage_cols"])
    df = build_target(df, config["data"]["target_column"])
    print(f"Ingestion complete. Shape: {df.shape}")
    return df
