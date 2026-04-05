"""Unit tests for data preprocessing pipeline."""

import pandas as pd
import numpy as np
import pytest
from src.data.preprocessing import (
    drop_high_null_columns,
    impute_missing,
    find_optimal_threshold,
)
from src.evaluation.metrics import find_optimal_threshold


def _sample_df():
    return pd.DataFrame({
        "loan_amnt": [1000, 2000, np.nan, 4000],
        "annual_inc": [50000, 60000, 70000, 80000],
        "mostly_null": [np.nan, np.nan, np.nan, 1.0],
        "default": [0, 1, 0, 1],
    })


def test_drop_high_null_columns():
    df = _sample_df()
    result = drop_high_null_columns(df, threshold=0.5)
    assert "mostly_null" not in result.columns
    assert "loan_amnt" in result.columns


def test_impute_missing():
    df = _sample_df().drop(columns=["mostly_null"])
    result = impute_missing(df)
    assert result["loan_amnt"].isnull().sum() == 0


def test_optimal_threshold():
    y_true = np.array([0, 0, 1, 1, 1])
    y_proba = np.array([0.1, 0.4, 0.6, 0.8, 0.9])
    thresh = find_optimal_threshold(y_true, y_proba)
    assert 0.1 <= thresh <= 0.9
