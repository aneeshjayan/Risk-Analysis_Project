"""
Data preprocessing module.
Handles missing values, encoding, scaling, and train/test splitting.
Owner: Jayasurya Sakthivel
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path


TARGET_COL   = "default"
TEST_SIZE    = 0.2
RANDOM_STATE = 42

# Categorical columns present AFTER feature engineering
# (purpose is already replaced by purpose_group at FE stage)
CAT_COLS = ["home_ownership", "purpose_group", "application_type"]

# Only these continuous columns get standardised
NUMERICAL_COLS = [
    # Borrower-level features
    "loan_amnt", "annual_inc", "credit_score",
    "dti", "loan_to_income", "credit_age_years",
    # FRED macroeconomic features (merged from second dataset)
    "unemployment_rate",    # US unemployment % at loan issuance
    "fed_funds_rate",       # Federal funds rate % at loan issuance
    "inflation_cpi",        # CPI index at loan issuance
    "real_disposable_inc",  # Real disposable income (billions $)
    "unemp_3m_change",      # 3-month unemployment momentum
    # Temporal features (extracted from issue_d)
    "issue_year",
    "issue_quarter",
]


def drop_high_null_columns(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Drop columns where null fraction exceeds threshold."""
    null_frac = df.isnull().mean()
    return df.loc[:, null_frac < threshold]


def impute_missing(df: pd.DataFrame, num_strategy: str = "median") -> pd.DataFrame:
    """Impute numerical columns with median, categorical with mode."""
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().any():
            fill_val = df[col].median() if num_strategy == "median" else df[col].mean()
            df[col] = df[col].fillna(fill_val)
    for col in df.select_dtypes(include=["object", "category"]).columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])
    return df


def encode_categoricals(df: pd.DataFrame,
                         cat_cols: list[str] = CAT_COLS) -> pd.DataFrame:
    """One-hot encode categorical columns; drop first to avoid multicollinearity."""
    existing = [c for c in cat_cols if c in df.columns]
    return pd.get_dummies(df, columns=existing, drop_first=True)


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    numerical_cols: list[str] = NUMERICAL_COLS,
    scaler_path: str = "data/processed/scaler.joblib",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit scaler on train numerical cols only, apply to test. Persists scaler."""
    cols = [c for c in numerical_cols if c in X_train.columns]
    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test  = X_test.copy()
    X_train[cols] = scaler.fit_transform(X_train[cols])
    X_test[cols]  = scaler.transform(X_test[cols])
    Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)
    return X_train, X_test


def split_data(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified 80/20 train-test split preserving class balance."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, stratify=y,
                            random_state=random_state)


def run_preprocessing(
    df: pd.DataFrame,
    scaler_path: str = "data/processed/scaler.joblib",
) -> tuple:
    """Full preprocessing pipeline → (X_train, X_test, y_train, y_test)."""
    df = drop_high_null_columns(df)
    df = impute_missing(df)
    df = encode_categoricals(df, CAT_COLS)
    X_train, X_test, y_train, y_test = split_data(df)
    X_train, X_test = scale_features(X_train, X_test,
                                      numerical_cols=NUMERICAL_COLS,
                                      scaler_path=scaler_path)
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"Default rate (train): {y_train.mean():.4f}")
    return X_train, X_test, y_train, y_test
