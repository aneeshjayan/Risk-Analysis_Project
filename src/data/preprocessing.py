"""
Data preprocessing module.
Handles missing values, encoding, and train/test splitting with stratification.
Owner: Jayasurya Sakthivel
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path


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


def encode_categoricals(df: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    """One-hot encode categorical columns, drop first to avoid multicollinearity."""
    existing = [c for c in cat_cols if c in df.columns]
    return pd.get_dummies(df, columns=existing, drop_first=True)


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    scaler_path: str = "data/processed/scaler.joblib",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit scaler on train, apply to test. Persists scaler to disk."""
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )
    Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)
    return X_train_scaled, X_test_scaled


def split_data(
    df: pd.DataFrame,
    target_col: str = "default",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified 80/20 train-test split to preserve class balance."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def run_preprocessing(df: pd.DataFrame, config: dict) -> tuple:
    """Full preprocessing pipeline."""
    df = drop_high_null_columns(df)
    df = impute_missing(df)
    df = encode_categoricals(df, config["preprocessing"]["categorical_cols"])
    X_train, X_test, y_train, y_test = split_data(
        df,
        target_col=config["data"]["target_column"],
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
    )
    X_train, X_test = scale_features(X_train, X_test)
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"Default rate (train): {y_train.mean():.4f}")
    return X_train, X_test, y_train, y_test
