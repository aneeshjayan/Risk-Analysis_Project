"""
Feature engineering module.
Creates domain-relevant features: Loan-to-Income, Credit Age, purpose macro-groups.
Owner: Jayasurya Sakthivel
"""

import pandas as pd
import numpy as np


# Mapping raw loan purposes to stable macro-categories
PURPOSE_MACRO_MAP = {
    "debt_consolidation": "Debt",
    "credit_card": "Debt",
    "home_improvement": "Personal",
    "major_purchase": "Personal",
    "medical": "Personal",
    "vacation": "Personal",
    "wedding": "Personal",
    "moving": "Personal",
    "house": "Personal",
    "car": "Personal",
    "small_business": "Business",
    "educational": "Other",
    "renewable_energy": "Other",
    "other": "Other",
}


def add_loan_to_income(df: pd.DataFrame) -> pd.DataFrame:
    """Loan amount / annual income ratio — key credit risk signal."""
    if "loan_amnt" in df.columns and "annual_inc" in df.columns:
        df["loan_to_income"] = df["loan_amnt"] / (df["annual_inc"].replace(0, np.nan))
    return df


def add_credit_score_avg(df: pd.DataFrame) -> pd.DataFrame:
    """Average of FICO low/high range as a single credit score feature."""
    if "fico_range_low" in df.columns and "fico_range_high" in df.columns:
        df["credit_score"] = (df["fico_range_low"] + df["fico_range_high"]) / 2
    return df


def add_credit_age(df: pd.DataFrame) -> pd.DataFrame:
    """Convert earliest credit line to credit age in years (if raw column present)."""
    if "earliest_cr_line" in df.columns:
        df["earliest_cr_line"] = pd.to_datetime(df["earliest_cr_line"], errors="coerce")
        reference_date = pd.Timestamp("2016-01-01")
        df["credit_age_years"] = (
            (reference_date - df["earliest_cr_line"]).dt.days / 365.25
        ).round(2)
        df.drop(columns=["earliest_cr_line"], inplace=True)
    return df


def add_employment_length_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Convert emp_length string (e.g. '10+ years') to integer years."""
    if "emp_length" in df.columns and df["emp_length"].dtype == object:
        df["emp_length"] = (
            df["emp_length"]
            .str.extract(r"(\d+)")
            .astype(float)
        )
    return df


def map_purpose_macro_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Group loan purpose into Debt / Personal / Business / Other."""
    if "purpose" in df.columns:
        df["purpose_group"] = df["purpose"].map(PURPOSE_MACRO_MAP).fillna("Other")
        df.drop(columns=["purpose"], inplace=True)
    return df


def run_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering steps in sequence."""
    df = add_loan_to_income(df)
    df = add_credit_score_avg(df)
    df = add_credit_age(df)
    df = add_employment_length_numeric(df)
    df = map_purpose_macro_groups(df)
    print(f"Feature engineering complete. Shape: {df.shape}")
    return df
