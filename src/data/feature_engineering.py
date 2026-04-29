"""
Feature engineering module.
Creates domain-relevant features: Loan-to-Income, Credit Age, purpose macro-groups.
Owner: Jayasurya Sakthivel
"""

import pandas as pd
import numpy as np


# Anchored reference date for reproducible credit-age calculation
REFERENCE_DATE = pd.Timestamp("2024-01-01")

PURPOSE_MACRO_MAP = {
    "debt_consolidation": "Debt",
    "credit_card":        "Debt",
    "home_improvement":   "Asset",
    "car":                "Asset",
    "house":              "Asset",
    "major_purchase":     "Asset",
    "renewable_energy":   "Asset",
    "small_business":     "Business",
    "medical":            "Personal",
    "vacation":           "Personal",
    "wedding":            "Personal",
    "moving":             "Personal",
    "educational":        "Other",
    "other":              "Other",
}

EMP_LENGTH_MAP = {
    "< 1 year": 0,
    "1 year":   1, "2 years": 2, "3 years": 3,
    "4 years":  4, "5 years": 5, "6 years": 6,
    "7 years":  7, "8 years": 8, "9 years": 9,
    "10+ years": 10,
}


def parse_term(df: pd.DataFrame) -> pd.DataFrame:
    """Convert ' 36 months' string to integer 36."""
    if "term" in df.columns and df["term"].dtype == object:
        df["term"] = df["term"].str.strip().str.replace(" months", "", regex=False).astype(int)
    return df


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
    """Convert earliest credit line to credit age in years."""
    if "earliest_cr_line" in df.columns:
        df["earliest_cr_line"] = pd.to_datetime(
            df["earliest_cr_line"], format="%b-%Y", errors="coerce"
        )
        df["credit_age_years"] = (
            (REFERENCE_DATE - df["earliest_cr_line"]).dt.days / 365.25
        ).round(2)
        df.drop(columns=["earliest_cr_line"], inplace=True)
    return df


def add_employment_length_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Map emp_length string to integer years via explicit lookup."""
    if "emp_length" in df.columns and df["emp_length"].dtype == object:
        df["emp_length"] = df["emp_length"].map(EMP_LENGTH_MAP)
    return df


def map_purpose_macro_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Group loan purpose into Debt / Asset / Business / Personal / Other."""
    if "purpose" in df.columns:
        df["purpose_group"] = df["purpose"].map(PURPOSE_MACRO_MAP).fillna("Other")
        df.drop(columns=["purpose"], inplace=True)
    return df


def extract_issue_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract year and quarter from issue_d as numeric features.

    Captures economic cycle effects — loans issued in 2009 (recession) have
    very different default rates than identical profiles in 2015 (expansion).
    issue_d itself is dropped here; macro_features.py uses it BEFORE this step.
    """
    if "issue_d" in df.columns:
        issue_dt = pd.to_datetime(df["issue_d"], format="mixed", errors="coerce")
        df["issue_year"]    = issue_dt.dt.year.fillna(2015).astype(int)
        df["issue_quarter"] = issue_dt.dt.quarter.fillna(2).astype(int)
        df.drop(columns=["issue_d"], inplace=True)
        print(f"  Extracted issue_year / issue_quarter from issue_d")
    return df


def run_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering steps in sequence."""
    df = parse_term(df)
    df = add_loan_to_income(df)
    df = add_credit_score_avg(df)
    df = add_credit_age(df)
    df = add_employment_length_numeric(df)
    df = map_purpose_macro_groups(df)
    df = extract_issue_date_features(df)   # must be AFTER macro merge (issue_d used there)
    print(f"Feature engineering complete. Shape: {df.shape}")
    return df
