"""
train_model.py — Full end-to-end training pipeline.

Trains XGBoost, LightGBM, Random Forest + Soft-Voting Ensemble.
Evaluates all models, runs SHAP analysis, and saves all artefacts.

Usage:
    python train_model.py
    python train_model.py --data /path/to/accepted_loans.csv
"""

import os
import sys
import argparse
import warnings
import json
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (no display needed)
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                              recall_score, brier_score_loss, classification_report)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ── Paths ─────────────────────────────────────────────────────────────────────
DEFAULT_CSV = r"C:\Users\balam\Downloads\archive\accepted_2007_to_2018q4.csv\accepted_2007_to_2018Q4.csv"
MODELS_DIR  = Path(__file__).parent.parent / "models"
REPORTS_DIR = Path(__file__).parent.parent / "reports" / "figures"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

# ── Feature constants ─────────────────────────────────────────────────────────
NUMERICAL_COLS = [
    "loan_amnt", "annual_inc", "credit_score",
    "dti", "loan_to_income", "credit_age_years",
]

EXPECTED_FEATURE_COLS = [
    "loan_amnt", "term", "annual_inc", "emp_length",
    "fico_range_low", "fico_range_high", "dti",
    "inq_last_6mths", "delinq_2yrs", "acc_now_delinq",
    "collections_12_mths_ex_med", "chargeoff_within_12_mths", "pub_rec",
    "credit_age_years", "credit_score", "loan_to_income",
    "home_ownership_MORTGAGE", "home_ownership_NONE", "home_ownership_OTHER",
    "home_ownership_OWN", "home_ownership_RENT",
    "application_type_Joint App",
    "purpose_group_Business", "purpose_group_Debt",
    "purpose_group_Other", "purpose_group_Personal",
]

PURPOSE_GROUP_MAP = {
    "debt_consolidation": "Debt",   "credit_card":      "Debt",
    "home_improvement":   "Asset",  "car":              "Asset",
    "house":              "Asset",  "major_purchase":   "Asset",
    "renewable_energy":   "Asset",  "small_business":   "Business",
    "medical":            "Personal", "vacation":       "Personal",
    "moving":             "Personal", "wedding":        "Personal",
    "educational":        "Personal", "other":          "Other",
}

EMP_MAP = {
    "< 1 year": 0,
    "1 year": 1, "2 years": 2, "3 years": 3,
    "4 years": 4, "5 years": 5, "6 years": 6,
    "7 years": 7, "8 years": 8, "9 years": 9,
    "10+ years": 10,
}

COMPLETED_STATUS = [
    "Fully Paid", "Charged Off", "Default",
    "Does not meet the credit policy. Status:Fully Paid",
    "Does not meet the credit policy. Status:Charged Off",
]

DEFAULT_STATUS = [
    "Charged Off", "Default",
    "Does not meet the credit policy. Status:Charged Off",
]


# ── Helpers ───────────────────────────────────────────────────────────────────
def evaluate(name: str, y_true, y_proba, threshold: float = 0.5) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    metrics = {
        "auc_roc":   round(float(roc_auc_score(y_true, y_proba)), 4),
        "f1":        round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "brier":     round(float(brier_score_loss(y_true, y_proba)), 4),
    }
    print(f"\n{'─'*50}")
    print(f"  {name}")
    for k, v in metrics.items():
        print(f"    {k:<12}: {v}")
    print(classification_report(y_true, y_pred, target_names=["No Default", "Default"]))
    return metrics


def cross_validate(name: str, estimator, X, y, cv: int = 5) -> dict:
    kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=SEED)
    auc_scores = cross_val_score(estimator, X, y, cv=kf, scoring="roc_auc", n_jobs=-1)
    f1_scores  = cross_val_score(estimator, X, y, cv=kf, scoring="f1",      n_jobs=-1)
    result = {
        "cv_auc_mean": round(float(auc_scores.mean()), 4),
        "cv_auc_std":  round(float(auc_scores.std()),  4),
        "cv_f1_mean":  round(float(f1_scores.mean()),  4),
        "cv_f1_std":   round(float(f1_scores.std()),   4),
    }
    print(f"  {name} CV AUC: {result['cv_auc_mean']:.4f} ± {result['cv_auc_std']:.4f}  "
          f"| F1: {result['cv_f1_mean']:.4f} ± {result['cv_f1_std']:.4f}")
    return result


def shap_analysis(model, X_sample: pd.DataFrame, model_name: str) -> dict:
    """Compute SHAP values, save plots, return global importance dict."""
    print(f"\n  Running SHAP for {model_name} …")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # For binary classifiers that return a list (e.g. RandomForest in some shap versions)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # Global importance bar chart
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    plt.title(f"{model_name} — Global Feature Importance (SHAP)")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / f"{model_name}_shap_bar.png", dpi=150)
    plt.close()

    # Beeswarm chart
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title(f"{model_name} — SHAP Beeswarm")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / f"{model_name}_shap_beeswarm.png", dpi=150)
    plt.close()

    # Serialisable global importance dict
    mean_abs = np.abs(shap_values).mean(axis=0)
    global_imp = {
        feat: round(float(v), 6)
        for feat, v in zip(X_sample.columns, mean_abs)
    }
    return global_imp


# ── Main pipeline ─────────────────────────────────────────────────────────────
def main(raw_csv: str):
    # 1. Load & filter ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("STEP 1 — Data Ingestion")
    print(f"{'='*60}")
    print(f"Loading: {raw_csv}")
    df = pd.read_csv(raw_csv, low_memory=False)
    print(f"  Raw shape: {df.shape}")

    df = df[df["loan_status"].isin(COMPLETED_STATUS)].copy()
    df["default"] = df["loan_status"].isin(DEFAULT_STATUS).astype(int)
    print(f"  After filter: {df.shape}  |  default rate: {df['default'].mean():.3f}")

    # 2. Select raw features ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("STEP 2 — Feature Engineering")
    print(f"{'='*60}")
    RAW_KEEP = [
        "loan_amnt", "term", "annual_inc", "emp_length",
        "home_ownership", "purpose", "application_type",
        "fico_range_low", "fico_range_high", "dti",
        "inq_last_6mths", "delinq_2yrs", "acc_now_delinq",
        "collections_12_mths_ex_med", "chargeoff_within_12_mths",
        "pub_rec", "earliest_cr_line", "default",
    ]
    df = df[RAW_KEEP].dropna(
        subset=["loan_amnt", "annual_inc", "dti", "emp_length",
                "fico_range_low", "fico_range_high", "earliest_cr_line"]
    ).copy()

    # term: " 36 months" → 36
    df["term"] = df["term"].str.strip().str.replace(" months", "").astype(int)

    # emp_length string → int
    df["emp_length"] = df["emp_length"].map(EMP_MAP)
    df = df.dropna(subset=["emp_length"])
    df["emp_length"] = df["emp_length"].astype(int)

    # credit history age (anchored to 2024-01-01 for reproducibility)
    df["earliest_cr_line"] = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y", errors="coerce")
    df = df.dropna(subset=["earliest_cr_line"])
    reference_date = pd.Timestamp("2024-01-01")
    df["credit_age_years"] = (reference_date - df["earliest_cr_line"]).dt.days / 365.25

    # derived features
    df["credit_score"]   = (df["fico_range_low"] + df["fico_range_high"]) / 2
    df["annual_inc"]     = df["annual_inc"].replace(0, np.nan)
    df = df.dropna(subset=["annual_inc"])
    df["loan_to_income"] = df["loan_amnt"] / df["annual_inc"]

    # rare event columns — fill 0
    for col in ["acc_now_delinq", "collections_12_mths_ex_med",
                "chargeoff_within_12_mths", "pub_rec"]:
        df[col] = df[col].fillna(0)

    # purpose group mapping
    df["purpose_group"] = df["purpose"].map(PURPOSE_GROUP_MAP).fillna("Other")

    # 3. Encode & build feature matrix ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print("STEP 3 — Preprocessing")
    print(f"{'='*60}")
    df = pd.get_dummies(df, columns=["home_ownership", "application_type", "purpose_group"],
                        drop_first=True)

    for col in EXPECTED_FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0

    X = df[EXPECTED_FEATURE_COLS]
    y = df["default"]
    print(f"  Feature matrix: {X.shape}")
    print(f"  Class balance  — 0: {(y==0).sum():,}   1: {(y==1).sum():,}")

    # Train / test split (stratified 80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    # Scale numerical features
    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test  = X_test.copy()
    X_train[NUMERICAL_COLS] = scaler.fit_transform(X_train[NUMERICAL_COLS])
    X_test[NUMERICAL_COLS]  = scaler.transform(X_test[NUMERICAL_COLS])

    print(f"  Train: {X_train.shape}  |  Test: {X_test.shape}")

    # 4. Train models ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("STEP 4 — Model Training")
    print(f"{'='*60}")

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pw = neg / pos
    print(f"  Class imbalance ratio (scale_pos_weight): {scale_pw:.2f}")

    # XGBoost
    print("\n  [1/3] Training XGBoost …")
    xgb = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pw,
        eval_metric="logloss", random_state=SEED, n_jobs=-1,
    )
    xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100)

    # LightGBM
    print("\n  [2/3] Training LightGBM …")
    lgbm = LGBMClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        is_unbalance=True, random_state=SEED, n_jobs=-1,
        verbose=-1,
    )
    lgbm.fit(X_train, y_train,
             eval_set=[(X_test, y_test)],
             callbacks=[])

    # Random Forest
    print("\n  [3/3] Training Random Forest …")
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=10,
        class_weight="balanced",
        random_state=SEED, n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    # 5. Evaluate ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("STEP 5 — Evaluation")
    print(f"{'='*60}")

    xgb_proba  = xgb.predict_proba(X_test)[:, 1]
    lgbm_proba = lgbm.predict_proba(X_test)[:, 1]
    rf_proba   = rf.predict_proba(X_test)[:, 1]
    ens_proba  = (xgb_proba + lgbm_proba + rf_proba) / 3   # soft ensemble

    model_metrics = {}
    model_metrics["xgboost"]       = evaluate("XGBoost",       y_test, xgb_proba)
    model_metrics["lightgbm"]      = evaluate("LightGBM",      y_test, lgbm_proba)
    model_metrics["random_forest"] = evaluate("Random Forest",  y_test, rf_proba)
    model_metrics["ensemble"]      = evaluate("Soft Ensemble",  y_test, ens_proba)

    # Cross-validation on training set (resource-intensive; use small cv=3 on full data)
    print(f"\n  Cross-Validation (3-fold on train set):")
    for name, est in [("XGBoost", xgb), ("LightGBM", lgbm), ("Random Forest", rf)]:
        cv_res = cross_validate(name, est, X_train, y_train, cv=3)
        model_metrics[name.lower().replace(" ", "_")].update(cv_res)

    # Save metrics JSON
    metrics_path = MODELS_DIR / "model_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(model_metrics, f, indent=2)
    print(f"\n  Metrics saved → {metrics_path}")

    # 6. SHAP analysis ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("STEP 6 — SHAP Explainability")
    print(f"{'='*60}")

    # Sample 2000 rows for SHAP (full dataset is too slow)
    X_shap = X_test.sample(n=min(2000, len(X_test)), random_state=SEED)

    shap_global = {}
    shap_global["xgboost"]       = shap_analysis(xgb,  X_shap, "xgboost")
    shap_global["lightgbm"]      = shap_analysis(lgbm, X_shap, "lightgbm")
    shap_global["random_forest"] = shap_analysis(rf,   X_shap, "random_forest")

    shap_path = MODELS_DIR / "shap_global.json"
    with open(shap_path, "w") as f:
        json.dump(shap_global, f, indent=2)
    print(f"  SHAP global importance saved → {shap_path}")
    print(f"  SHAP plots saved → {REPORTS_DIR}/")

    # 7. Save artefacts ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("STEP 7 — Saving Artefacts")
    print(f"{'='*60}")

    joblib.dump(xgb,                    MODELS_DIR / "xgb_model.pkl")
    joblib.dump(lgbm,                   MODELS_DIR / "lgbm_model.pkl")
    joblib.dump(rf,                     MODELS_DIR / "rf_model.pkl")
    joblib.dump(scaler,                 MODELS_DIR / "scaler.pkl")
    joblib.dump(EXPECTED_FEATURE_COLS,  MODELS_DIR / "feature_names.pkl")
    joblib.dump(NUMERICAL_COLS,         MODELS_DIR / "numerical_cols.pkl")

    saved = [
        "xgb_model.pkl", "lgbm_model.pkl", "rf_model.pkl",
        "scaler.pkl", "feature_names.pkl", "numerical_cols.pkl",
        "model_metrics.json", "shap_global.json",
    ]
    for name in saved:
        print(f"  ✓  {MODELS_DIR / name}")

    print(f"\n{'='*60}")
    print("Pipeline complete!")
    best = max(
        ["xgboost", "lightgbm", "random_forest", "ensemble"],
        key=lambda k: model_metrics[k]["auc_roc"]
    )
    print(f"  Best model by AUC-ROC: {best.upper()}  "
          f"({model_metrics[best]['auc_roc']:.4f})")
    print(f"{'='*60}\n")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Credit risk model training pipeline")
    parser.add_argument(
        "--data", default=DEFAULT_CSV,
        help="Path to the raw LendingClub accepted-loans CSV"
    )
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"ERROR: Data file not found: {args.data}")
        print("Update --data or edit DEFAULT_CSV in this script.")
        sys.exit(1)

    main(args.data)
