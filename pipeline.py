"""
pipeline.py — End-to-end Credit Risk training pipeline.

Two data sources:
  1. LendingClub accepted loans CSV  — trains the default-risk ML models
  2. FRED macroeconomic API          — enriches features with economic context
                                       (auto-fetched; skipped if no API key)

Steps:
  Ingestion → FRED merge (optional) → Feature Engineering → Preprocessing
  → Train 3 models → Pick best by AUC → SHAP → Save artifacts

Usage:
    python pipeline.py
    python pipeline.py --data path/to/lending_club.csv
"""

import argparse
import json
import shutil
import os
import sys
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from dotenv import load_dotenv
from lightgbm import log_evaluation, early_stopping as lgbm_early_stopping

load_dotenv()

# ── src/ imports ───────────────────────────────────────────────────────────────
from src.data.ingestion            import run_ingestion, is_preprocessed
from src.data.feature_engineering  import run_feature_engineering
from src.data.macro_features       import load_or_fetch_macro, merge_macro_features
from src.data.preprocessing        import (
    run_preprocessing, split_data, scale_features, NUMERICAL_COLS,
)
from src.models.xgboost_model      import XGBoostModel
from src.models.lightgbm_model     import LightGBMModel
from src.models.random_forest_model import RandomForestModel
from src.evaluation.metrics        import (
    compute_metrics, print_metrics,
    compare_models, find_optimal_threshold,
)
from src.explainability.shap_analysis import SHAPAnalyzer
from src.utils.helpers             import set_seed, get_logger, ensure_dirs

import numpy as np
import pandas as pd
import joblib

# ── Paths ──────────────────────────────────────────────────────────────────────
DEFAULT_CSV        = r"D:\ASU SEM4\Capstone\Lending_club_data.csv"
PROCESSED_DIR      = Path("data/processed")
MODELS_DIR         = PROCESSED_DIR / "models"
REPORTS_DIR        = Path("reports/figures")
BACKEND_MODELS_DIR = Path("models")   # backend/main.py reads from here

logger = get_logger("pipeline")


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved → {path}")


def main(raw_csv: str, df: pd.DataFrame = None) -> None:
    """
    Full training pipeline: ingestion → FRED → FE → preprocessing
    → train 3 models → pick best → SHAP → save artifacts.

    Args:
        raw_csv: Path to the LendingClub CSV.
        df:      Optional pre-loaded DataFrame — skips re-reading the file.
    """
    set_seed(42)
    ensure_dirs(str(PROCESSED_DIR), str(MODELS_DIR), str(REPORTS_DIR))

    # ── STEP 1 — Ingestion ────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1 — Data Ingestion  (Dataset 1: LendingClub loans)")
    if df is None:
        df = run_ingestion(raw_csv)
    else:
        logger.info(f"  Using pre-loaded DataFrame {df.shape}")

    # ── STEP 1b — FRED Macro merge (Dataset 2: Federal Reserve) ──────────────
    # Fully optional — skipped silently if no API key or no issue_d column.
    fred_api_key = os.getenv("FRED_API_KEY", "")
    if fred_api_key and "issue_d" in df.columns:
        try:
            logger.info("STEP 1b — FRED Macroeconomic Feature Merge  (Dataset 2: FRED)")
            macro_df = load_or_fetch_macro(
                api_key=fred_api_key, start="2007-01-01", end="2019-12-31"
            )
            df = merge_macro_features(df, macro_df, issue_col="issue_d")
            logger.info(f"  FRED features merged — dataset shape: {df.shape}")
        except Exception as e:
            logger.info(f"  FRED skipped ({e}) — continuing without macro features.")
            if "issue_d" in df.columns:
                df = df.drop(columns=["issue_d"])
    else:
        if "issue_d" in df.columns:
            df = df.drop(columns=["issue_d"])
        if not fred_api_key:
            logger.info("  FRED skipped — no FRED_API_KEY in .env")

    # ── STEP 2 — Feature Engineering ─────────────────────────────────────────
    if is_preprocessed(df):
        logger.info("STEP 2 — Feature Engineering: SKIPPED (data already processed)")
        from src.data.feature_engineering import extract_issue_date_features
        df = extract_issue_date_features(df)
    else:
        logger.info("STEP 2 — Feature Engineering")
        df = run_feature_engineering(df)

    # ── STEP 3 — Preprocessing ────────────────────────────────────────────────
    logger.info("STEP 3 — Preprocessing")
    already_encoded = "home_ownership_MORTGAGE" in df.columns

    if already_encoded:
        logger.info("  Data already one-hot encoded — running split + scale only.")
        X_train, X_test, y_train, y_test = split_data(df)

        num_cols_present = [c for c in NUMERICAL_COLS if c in X_train.columns]
        sample_std = X_train[num_cols_present].std().mean()
        if sample_std < 5.0:
            logger.info(
                f"  Columns appear already scaled (avg std={sample_std:.2f}) "
                "— skipping StandardScaler."
            )
        else:
            logger.info(f"  Scaling numerical columns (avg std={sample_std:.2f}) …")
            X_train, X_test = scale_features(
                X_train, X_test,
                numerical_cols=NUMERICAL_COLS,
                scaler_path=str(PROCESSED_DIR / "scaler.joblib"),
            )
        logger.info(f"  Train: {X_train.shape} | Test: {X_test.shape}")
    else:
        X_train, X_test, y_train, y_test = run_preprocessing(
            df, scaler_path=str(PROCESSED_DIR / "scaler.joblib")
        )

    # ── STEP 4 — Model Training ───────────────────────────────────────────────
    logger.info("STEP 4 — Model Training  (XGBoost / LightGBM / Random Forest)")

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    logger.info(f"  Class distribution — 0 (paid):{neg:,}  1 (default):{pos:,}  ratio:{neg/pos:.1f}:1")
    logger.info("  Imbalance handled via class weights in each model + optimal threshold tuning.")

    # XGBoost ──────────────────────────────────────────────────────────────────
    logger.info("  [1/3] Training XGBoost …")
    xgb = XGBoostModel.from_class_ratio(
        y_train,
        n_estimators          = 1000,
        max_depth             = 6,
        min_child_weight      = 3.0,
        gamma                 = 0.05,
        learning_rate         = 0.02,
        early_stopping_rounds = 50,   # XGBoost 2.x: must be in constructor
        subsample             = 0.85,
        colsample_bytree      = 0.8,
        colsample_bylevel     = 0.9,
        reg_alpha             = 0.05,
        reg_lambda            = 1.5,
    )
    xgb.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=100,
    )
    logger.info(f"  XGBoost best iteration: {xgb.model.best_iteration}")

    # LightGBM ─────────────────────────────────────────────────────────────────
    logger.info("  [2/3] Training LightGBM …")
    lgbm = LightGBMModel(
        n_estimators      = 1000,
        max_depth         = 8,
        num_leaves        = 63,
        min_child_samples = 30,
        learning_rate     = 0.02,
        subsample         = 0.85,
        subsample_freq    = 1,
        colsample_bytree  = 0.8,
        reg_alpha         = 0.05,
        reg_lambda        = 1.0,
        is_unbalance      = True,
    )
    lgbm.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[
            lgbm_early_stopping(stopping_rounds=50, verbose=True),
            log_evaluation(period=100),
        ],
    )

    # Random Forest ────────────────────────────────────────────────────────────
    logger.info("  [3/3] Training Random Forest …")
    rf = RandomForestModel(
        n_estimators      = 400,
        max_depth         = 14,
        min_samples_split = 8,
        min_samples_leaf  = 3,
        max_features      = "sqrt",
        max_samples       = 0.8,
        class_weight      = "balanced",
    )
    rf.fit(X_train, y_train)

    # ── STEP 5 — Evaluation ───────────────────────────────────────────────────
    logger.info("STEP 5 — Evaluation")

    xgb_proba  = xgb.predict_proba(X_test)
    lgbm_proba = lgbm.predict_proba(X_test)
    rf_proba   = rf.predict_proba(X_test)

    all_metrics: dict[str, dict] = {}
    for name, proba in [
        ("xgboost",       xgb_proba),
        ("lightgbm",      lgbm_proba),
        ("random_forest", rf_proba),
    ]:
        thresh = find_optimal_threshold(y_test.values, proba)
        pred   = (proba >= thresh).astype(int)
        m      = compute_metrics(y_test.values, pred, proba)
        m["optimal_threshold"] = round(float(thresh), 4)
        print_metrics(m, model_name=name)
        all_metrics[name] = {k: round(float(v), 4) for k, v in m.items()}

    save_json(all_metrics, PROCESSED_DIR / "model_metrics.json")

    # Pick best model by AUC-ROC ───────────────────────────────────────────────
    MODEL_OBJ_MAP = {"xgboost": xgb, "lightgbm": lgbm, "random_forest": rf}
    best_name = max(MODEL_OBJ_MAP.keys(), key=lambda k: all_metrics[k]["roc_auc"])

    comp = compare_models(all_metrics)
    logger.info(f"\n  Model Ranking:\n{comp[['roc_auc', 'f1', 'brier_score']].to_string()}")
    logger.info(
        f"\n  ★ Best model: {best_name.upper()}  "
        f"AUC={all_metrics[best_name]['roc_auc']:.4f}  "
        f"F1={all_metrics[best_name]['f1']:.4f}"
    )
    logger.info("  This model will be used for all production predictions + SHAP explanations.")

    # ── STEP 6 — Save best model weights FIRST ───────────────────────────────
    # Save before SHAP so the backend is ready even if SHAP is slow.
    logger.info("STEP 6 — Saving best model weights")
    BACKEND_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    best_model_obj = MODEL_OBJ_MAP[best_name]
    joblib.dump(xgb.model,  BACKEND_MODELS_DIR / "xgb_model.pkl")
    joblib.dump(lgbm.model, BACKEND_MODELS_DIR / "lgbm_model.pkl")
    joblib.dump(rf.model,   BACKEND_MODELS_DIR / "rf_model.pkl")
    joblib.dump(best_name,  BACKEND_MODELS_DIR / "best_model_name.pkl")
    joblib.dump(list(X_train.columns), BACKEND_MODELS_DIR / "feature_names.pkl")
    joblib.dump(NUMERICAL_COLS,        BACKEND_MODELS_DIR / "numerical_cols.pkl")

    scaler_src = PROCESSED_DIR / "scaler.joblib"
    scaler_dst = BACKEND_MODELS_DIR / "scaler.pkl"
    if scaler_src.exists():
        shutil.copy2(scaler_src, scaler_dst)
    else:
        from sklearn.preprocessing import StandardScaler
        _scaler = StandardScaler()
        num_present = [c for c in NUMERICAL_COLS if c in X_train.columns]
        _scaler.fit(X_train[num_present])
        joblib.dump(_scaler, scaler_dst)

    shutil.copy2(PROCESSED_DIR / "model_metrics.json",
                 BACKEND_MODELS_DIR / "model_metrics.json")
    logger.info(f"  Models + scaler + metrics saved → {BACKEND_MODELS_DIR.resolve()}")
    logger.info(f"  Best model weights: {best_name.upper()} → best_model_name.pkl")

    # ── STEP 7 — SHAP (best model only — fast) ───────────────────────────────
    # Only the winner gets SHAP. Skips slow RF/LGBM SHAP entirely.
    logger.info(f"STEP 7 — SHAP Analysis  (best model only: {best_name.upper()})")
    X_shap = X_test.sample(n=min(1000, len(X_test)), random_state=42)

    analyzer = SHAPAnalyzer(best_model_obj.model, model_name=best_name)
    analyzer.compute_shap_values(X_shap)
    analyzer.plot_global_importance(str(REPORTS_DIR))
    analyzer.plot_beeswarm(str(REPORTS_DIR))

    shap_global = {
        best_name:      analyzer.global_importance_dict(),
        "_best_model":  best_name,
    }
    save_json(shap_global, PROCESSED_DIR / "shap_global.json")
    shutil.copy2(PROCESSED_DIR / "shap_global.json",
                 BACKEND_MODELS_DIR / "shap_global.json")
    logger.info(f"  SHAP saved → shap_global.json  ({len(X_shap)} sample rows)")

    # ── STEP 8 — Save pipeline-internal model store ──────────────────────────
    logger.info("STEP 8 — Saving models to pipeline store")
    xgb.save(str(MODELS_DIR))
    lgbm.save(str(MODELS_DIR))
    rf.save(str(MODELS_DIR))

    logger.info("=" * 60)
    logger.info(
        f"  PIPELINE COMPLETE!  Best model: {best_name.upper()}  "
        f"AUC={all_metrics[best_name]['roc_auc']:.4f}  "
        f"F1={all_metrics[best_name]['f1']:.4f}"
    )
    logger.info(f"  All artifacts ready in: {BACKEND_MODELS_DIR.resolve()}")
    logger.info("=" * 60)


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoanLens — Credit Risk Training Pipeline")
    parser.add_argument(
        "--data", default=DEFAULT_CSV,
        help="Path to the LendingClub loans CSV",
    )
    args = parser.parse_args()

    if not os.path.exists(args.data):
        logger.error(f"Data file not found: {args.data}")
        logger.error("Update --data or edit DEFAULT_CSV at the top of this script.")
        sys.exit(1)

    # Load CSV once and pass into main() — avoids reading 1.1M rows twice
    df = run_ingestion(args.data)
    main(args.data, df=df)
