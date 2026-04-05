"""
Model evaluation metrics for credit risk models.
Computes AUC-ROC, F1, Precision, Recall, Brier Score, and cross-validation.
Owner: Jayasurya Sakthivel
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    brier_score_loss,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    """Return a flat dict of all evaluation metrics."""
    return {
        "roc_auc": roc_auc_score(y_true, y_proba),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "brier_score": brier_score_loss(y_true, y_proba),
    }


def print_metrics(metrics: dict, model_name: str = "") -> None:
    header = f"=== {model_name} ===" if model_name else "=== Metrics ==="
    print(header)
    for k, v in metrics.items():
        print(f"  {k:>15}: {v:.4f}")


def cross_validate_model(model, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> dict:
    """Stratified K-Fold cross-validation. Returns mean/std for AUC and F1."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    auc_scores = cross_val_score(model.model, X, y, cv=skf, scoring="roc_auc", n_jobs=-1)
    f1_scores = cross_val_score(model.model, X, y, cv=skf, scoring="f1", n_jobs=-1)
    return {
        "cv_auc_mean": auc_scores.mean(),
        "cv_auc_std": auc_scores.std(),
        "cv_f1_mean": f1_scores.mean(),
        "cv_f1_std": f1_scores.std(),
    }


def compare_models(results: dict[str, dict]) -> pd.DataFrame:
    """Build a comparison DataFrame from a {model_name: metrics_dict} mapping."""
    return pd.DataFrame(results).T.sort_values("roc_auc", ascending=False)


def find_optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Find threshold that maximises F1 score on the given set."""
    thresholds = np.linspace(0.1, 0.9, 81)
    best_thresh, best_f1 = 0.5, 0.0
    for t in thresholds:
        f1 = f1_score(y_true, (y_proba >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t
    return best_thresh
