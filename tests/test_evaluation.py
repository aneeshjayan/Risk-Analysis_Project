"""Tests for evaluation metrics."""

import numpy as np
from src.evaluation.metrics import compute_metrics, compare_models


def test_compute_metrics_shapes():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    y_proba = np.array([0.1, 0.6, 0.7, 0.9])
    metrics = compute_metrics(y_true, y_pred, y_proba)
    assert set(metrics.keys()) == {"roc_auc", "f1", "precision", "recall", "brier_score"}
    assert all(0.0 <= v <= 1.0 for v in metrics.values())


def test_compare_models_sorted():
    results = {
        "xgb": {"roc_auc": 0.85, "f1": 0.7},
        "lgbm": {"roc_auc": 0.90, "f1": 0.75},
    }
    df = compare_models(results)
    assert df.index[0] == "lgbm"
