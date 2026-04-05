"""
SHAP-based explainability for credit risk models.
Provides global feature importance and local (per-prediction) explanations.
Owner: Subramanian Raj Narayanan
"""

import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


class SHAPAnalyzer:

    def __init__(self, model, X_train: pd.DataFrame, model_name: str = "model"):
        self.model = model
        self.model_name = model_name
        self.explainer = None
        self.shap_values = None
        self._fit_explainer(X_train)

    def _fit_explainer(self, X_train: pd.DataFrame) -> None:
        """Initialise TreeExplainer (works for XGBoost, LightGBM, RF)."""
        self.explainer = shap.TreeExplainer(self.model)

    def compute_shap_values(self, X: pd.DataFrame, sample_size: int = 1000) -> np.ndarray:
        """Compute SHAP values on a (sampled) dataset."""
        if len(X) > sample_size:
            X = X.sample(n=sample_size, random_state=42)
        self.shap_values = self.explainer.shap_values(X)
        self._shap_X = X
        return self.shap_values

    def plot_global_importance(self, output_dir: str = "reports/figures") -> None:
        """Bar plot of mean |SHAP| per feature — global importance."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        shap.summary_plot(
            self.shap_values, self._shap_X, plot_type="bar", show=False
        )
        plt.title(f"{self.model_name} — Global Feature Importance (SHAP)")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{self.model_name}_shap_global.png", dpi=150)
        plt.close()

    def plot_beeswarm(self, output_dir: str = "reports/figures") -> None:
        """Beeswarm plot showing feature value direction of impact."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        shap.summary_plot(self.shap_values, self._shap_X, show=False)
        plt.title(f"{self.model_name} — SHAP Beeswarm")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{self.model_name}_shap_beeswarm.png", dpi=150)
        plt.close()

    def explain_single_prediction(self, X_row: pd.DataFrame) -> dict:
        """Return a {feature: shap_value} dict for one borrower row."""
        sv = self.explainer.shap_values(X_row)
        if isinstance(sv, list):
            sv = sv[1]          # binary classifier: take positive class
        return dict(zip(X_row.columns, sv.flatten()))

    def top_features(self, n: int = 10) -> pd.DataFrame:
        """Return top-n features by mean absolute SHAP value."""
        mean_shap = np.abs(self.shap_values).mean(axis=0)
        return (
            pd.DataFrame({"feature": self._shap_X.columns, "mean_shap": mean_shap})
            .sort_values("mean_shap", ascending=False)
            .head(n)
            .reset_index(drop=True)
        )
