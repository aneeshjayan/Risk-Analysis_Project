"""
Abstract base class for all credit risk models.
All models (XGBoost, LightGBM, RandomForest) implement this interface.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import joblib
from pathlib import Path


class BaseCreditModel(ABC):

    def __init__(self):
        self.model = None
        self.model_name: str = "base"

    @abstractmethod
    def build(self) -> None:
        """Instantiate the underlying sklearn-compatible estimator."""
        ...

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        if self.model is None:
            self.build()
        self.model.fit(X_train, y_train)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def save(self, output_dir: str = "data/processed/models") -> str:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        path = f"{output_dir}/{self.model_name}.joblib"
        joblib.dump(self.model, path)
        print(f"Model saved: {path}")
        return path

    def load(self, model_path: str) -> None:
        self.model = joblib.load(model_path)
