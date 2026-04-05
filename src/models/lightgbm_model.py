"""
LightGBM model for Probability of Default prediction.
Owner: Balamadhan Sivaraman
"""

from lightgbm import LGBMClassifier
from .base_model import BaseCreditModel


class LightGBMModel(BaseCreditModel):

    def __init__(self, n_estimators=300, max_depth=6, learning_rate=0.05,
                 is_unbalance=True, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.is_unbalance = is_unbalance
        self.random_state = random_state
        self.model = None
        self.model_name = "lightgbm"

    def build(self) -> None:
        self.model = LGBMClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            is_unbalance=self.is_unbalance,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1,
        )
