"""
LightGBM model for Probability of Default prediction.
Owner: Balamadhan Sivaraman
"""

from lightgbm import LGBMClassifier
from .base_model import BaseCreditModel


class LightGBMModel(BaseCreditModel):

    def __init__(self, config: dict):
        super().__init__(config)
        self.model_name = "lightgbm"

    def build(self) -> None:
        params = self.config["models"]["lightgbm"]
        self.model = LGBMClassifier(
            n_estimators=params.get("n_estimators", 300),
            max_depth=params.get("max_depth", 6),
            learning_rate=params.get("learning_rate", 0.05),
            is_unbalance=params.get("is_unbalance", True),
            random_state=params.get("random_state", 42),
            n_jobs=-1,
            verbose=-1,
        )
