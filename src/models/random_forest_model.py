"""
Random Forest model for Probability of Default prediction.
Owner: Kalaivani Ravichandran
"""

from sklearn.ensemble import RandomForestClassifier
from .base_model import BaseCreditModel


class RandomForestModel(BaseCreditModel):

    def __init__(self, config: dict):
        super().__init__(config)
        self.model_name = "random_forest"

    def build(self) -> None:
        params = self.config["models"]["random_forest"]
        self.model = RandomForestClassifier(
            n_estimators=params.get("n_estimators", 200),
            max_depth=params.get("max_depth", 10),
            class_weight=params.get("class_weight", "balanced"),
            random_state=params.get("random_state", 42),
            n_jobs=-1,
        )
