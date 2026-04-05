"""
Random Forest model for Probability of Default prediction.
Owner: Kalaivani Ravichandran
"""

from sklearn.ensemble import RandomForestClassifier
from .base_model import BaseCreditModel


class RandomForestModel(BaseCreditModel):

    def __init__(self, n_estimators=200, max_depth=10, class_weight="balanced",
                 random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.class_weight = class_weight
        self.random_state = random_state
        self.model = None
        self.model_name = "random_forest"

    def build(self) -> None:
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=-1,
        )
