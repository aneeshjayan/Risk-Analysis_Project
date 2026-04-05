"""
XGBoost baseline model for Probability of Default prediction.
Owner: Aneesh Jayan Prabhu
"""

from xgboost import XGBClassifier
from .base_model import BaseCreditModel


class XGBoostModel(BaseCreditModel):

    def __init__(self, n_estimators=300, max_depth=6, learning_rate=0.05,
                 scale_pos_weight=10, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.scale_pos_weight = scale_pos_weight
        self.random_state = random_state
        self.model = None
        self.model_name = "xgboost"

    def build(self) -> None:
        self.model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            scale_pos_weight=self.scale_pos_weight,
            eval_metric="auc",
            random_state=self.random_state,
            n_jobs=-1,
        )
