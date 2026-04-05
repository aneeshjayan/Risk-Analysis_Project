"""
XGBoost baseline model for Probability of Default prediction.
Owner: Aneesh Jayan Prabhu
"""

from xgboost import XGBClassifier
from .base_model import BaseCreditModel


class XGBoostModel(BaseCreditModel):

    def __init__(self, config: dict):
        super().__init__(config)
        self.model_name = "xgboost"

    def build(self) -> None:
        params = self.config["models"]["xgboost"]
        self.model = XGBClassifier(
            n_estimators=params.get("n_estimators", 300),
            max_depth=params.get("max_depth", 6),
            learning_rate=params.get("learning_rate", 0.05),
            scale_pos_weight=params.get("scale_pos_weight", 10),
            eval_metric=params.get("eval_metric", "auc"),
            random_state=params.get("random_state", 42),
            n_jobs=-1,
        )
