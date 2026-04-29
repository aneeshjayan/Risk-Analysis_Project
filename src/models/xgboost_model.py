"""
XGBoost model for Probability of Default prediction.
Owner: Aneesh Jayan Prabhu
"""

from xgboost import XGBClassifier
from .base_model import BaseCreditModel


class XGBoostModel(BaseCreditModel):

    def __init__(
        self,
        # Tree structure
        n_estimators:          int   = 300,
        max_depth:             int   = 4,
        min_child_weight:      float = 1.0,
        gamma:                 float = 0.0,
        # Learning
        learning_rate:         float = 0.05,
        # Early stopping (XGBoost 2.x: must be set in constructor, not fit())
        early_stopping_rounds: int   = 50,
        # Sampling
        subsample:             float = 0.8,
        colsample_bytree:      float = 0.8,
        colsample_bylevel:     float = 1.0,
        # L1 / L2 regularisation
        reg_alpha:             float = 0.0,
        reg_lambda:            float = 1.0,
        # Class imbalance
        scale_pos_weight:      float = 1.0,
        # Misc
        random_state:          int   = 42,
    ):
        self.n_estimators          = n_estimators
        self.max_depth             = max_depth
        self.min_child_weight      = min_child_weight
        self.gamma                 = gamma
        self.learning_rate         = learning_rate
        self.early_stopping_rounds = early_stopping_rounds
        self.subsample             = subsample
        self.colsample_bytree      = colsample_bytree
        self.colsample_bylevel     = colsample_bylevel
        self.reg_alpha             = reg_alpha
        self.reg_lambda            = reg_lambda
        self.scale_pos_weight      = scale_pos_weight
        self.random_state          = random_state
        self.model                 = None
        self.model_name            = "xgboost"

    @classmethod
    def from_class_ratio(cls, y_train, **kwargs) -> "XGBoostModel":
        """Compute scale_pos_weight from label distribution and pass remaining kwargs."""
        neg = (y_train == 0).sum()
        pos = (y_train == 1).sum()
        return cls(scale_pos_weight=neg / pos, **kwargs)

    def build(self) -> None:
        self.model = XGBClassifier(
            n_estimators          = self.n_estimators,
            max_depth             = self.max_depth,
            min_child_weight      = self.min_child_weight,
            gamma                 = self.gamma,
            learning_rate         = self.learning_rate,
            early_stopping_rounds = self.early_stopping_rounds,  # XGBoost 2.x: constructor
            subsample             = self.subsample,
            colsample_bytree      = self.colsample_bytree,
            colsample_bylevel     = self.colsample_bylevel,
            reg_alpha             = self.reg_alpha,
            reg_lambda            = self.reg_lambda,
            scale_pos_weight      = self.scale_pos_weight,
            eval_metric           = "logloss",
            random_state          = self.random_state,
            n_jobs                = -1,
        )
