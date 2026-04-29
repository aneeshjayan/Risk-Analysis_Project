"""
LightGBM model for Probability of Default prediction.
Owner: Balamadhan Sivaraman
"""

from lightgbm import LGBMClassifier
from .base_model import BaseCreditModel


class LightGBMModel(BaseCreditModel):

    def __init__(
        self,
        # Tree structure
        n_estimators:      int   = 300,
        max_depth:         int   = 6,
        num_leaves:        int   = 63,     # max leaves per tree; must be < 2^max_depth
        min_child_samples: int   = 20,     # min data in a leaf; higher → less overfitting
        # Learning
        learning_rate:     float = 0.05,
        # Sampling
        subsample:         float = 0.8,    # row sampling (bagging_fraction)
        subsample_freq:    int   = 1,      # perform bagging every k iterations
        colsample_bytree:  float = 0.8,    # feature sampling (feature_fraction)
        # L1 / L2 regularisation
        reg_alpha:         float = 0.0,    # L1
        reg_lambda:        float = 0.0,    # L2
        # Class imbalance
        is_unbalance:      bool  = True,   # auto-reweights minority class
        # Misc
        random_state:      int   = 42,
    ):
        self.n_estimators      = n_estimators
        self.max_depth         = max_depth
        self.num_leaves        = num_leaves
        self.min_child_samples = min_child_samples
        self.learning_rate     = learning_rate
        self.subsample         = subsample
        self.subsample_freq    = subsample_freq
        self.colsample_bytree  = colsample_bytree
        self.reg_alpha         = reg_alpha
        self.reg_lambda        = reg_lambda
        self.is_unbalance      = is_unbalance
        self.random_state      = random_state
        self.model             = None
        self.model_name        = "lightgbm"

    def build(self) -> None:
        self.model = LGBMClassifier(
            n_estimators      = self.n_estimators,
            max_depth         = self.max_depth,
            num_leaves        = self.num_leaves,
            min_child_samples = self.min_child_samples,
            learning_rate     = self.learning_rate,
            subsample         = self.subsample,
            subsample_freq    = self.subsample_freq,
            colsample_bytree  = self.colsample_bytree,
            reg_alpha         = self.reg_alpha,
            reg_lambda        = self.reg_lambda,
            is_unbalance      = self.is_unbalance,
            random_state      = self.random_state,
            n_jobs            = -1,
            verbose           = -1,
        )
