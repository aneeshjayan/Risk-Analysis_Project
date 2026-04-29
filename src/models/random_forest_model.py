"""
Random Forest model for Probability of Default prediction.
Owner: Kalaivani Ravichandran
"""

from sklearn.ensemble import RandomForestClassifier
from .base_model import BaseCreditModel


class RandomForestModel(BaseCreditModel):

    def __init__(
        self,
        # Tree structure
        n_estimators:    int         = 200,
        max_depth:       int | None  = 10,      # None = expand until pure leaves
        min_samples_split: int       = 2,       # min samples to split a node
        min_samples_leaf:  int       = 1,       # min samples required at a leaf
        max_features:    str | float = "sqrt",  # features per split: "sqrt","log2", float
        max_samples:     float | None = None,   # row sampling fraction (None = all)
        # Class imbalance
        class_weight:    str | dict  = "balanced",
        # Misc
        random_state:    int         = 42,
    ):
        self.n_estimators      = n_estimators
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf  = min_samples_leaf
        self.max_features      = max_features
        self.max_samples       = max_samples
        self.class_weight      = class_weight
        self.random_state      = random_state
        self.model             = None
        self.model_name        = "random_forest"

    def build(self) -> None:
        self.model = RandomForestClassifier(
            n_estimators      = self.n_estimators,
            max_depth         = self.max_depth,
            min_samples_split = self.min_samples_split,
            min_samples_leaf  = self.min_samples_leaf,
            max_features      = self.max_features,
            max_samples       = self.max_samples,
            class_weight      = self.class_weight,
            random_state      = self.random_state,
            n_jobs            = -1,
        )
