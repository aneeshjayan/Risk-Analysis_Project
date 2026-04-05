"""Smoke tests: models build, fit, and predict without errors."""

import pandas as pd
import numpy as np
import pytest
from src.models.xgboost_model import XGBoostModel
from src.models.lightgbm_model import LightGBMModel
from src.models.random_forest_model import RandomForestModel

_CONFIG = {
    "models": {
        "xgboost": {"n_estimators": 10, "max_depth": 3, "learning_rate": 0.1,
                    "scale_pos_weight": 1, "eval_metric": "auc", "random_state": 42},
        "lightgbm": {"n_estimators": 10, "max_depth": 3, "learning_rate": 0.1,
                     "is_unbalance": False, "random_state": 42},
        "random_forest": {"n_estimators": 10, "max_depth": 3,
                          "class_weight": "balanced", "random_state": 42},
    }
}

np.random.seed(42)
_X = pd.DataFrame(np.random.randn(100, 5), columns=[f"f{i}" for i in range(5)])
_y = pd.Series(np.random.randint(0, 2, 100))


@pytest.mark.parametrize("ModelClass", [XGBoostModel, LightGBMModel, RandomForestModel])
def test_model_fit_predict(ModelClass):
    model = ModelClass(_CONFIG)
    model.fit(_X, _y)
    proba = model.predict_proba(_X)
    assert proba.shape == (100,)
    assert 0.0 <= proba.min() and proba.max() <= 1.0
