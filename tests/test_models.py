"""Smoke tests: models build, fit, and predict without errors."""

import pandas as pd
import numpy as np
import pytest
from src.models.xgboost_model import XGBoostModel
from src.models.lightgbm_model import LightGBMModel
from src.models.random_forest_model import RandomForestModel

np.random.seed(42)
_X = pd.DataFrame(np.random.randn(100, 5), columns=[f"f{i}" for i in range(5)])
_y = pd.Series(np.random.randint(0, 2, 100))


@pytest.mark.parametrize("ModelClass", [XGBoostModel, LightGBMModel, RandomForestModel])
def test_model_fit_predict(ModelClass):
    model = ModelClass(n_estimators=10, max_depth=3, random_state=42)
    model.fit(_X, _y)
    proba = model.predict_proba(_X)
    assert proba.shape == (100,)
    assert 0.0 <= proba.min() and proba.max() <= 1.0
