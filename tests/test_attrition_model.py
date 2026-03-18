"""
Unit tests for the AttritionModel module.

Author: Gabriel Demetrios Lafis
License: MIT
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.attrition_model import AttritionModel
from src.data_loader import load_hr_data


@pytest.fixture
def hr_data() -> pd.DataFrame:
    """Load a small synthetic HR dataset."""
    return load_hr_data(n_employees=200)


@pytest.fixture
def trained_model(hr_data: pd.DataFrame) -> AttritionModel:
    """Return a trained AttritionModel."""
    model = AttritionModel(random_state=42, n_estimators=50)
    model.train(hr_data)
    return model


class TestAttritionModel:
    """Tests for AttritionModel class."""

    def test_init_defaults(self) -> None:
        model = AttritionModel()
        assert model.algorithm == "random_forest"
        assert model.pipeline is None
        assert model.feature_names == []

    def test_train_returns_self(self, hr_data: pd.DataFrame) -> None:
        model = AttritionModel(random_state=42, n_estimators=50)
        result = model.train(hr_data)
        assert result is model

    def test_train_sets_pipeline(
        self, trained_model: AttritionModel
    ) -> None:
        assert trained_model.pipeline is not None
        assert len(trained_model.feature_names) > 0

    def test_predict_proba_shape(
        self, trained_model: AttritionModel, hr_data: pd.DataFrame
    ) -> None:
        probs = trained_model.predict_proba(hr_data)
        assert isinstance(probs, np.ndarray)
        assert len(probs) == len(hr_data)
        assert all(0 <= p <= 1 for p in probs)

    def test_predict_proba_before_train(
        self, hr_data: pd.DataFrame
    ) -> None:
        model = AttritionModel()
        with pytest.raises(RuntimeError, match="Model not trained"):
            model.predict_proba(hr_data)

    def test_evaluate_returns_metrics(
        self, trained_model: AttritionModel, hr_data: pd.DataFrame
    ) -> None:
        metrics = trained_model.evaluate(hr_data)
        assert "roc_auc" in metrics
        assert "classification_report" in metrics
        assert 0 <= metrics["roc_auc"] <= 1

    def test_feature_importance(
        self, trained_model: AttritionModel
    ) -> None:
        fi = trained_model.feature_importance()
        assert isinstance(fi, pd.DataFrame)
        assert "feature" in fi.columns
        assert "importance" in fi.columns
        assert len(fi) == len(trained_model.feature_names)
        # Importances should sum to ~1
        assert abs(fi["importance"].sum() - 1.0) < 0.01

    def test_feature_importance_before_train(self) -> None:
        model = AttritionModel()
        with pytest.raises(RuntimeError, match="Model not trained"):
            model.feature_importance()

    def test_save_and_load(
        self, trained_model: AttritionModel, hr_data: pd.DataFrame
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pkl"
            saved_path = trained_model.save(path)
            assert saved_path.exists()

            loaded = AttritionModel.load(saved_path)
            assert loaded.pipeline is not None

            # Predictions should match
            orig_probs = trained_model.predict_proba(hr_data)
            loaded_probs = loaded.predict_proba(hr_data)
            np.testing.assert_array_almost_equal(orig_probs, loaded_probs)

    def test_encode_handles_categorical(
        self, hr_data: pd.DataFrame
    ) -> None:
        model = AttritionModel()
        encoded = model._encode(hr_data)
        # Categorical columns should be numeric after encoding
        for col in ["Gender", "Department", "OverTime"]:
            if col in encoded.columns:
                assert encoded[col].dtype in [np.int32, np.int64, np.float64]
