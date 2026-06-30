"""
Expanded unit tests for XGBoostBaseline model.
Covers: initialization, training, evaluation metrics, predict_proba shape,
feature importance availability, and model serialization / reload.
"""

import os
import tempfile
import pytest
import numpy as np
import pandas as pd

try:
    import xgboost as xgb

    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from src.models.xgboost_baseline import XGBoostBaseline

pytestmark = pytest.mark.skipif(not HAS_XGB, reason="xgboost not installed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dataset(n=200, n_features=10, seed=42):
    """Return a small labelled DataFrame for fast tests."""
    np.random.seed(seed)
    X = pd.DataFrame(
        np.random.randn(n, n_features),
        columns=[f"feat{i}" for i in range(n_features)],
    )
    # Introduce a learnable signal so AUC > random
    y = np.where(X["feat0"] + 0.5 * X["feat1"] > 1.0, 1, 0)
    return X, y


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestXGBoostInitialization:
    def test_default_initialization(self):
        model = XGBoostBaseline(random_state=42)
        assert model.random_state == 42
        assert (
            model.model is not None
        ), "XGBoost model object should be set when xgb is installed"

    def test_custom_random_state(self):
        model = XGBoostBaseline(random_state=7)
        assert model.random_state == 7

    def test_model_is_xgb_classifier(self):
        model = XGBoostBaseline()
        assert isinstance(model.model, xgb.XGBClassifier)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


class TestXGBoostTraining:
    def test_train_completes_without_error(self):
        model = XGBoostBaseline(random_state=0)
        X, y = _make_dataset()
        model.train(X, y)  # Should not raise

    def test_train_with_validation_set(self):
        model = XGBoostBaseline(random_state=0)
        X, y = _make_dataset(n=300)
        split = 240
        model.train(X[:split], y[:split], X[split:], y[split:])

    def test_model_fitted_after_training(self):
        model = XGBoostBaseline(random_state=0)
        X, y = _make_dataset()
        model.train(X, y)
        # XGBoost fitted models expose `feature_importances_`
        assert hasattr(model.model, "feature_importances_")
        assert model.model.feature_importances_ is not None


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


class TestXGBoostPrediction:
    @pytest.fixture(autouse=True)
    def trained_model(self):
        self.model = XGBoostBaseline(random_state=42)
        X, y = _make_dataset()
        self.model.train(X, y)
        self.X, self.y = X, y

    def test_predict_proba_shape(self):
        probs = self.model.model.predict_proba(self.X)
        assert probs.shape == (
            len(self.X),
            2,
        ), "predict_proba must return (n_samples, 2)"

    def test_predict_proba_sums_to_one(self):
        probs = self.model.model.predict_proba(self.X)
        row_sums = probs.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(len(self.X)), atol=1e-5)

    def test_predict_proba_values_in_range(self):
        probs = self.model.model.predict_proba(self.X)
        assert probs.min() >= 0.0 and probs.max() <= 1.0

    def test_binary_predictions(self):
        preds = self.model.model.predict(self.X)
        assert set(preds).issubset({0, 1}), "Binary classifier must only output 0 or 1"


# ---------------------------------------------------------------------------
# Evaluation Metrics
# ---------------------------------------------------------------------------


class TestXGBoostEvaluation:
    @pytest.fixture(autouse=True)
    def trained_model(self):
        self.model = XGBoostBaseline(random_state=42)
        X, y = _make_dataset()
        self.model.train(X, y)
        self.X, self.y = X, y

    def test_evaluate_returns_all_metrics(self):
        metrics = self.model.evaluate(self.X, self.y)
        for key in ("roc_auc", "f1_score", "precision", "recall"):
            assert key in metrics, f"Missing metric: {key}"

    def test_roc_auc_above_random(self):
        metrics = self.model.evaluate(self.X, self.y)
        assert metrics["roc_auc"] >= 0.5, "AUC should be above random baseline (0.5)"

    def test_precision_recall_in_range(self):
        metrics = self.model.evaluate(self.X, self.y)
        assert 0.0 <= metrics["precision"] <= 1.0
        assert 0.0 <= metrics["recall"] <= 1.0

    def test_evaluate_returns_empty_dict_when_model_none(self):
        model = XGBoostBaseline.__new__(XGBoostBaseline)
        model.model = None
        result = model.evaluate(self.X, self.y)
        assert result == {}


# ---------------------------------------------------------------------------
# Feature Importance
# ---------------------------------------------------------------------------


class TestXGBoostFeatureImportance:
    def test_feature_importance_available_after_training(self):
        model = XGBoostBaseline(random_state=0)
        X, y = _make_dataset()
        model.train(X, y)
        importances = model.model.feature_importances_
        assert len(importances) == X.shape[1]

    def test_feature_importance_nonnegative(self):
        model = XGBoostBaseline(random_state=0)
        X, y = _make_dataset()
        model.train(X, y)
        assert (model.model.feature_importances_ >= 0).all()

    def test_feature_importance_sums_to_one(self):
        model = XGBoostBaseline(random_state=0)
        X, y = _make_dataset()
        model.train(X, y)
        total = model.model.feature_importances_.sum()
        assert (
            abs(total - 1.0) < 0.01
        ), f"Feature importances should sum to ~1.0, got {total}"


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestXGBoostSerialization:
    def test_model_save_and_reload(self):
        model = XGBoostBaseline(random_state=42)
        X, y = _make_dataset()
        model.train(X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "xgb_model.json")
            model.model.save_model(save_path)
            assert os.path.exists(save_path)

            # Reload
            reloaded = xgb.XGBClassifier()
            reloaded.load_model(save_path)

            # Predictions should match
            original_probs = model.model.predict_proba(X)[:, 1]
            reloaded_probs = reloaded.predict_proba(X)[:, 1]
            np.testing.assert_allclose(original_probs, reloaded_probs, atol=1e-5)
