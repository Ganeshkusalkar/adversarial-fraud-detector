import pytest
import numpy as np
import pandas as pd
from src.models.xgboost_baseline import XGBoostBaseline

def test_xgboost_baseline_initialization():
    """Test that the XGBoost baseline initializes correctly."""
    model = XGBoostBaseline(random_state=42)
    assert model.random_state == 42
    # If xgboost is installed, model.model should not be None
    # We just ensure it doesn't crash on init.

def test_xgboost_baseline_training():
    """Test that the XGBoost baseline can train on dummy data."""
    model = XGBoostBaseline(random_state=42)
    if model.model is None:
        pytest.skip("xgboost not installed")
        
    np.random.seed(42)
    X = pd.DataFrame({
        "feat1": np.random.rand(100),
        "feat2": np.random.rand(100)
    })
    y = np.random.randint(0, 2, 100)
    
    # Train
    model.train(X, y)
    
    # Evaluate
    metrics = model.evaluate(X, y)
    assert "roc_auc" in metrics
    assert "f1_score" in metrics
    assert metrics["roc_auc"] >= 0.0 # Should be a valid float
