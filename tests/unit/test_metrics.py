import pytest
import numpy as np
from src.evaluation.metrics import FraudEvaluationEngine

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@pytest.fixture
def base_config():
    return {
        "training": {
            "decision_threshold": 0.5,
        }
    }


def test_compute_standard_metrics_perfect_predictions(base_config):
    engine = FraudEvaluationEngine(base_config)

    # Perfect predictions
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.9, 0.8])

    metrics = engine.compute_standard_metrics(y_true, y_prob)
    assert metrics["AUC-ROC"] == 1.0
    assert metrics["Average_Precision"] == 1.0
    assert metrics["Precision_Fraud_Class"] == 1.0
    assert metrics["Recall_Sensitivity"] == 1.0
    assert metrics["F1_Score_Fraud_Class"] == 1.0


def test_compute_standard_metrics_imperfect_predictions(base_config):
    engine = FraudEvaluationEngine(base_config)

    # Simple imperfect predictions
    y_true = np.array([0, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.6, 0.8, 0.4, 0.9])  # threshold 0.5
    # y_pred = [0, 1, 1, 0, 1]
    # TP: 2, FP: 1, TN: 1, FN: 1

    metrics = engine.compute_standard_metrics(y_true, y_prob)
    assert 0.0 < metrics["AUC-ROC"] < 1.0
    assert 0.0 < metrics["Average_Precision"] < 1.0
    assert metrics["Precision_Fraud_Class"] == 2.0 / 3.0
    assert metrics["Recall_Sensitivity"] == 2.0 / 3.0


def test_precision_at_k(base_config):
    # Set config to evaluate top 20%
    engine = FraudEvaluationEngine(base_config)
    engine.target_precision_k = 0.2

    y_true = np.array([0, 0, 0, 0, 1])  # 5 samples, top 20% is top 1 sample
    y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.9])  # top 1 is index 4 (y_true=1)

    metrics = engine.compute_standard_metrics(y_true, y_prob)
    assert metrics["Precision_at_Top_1_Percent"] == 1.0


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_calculate_adversarial_robustness_torch(base_config):
    # Mocking torch modules to test calculate_adversarial_robustness function
    from unittest.mock import MagicMock

    engine = FraudEvaluationEngine(base_config)

    mock_model = MagicMock(spec=torch.nn.Module)
    # in_channels attribute
    mock_model.conv1 = MagicMock()
    mock_model.conv1.in_channels = 20

    mock_generator = MagicMock(spec=torch.nn.Module)
    mock_generator.feature_dim = 15
    mock_generator.sample_noise.return_value = torch.randn(1, 5, 32)
    mock_generator.return_value = torch.randn(1, 5, 15)

    # Mock predict method on the model to return scores
    # logits shape: (num_nodes, out_channels) -> we classify based on output
    # let's mock it to always return [0.1, 0.9] (class 1 index 1 is high)
    mock_logits = torch.tensor([[0.1, 0.9]])
    mock_model.return_value = mock_logits

    # Run simulation
    robustness = engine.calculate_adversarial_robustness(
        hardened_model=mock_model, generator=mock_generator, n_attacks=3, device="cpu"
    )
    # returns float fraction of caught attacks
    assert isinstance(robustness, float)
    assert 0.0 <= robustness <= 1.0
