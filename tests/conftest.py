"""
Shared pytest fixtures for the Adversarial Fraud Detector test suite.
All fixtures defined here are automatically available to every test file.
"""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# PyTorch availability guard
# ---------------------------------------------------------------------------
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

requires_torch = pytest.mark.skipif(
    not HAS_TORCH, reason="PyTorch not installed in this environment"
)


# ---------------------------------------------------------------------------
# Transaction payload fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_transaction_payload() -> dict:
    """A minimal valid TransactionInput payload dict."""
    return {
        "TransactionID": "TXN-TEST-001",
        "card1": 12345,
        "TransactionAmt": 150.75,
        "TransactionDT": 86400,
        "ProductCD": "W",
        "card4": "visa",
        "card6": "credit",
        "P_emaildomain": "gmail.com",
        "R_emaildomain": "gmail.com",
        "C1": 1.0,
        "C2": 1.0,
        "D1": 10.0,
        "vesta_features": [0.0] * 339,
    }


@pytest.fixture
def mock_fraud_payload(mock_transaction_payload) -> dict:
    """A payload designed to look suspicious (high amount, extreme Vesta features)."""
    payload = mock_transaction_payload.copy()
    payload["TransactionID"] = "TXN-FRAUD-999"
    payload["TransactionAmt"] = 9999.99
    payload["vesta_features"] = [5.0] * 339  # extreme values
    return payload


# ---------------------------------------------------------------------------
# DataFrame fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def reference_dataframe() -> pd.DataFrame:
    """Small reference DataFrame for drift detection tests."""
    np.random.seed(42)
    return pd.DataFrame({
        "amount": np.random.normal(100, 20, 500),
        "velocity": np.random.normal(1.0, 0.2, 500),
        "C1": np.random.normal(2.0, 0.5, 500),
    })


@pytest.fixture
def production_dataframe_no_drift(reference_dataframe) -> pd.DataFrame:
    """Production batch drawn from the same distribution — no drift expected."""
    np.random.seed(100)
    return pd.DataFrame({
        "amount": np.random.normal(100, 20, 200),
        "velocity": np.random.normal(1.0, 0.2, 200),
        "C1": np.random.normal(2.0, 0.5, 200),
    })


@pytest.fixture
def production_dataframe_with_drift() -> pd.DataFrame:
    """Production batch with severely shifted distributions — drift expected."""
    np.random.seed(200)
    return pd.DataFrame({
        "amount": np.random.normal(300, 80, 200),   # mean shifted 3x
        "velocity": np.random.normal(5.0, 1.5, 200),  # mean shifted 5x
        "C1": np.random.normal(2.0, 0.5, 200),        # unchanged
    })


# ---------------------------------------------------------------------------
# Mock model predictor fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_predictor():
    """
    A MagicMock CalibratedFraudPredictor that returns a fixed fraud probability.
    Useful for testing API routes without loading real ONNX artifacts.
    """
    predictor = MagicMock()
    predictor.predict.return_value = 0.15  # low-risk by default
    return predictor


@pytest.fixture
def mock_fraud_predictor():
    """A MagicMock predictor that always flags fraud (prob = 0.92)."""
    predictor = MagicMock()
    predictor.predict.return_value = 0.92
    return predictor


# ---------------------------------------------------------------------------
# GNN graph fixtures (requires torch)
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_graph_tensors():
    """
    A tiny 4-node line graph with feature dim=16.
    Returns (x, edge_index) as torch Tensors (or numpy if torch unavailable).
    """
    if HAS_TORCH:
        x = torch.randn(4, 16)
        edge_index = torch.tensor(
            [[0, 1, 1, 2, 2, 3],
             [1, 0, 2, 1, 3, 2]], dtype=torch.long
        )
        return x, edge_index
    else:
        x = np.random.randn(4, 16).astype(np.float32)
        edge_index = np.array([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=np.int64)
        return x, edge_index


@pytest.fixture
def isolated_node_tensors():
    """3 nodes, zero edges — tests behaviour on disconnected graphs."""
    if HAS_TORCH:
        x = torch.randn(3, 16)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        return x, edge_index
    else:
        x = np.random.randn(3, 16).astype(np.float32)
        edge_index = np.zeros((2, 0), dtype=np.int64)
        return x, edge_index
