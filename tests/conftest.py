import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

# PyTorch availability guard
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

requires_torch = pytest.mark.skipif(
    not HAS_TORCH, reason="PyTorch not installed in this environment"
)


# Transaction payload fixtures
@pytest.fixture
def mock_transaction_payload() -> dict:
    """Minimal valid TransactionInput payload."""
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
    """Fraud payload with high amount and extreme Vesta features."""
    payload = mock_transaction_payload.copy()
    payload["TransactionID"] = "TXN-FRAUD-999"
    payload["TransactionAmt"] = 9999.99
    payload["vesta_features"] = [5.0] * 339
    return payload


# DataFrame fixtures for drift detection
@pytest.fixture
def reference_dataframe() -> pd.DataFrame:
    """Reference DataFrame for drift detection tests."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "amount": np.random.normal(100, 20, 500),
            "velocity": np.random.normal(1.0, 0.2, 500),
            "C1": np.random.normal(2.0, 0.5, 500),
        }
    )


@pytest.fixture
def production_dataframe_no_drift(reference_dataframe) -> pd.DataFrame:
    """Production batch with no drift."""
    np.random.seed(100)
    return pd.DataFrame(
        {
            "amount": np.random.normal(100, 20, 200),
            "velocity": np.random.normal(1.0, 0.2, 200),
            "C1": np.random.normal(2.0, 0.5, 200),
        }
    )


@pytest.fixture
def production_dataframe_with_drift() -> pd.DataFrame:
    """Production batch with feature drift."""
    np.random.seed(200)
    return pd.DataFrame(
        {
            "amount": np.random.normal(300, 80, 200),
            "velocity": np.random.normal(5.0, 1.5, 200),
            "C1": np.random.normal(2.0, 0.5, 200),
        }
    )


# Mock predictor fixtures
@pytest.fixture
def mock_predictor():
    """Predictor that returns a low-risk fraud probability."""
    predictor = MagicMock()
    predictor.predict.return_value = 0.15
    return predictor


@pytest.fixture
def mock_fraud_predictor():
    """Predictor that flags fraud."""
    predictor = MagicMock()
    predictor.predict.return_value = 0.92
    return predictor


# GNN graph fixtures
@pytest.fixture
def simple_graph_tensors():
    """4-node line graph with feature dim=16."""
    if HAS_TORCH:
        x = torch.randn(4, 16)
        edge_index = torch.tensor(
            [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long
        )
        return x, edge_index
    else:
        x = np.random.randn(4, 16).astype(np.float32)
        edge_index = np.array([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=np.int64)
        return x, edge_index


@pytest.fixture
def isolated_node_tensors():
    """3 nodes, zero edges."""
    if HAS_TORCH:
        x = torch.randn(3, 16)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        return x, edge_index
    else:
        x = np.random.randn(3, 16).astype(np.float32)
        edge_index = np.zeros((2, 0), dtype=np.int64)
        return x, edge_index
