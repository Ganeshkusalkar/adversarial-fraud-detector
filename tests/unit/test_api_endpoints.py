import pytest
import os
from unittest.mock import patch, MagicMock
from pydantic import ValidationError
from fastapi import HTTPException
import api.main as api_module
from api.dependencies import verify_api_key
from api.schemas import TransactionInput, SystemHealthResponse


@pytest.fixture
def sample_payload():
    return {
        "TransactionID": "TXN-UNIT-001",
        "card1": 12345,
        "TransactionAmt": 150.0,
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


def test_health_response_schema():
    stats = api_module.GRAPH_ENGINE.get_stats()
    cb = api_module.CIRCUIT_BREAKER.get_status()
    resp = SystemHealthResponse(
        status="healthy",
        model_version="v1.0.0-onnx",
        graph_store=stats,
        circuit_breaker=cb,
        uptime_seconds=42.0,
    )
    assert resp.status == "healthy"
    assert resp.model_version == "v1.0.0-onnx"
    assert "total_nodes" in resp.graph_store
    assert "state" in resp.circuit_breaker
    assert resp.uptime_seconds == 42.0


def test_api_key_auth_verification():
    with patch.dict(os.environ, {"API_KEYS": "valid-key-123,valid-key-456"}):
        # Missing key raises 403
        with pytest.raises(HTTPException) as exc1:
            verify_api_key(None)
        assert exc1.value.status_code == 403

        # Invalid key raises 403
        with pytest.raises(HTTPException) as exc2:
            verify_api_key("invalid-token")
        assert exc2.value.status_code == 403

        # Valid key passes
        assert verify_api_key("valid-key-123") == "valid-key-123"
        assert verify_api_key("valid-key-456") == "valid-key-456"


def test_payload_validation_success(sample_payload):
    tx = TransactionInput(**sample_payload)
    assert tx.TransactionID == "TXN-UNIT-001"
    assert tx.TransactionAmt == 150.0
    assert len(tx.vesta_features) == 339


def test_payload_validation_negative_amount():
    with pytest.raises(ValidationError):
        TransactionInput(
            TransactionID="TX-BAD",
            card1=12345,
            TransactionAmt=-10.0,  # Negative amount must fail ge=0.0
            TransactionDT=1000,
            ProductCD="W",
            card4="visa",
            card6="credit",
        )


def test_get_predictor_uninitialized():
    with patch.object(api_module, "PREDICTOR", None):
        with pytest.raises(HTTPException) as exc:
            api_module.get_predictor()
        assert exc.value.status_code == 503
        assert "uninitialized" in exc.value.detail.lower()


def test_get_explainer_uninitialized():
    with patch.object(api_module, "SHAP_EXPLAINER", None):
        with pytest.raises(HTTPException) as exc:
            api_module.get_explainer()
        assert exc.value.status_code == 503
        assert "uninitialized" in exc.value.detail.lower()


def test_tabular_fallback_predict():
    # Low risk
    low = api_module._tabular_fallback_predict(amount=50.0, c2=1.0)
    assert 0.01 <= low < 0.20

    # High amount and velocity
    high = api_module._tabular_fallback_predict(amount=1000.0, c2=5.0)
    assert high > 0.50
    assert high <= 0.99
