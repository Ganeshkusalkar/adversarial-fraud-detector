import pytest
import os
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from api.ab_testing import calculate_z_test, ABTestingEngine

# Import app for testing
with patch.dict("os.environ", {"API_KEYS": "test-key-ab,another-key-ab"}):
    import api.main as api_module
    from api.main import app


@pytest.fixture
def ab_client():
    mock_pred = MagicMock()
    mock_pred.predict.return_value = 0.15

    # Mock baseline models
    with patch.dict("os.environ", {"API_KEYS": "test-key-ab,another-key-ab"}):
        with patch.object(api_module, "PREDICTOR", mock_pred):
            yield TestClient(app, raise_server_exceptions=False)


def test_z_test_statistical_calculation():
    # Insufficient data check
    insufficient = calculate_z_test(5, 1, 5, 2)
    assert insufficient["significant"] is False
    assert "Insufficient data" in insufficient["message"]

    # Statistically significant check
    # 1000 requests in GNN, 150 fraud caught (15%)
    # 1000 requests in XGB, 90 fraud caught (9%)
    significant = calculate_z_test(1000, 150, 1000, 90)
    assert significant["significant"] is True
    assert significant["z_score"] > 3.0
    assert significant["p_value"] < 0.05

    # Not significant check
    not_sig = calculate_z_test(500, 50, 500, 48)
    assert not_sig["significant"] is False
    assert not_sig["p_value"] > 0.05


def test_ab_routing_determinism():
    engine = ABTestingEngine(ab_split=0.5)

    # Verify sticky routing for identical transaction IDs
    route_1a = engine.get_route("TX-001")
    route_1b = engine.get_route("TX-001")
    assert route_1a == route_1b

    route_2a = engine.get_route("TX-999")
    route_2b = engine.get_route("TX-999")
    assert route_2a == route_2b


def test_predict_ab_endpoint_auth(ab_client):
    payload = {
        "TransactionID": "TX-AB-001",
        "card1": 11111,
        "TransactionAmt": 150.0,
        "TransactionDT": 1000,
        "ProductCD": "W",
        "card4": "visa",
        "card6": "credit",
        "vesta_features": [0.0] * 339,
    }

    # Unauthorized
    response = ab_client.post("/api/v1/predict_ab", json=payload)
    assert response.status_code == 403

    # Authorized (Group A or B)
    response_auth = ab_client.post(
        "/api/v1/predict_ab", json=payload, headers={"X-API-Key": "test-key-ab"}
    )
    assert response_auth.status_code == 200
    data = response_auth.json()
    assert "ab_group" in data
    assert data["ab_group"] in ["A", "B"]
    assert "model_used" in data
    assert "fraud_score" in data


def test_ab_status_endpoint(ab_client):
    # Unauthorized
    response = ab_client.get("/api/v1/ab_status")
    assert response.status_code == 403

    # Authorized
    response_auth = ab_client.get(
        "/api/v1/ab_status", headers={"X-API-Key": "test-key-ab"}
    )
    assert response_auth.status_code == 200
    data = response_auth.json()
    assert "group_a" in data
    assert "group_b" in data
    assert "z_test" in data
