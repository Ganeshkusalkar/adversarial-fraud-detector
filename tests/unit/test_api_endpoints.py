import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Mock os.environ BEFORE importing api.main to configure API keys
with patch.dict("os.environ", {"API_KEYS": "test-key-123,another-key-456"}):
    import api.main as api_module
    from api.main import app


@pytest.fixture
def client():
    # Provide a client with mocked Predictor and SHAP Explainer
    mock_pred = MagicMock()
    mock_pred.predict.return_value = 0.15

    mock_shap = MagicMock()
    mock_shap.explain.return_value = {"Vesta_0": 0.05, "TransactionAmt": 0.12}

    with patch.object(api_module, "PREDICTOR", mock_pred):
        with patch.object(api_module, "SHAP_EXPLAINER", mock_shap):
            yield TestClient(app, raise_server_exceptions=False)


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


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    # PREDICTOR is mocked, so engine_loaded should be True
    assert response.json()["engine_loaded"] is True


def test_health_endpoint_when_predictor_none(client):
    with patch.object(api_module, "PREDICTOR", None):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["engine_loaded"] is False


def test_predict_endpoint_unauthorized(client, sample_payload):
    # Missing API key header
    response = client.post("/api/v1/predict", json=sample_payload)
    assert response.status_code == 403

    # Incorrect API key header
    response = client.post(
        "/api/v1/predict", json=sample_payload, headers={"X-API-Key": "wrong"}
    )
    assert response.status_code == 403


def test_predict_endpoint_success(client, sample_payload):
    response = client.post(
        "/api/v1/predict", json=sample_payload, headers={"X-API-Key": "test-key-123"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["transaction_id"] == "TXN-UNIT-001"
    assert data["fraud_score"] == 0.15
    assert data["is_fraudulent"] is False
    assert data["processing_latency_ms"] >= 0.0
    assert "X-Correlation-ID" in response.headers


def test_predict_endpoint_validation_error(client):
    # Negative transaction amount should fail validation (ge=0.0)
    bad_payload = {
        "TransactionID": "TXN-BAD",
        "card1": 12345,
        "TransactionAmt": -10.0,
        "TransactionDT": 86400,
        "ProductCD": "W",
        "card4": "visa",
        "card6": "credit",
    }
    response = client.post(
        "/api/v1/predict", json=bad_payload, headers={"X-API-Key": "test-key-123"}
    )
    assert response.status_code == 422


def test_predict_endpoint_uninitialized_predictor(client, sample_payload):
    with patch.object(api_module, "PREDICTOR", None):
        response = client.post(
            "/api/v1/predict",
            json=sample_payload,
            headers={"X-API-Key": "test-key-123"},
        )
        assert response.status_code == 503
        assert "uninitialized" in response.json()["detail"]


def test_explain_endpoint_success(client, sample_payload):
    response = client.post(
        "/api/v1/explain", json=sample_payload, headers={"X-API-Key": "test-key-123"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["transaction_id"] == "TXN-UNIT-001"
    assert "explanation" in data


def test_explain_endpoint_uninitialized_explainer(client, sample_payload):
    with patch.object(api_module, "SHAP_EXPLAINER", None):
        response = client.post(
            "/api/v1/explain",
            json=sample_payload,
            headers={"X-API-Key": "test-key-123"},
        )
        assert response.status_code == 503
        assert "uninitialized" in response.json()["detail"]


def test_drift_endpoint_success(client):
    response = client.get("/monitoring/drift")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "drift_report" in data


def test_drift_endpoint_uninitialized(client):
    with patch.object(api_module, "_REFERENCE_STATS", None):
        response = client.get("/monitoring/drift")
        assert response.status_code == 503
        assert "unavailable" in response.json()["status"]
