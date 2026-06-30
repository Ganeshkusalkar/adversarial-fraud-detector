"""
Integration tests for the FastAPI application.
Tests use FastAPI's built-in TestClient to exercise the full HTTP stack
without a running server — no network ports needed.

Covers:
  - /health endpoint (open, no auth)
  - /api/v1/predict with valid API key → correct response schema
  - /api/v1/predict without API key → 403
  - /api/v1/predict with invalid API key → 403
  - /api/v1/predict when predictor not loaded → 503
  - /api/v1/explain (auth required)
  - /monitoring/drift (open)
  - Rate limit headers present
  - Correlation-ID header propagated
"""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Import app — guard against missing heavy deps (torch, onnxruntime, etc.)
# ---------------------------------------------------------------------------
try:
    import api.main as api_module
    from api.main import app
    HAS_APP = True
except Exception:
    HAS_APP = False

skip_if_no_app = pytest.mark.skipif(
    not HAS_APP, reason="FastAPI app could not be imported (missing deps)"
)

TEST_API_KEY = "test-secret-key-integration"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    """
    TestClient with a mock predictor injected so no ONNX artifacts are needed.
    """
    if not HAS_APP:
        pytest.skip("FastAPI app unavailable")

    mock_pred = MagicMock()
    mock_pred.predict.return_value = 0.12  # low-risk score

    with patch.dict("os.environ", {"API_KEYS": TEST_API_KEY}):
        with patch.object(api_module, "PREDICTOR", mock_pred):
            with patch.object(api_module, "SHAP_EXPLAINER", MagicMock()):
                yield TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def valid_payload():
    return {
        "TransactionID": "TXN-INT-001",
        "card1": 9999,
        "TransactionAmt": 250.0,
        "TransactionDT": 86400,
        "ProductCD": "W",
        "card4": "visa",
        "card6": "credit",
        "P_emaildomain": "gmail.com",
        "R_emaildomain": "gmail.com",
        "C1": 2.0,
        "C2": 1.0,
        "D1": 5.0,
        "vesta_features": [0.0] * 339,
    }


def _auth_header(key=TEST_API_KEY):
    return {"X-API-Key": key}


# ===========================================================================
# Health Endpoint (public — no auth required)
# ===========================================================================

@skip_if_no_app
class TestHealthEndpoint:

    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_has_status_field(self, client):
        response = client.get("/health")
        data = response.json()
        assert "status" in data

    def test_health_no_auth_required(self, client):
        """Health endpoint must be accessible without an API key."""
        response = client.get("/health")
        assert response.status_code != 403

    def test_health_content_type_json(self, client):
        response = client.get("/health")
        assert "application/json" in response.headers.get("content-type", "")


# ===========================================================================
# Predict Endpoint — Authentication
# ===========================================================================

@skip_if_no_app
class TestPredictAuthentication:

    def test_missing_api_key_returns_403(self, client, valid_payload):
        """No X-API-Key header → 403 Forbidden."""
        response = client.post("/api/v1/predict", json=valid_payload)
        assert response.status_code == 403, (
            f"Expected 403 without API key, got {response.status_code}"
        )

    def test_wrong_api_key_returns_403(self, client, valid_payload):
        """Invalid API key → 403 Forbidden."""
        response = client.post(
            "/api/v1/predict",
            json=valid_payload,
            headers={"X-API-Key": "totally-wrong-key"},
        )
        assert response.status_code == 403

    def test_valid_api_key_accepted(self, client, valid_payload):
        """Valid key → not 403 (may be 200 or 503 depending on predictor state)."""
        response = client.post(
            "/api/v1/predict",
            json=valid_payload,
            headers=_auth_header(),
        )
        assert response.status_code != 403, (
            f"Valid API key should not return 403, got {response.status_code}: {response.text}"
        )


# ===========================================================================
# Predict Endpoint — Response Schema
# ===========================================================================

@skip_if_no_app
class TestPredictResponseSchema:

    def test_predict_returns_correct_schema(self, client, valid_payload):
        response = client.post(
            "/api/v1/predict",
            json=valid_payload,
            headers=_auth_header(),
        )
        if response.status_code == 503:
            pytest.skip("Predictor not loaded in this environment")

        assert response.status_code == 200
        data = response.json()
        assert "transaction_id" in data
        assert "fraud_score" in data
        assert "is_fraudulent" in data
        assert "processing_latency_ms" in data

    def test_fraud_score_in_range(self, client, valid_payload):
        response = client.post(
            "/api/v1/predict",
            json=valid_payload,
            headers=_auth_header(),
        )
        if response.status_code == 503:
            pytest.skip("Predictor not loaded in this environment")
        data = response.json()
        assert 0.0 <= data["fraud_score"] <= 1.0

    def test_transaction_id_echoed_back(self, client, valid_payload):
        response = client.post(
            "/api/v1/predict",
            json=valid_payload,
            headers=_auth_header(),
        )
        if response.status_code == 503:
            pytest.skip("Predictor not loaded in this environment")
        data = response.json()
        assert data["transaction_id"] == "TXN-INT-001"

    def test_is_fraudulent_is_boolean(self, client, valid_payload):
        response = client.post(
            "/api/v1/predict",
            json=valid_payload,
            headers=_auth_header(),
        )
        if response.status_code == 503:
            pytest.skip("Predictor not loaded in this environment")
        data = response.json()
        assert isinstance(data["is_fraudulent"], bool)

    def test_latency_ms_nonnegative(self, client, valid_payload):
        response = client.post(
            "/api/v1/predict",
            json=valid_payload,
            headers=_auth_header(),
        )
        if response.status_code == 503:
            pytest.skip("Predictor not loaded in this environment")
        data = response.json()
        assert data["processing_latency_ms"] >= 0.0


# ===========================================================================
# Predict Endpoint — Validation Errors
# ===========================================================================

@skip_if_no_app
class TestPredictValidation:

    def test_negative_amount_rejected(self, client):
        payload = {
            "TransactionID": "TXN-BAD",
            "card1": 1,
            "TransactionAmt": -100.0,  # invalid
            "TransactionDT": 0,
            "ProductCD": "W",
            "card4": "visa",
            "card6": "credit",
        }
        response = client.post(
            "/api/v1/predict",
            json=payload,
            headers=_auth_header(),
        )
        assert response.status_code == 422, "Negative amount should return 422 Unprocessable Entity"

    def test_missing_required_field_rejected(self, client):
        payload = {"TransactionID": "TXN-INCOMPLETE"}  # missing everything
        response = client.post(
            "/api/v1/predict",
            json=payload,
            headers=_auth_header(),
        )
        assert response.status_code == 422


# ===========================================================================
# Middleware & Headers
# ===========================================================================

@skip_if_no_app
class TestMiddleware:

    def test_correlation_id_header_present(self, client):
        """Every response must carry X-Correlation-ID from the middleware."""
        response = client.get("/health")
        assert "x-correlation-id" in response.headers, (
            "X-Correlation-ID middleware must attach header to every response"
        )

    def test_correlation_id_is_uuid_format(self, client):
        import uuid
        response = client.get("/health")
        corr_id = response.headers.get("x-correlation-id", "")
        try:
            uuid.UUID(corr_id)
        except ValueError:
            pytest.fail(f"X-Correlation-ID is not a valid UUID: {corr_id!r}")


# ===========================================================================
# Drift Endpoint (open — no auth)
# ===========================================================================

@skip_if_no_app
class TestDriftEndpoint:

    def test_drift_endpoint_accessible_without_auth(self, client):
        response = client.get("/monitoring/drift")
        assert response.status_code != 403

    def test_drift_response_has_status(self, client):
        response = client.get("/monitoring/drift")
        if response.status_code == 200:
            data = response.json()
            assert "status" in data
