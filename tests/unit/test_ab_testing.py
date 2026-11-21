import pytest
import os
from unittest.mock import patch
from fastapi import HTTPException
from api.ab_testing import calculate_z_test, ABTestingEngine
from api.dependencies import verify_api_key


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
    assert route_1a in ["A", "B"]

    route_2a = engine.get_route("TX-999")
    route_2b = engine.get_route("TX-999")
    assert route_2a == route_2b
    assert route_2a in ["A", "B"]


def test_ab_metrics_recording():
    engine = ABTestingEngine(ab_split=0.5)
    engine.record_metrics(group="A", is_fraud=True, latency_ms=12.5)
    engine.record_metrics(group="A", is_fraud=False, latency_ms=15.0)
    engine.record_metrics(group="B", is_fraud=False, latency_ms=8.0)

    status = engine.get_status_report()
    assert status["group_a"]["total_requests"] == 2
    assert status["group_a"]["fraud_detected"] == 1
    assert status["group_b"]["total_requests"] == 1
    assert status["group_b"]["fraud_detected"] == 0
    assert "z_test" in status


def test_predict_ab_auth_security():
    with patch.dict(os.environ, {"API_KEYS": "valid-key-1,valid-key-2"}):
        # Missing key raises 403
        with pytest.raises(HTTPException) as exc_info:
            verify_api_key(None)
        assert exc_info.value.status_code == 403

        # Invalid key raises 403
        with pytest.raises(HTTPException) as exc_info2:
            verify_api_key("unauthorized-key")
        assert exc_info2.value.status_code == 403

        # Valid key succeeds
        assert verify_api_key("valid-key-1") == "valid-key-1"
        assert verify_api_key("valid-key-2") == "valid-key-2"

