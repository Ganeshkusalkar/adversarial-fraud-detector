"""
Unit tests for Pydantic API schemas:
  - TransactionInput validation (field types, ranges, defaults)
  - FraudPredictionResponse construction and field types
"""
import pytest
from pydantic import ValidationError

from api.schemas import TransactionInput, FraudPredictionResponse


# ===========================================================================
# TransactionInput Validation
# ===========================================================================

class TestTransactionInputSchema:

    def _valid_payload(self, **overrides) -> dict:
        payload = {
            "TransactionID": "TXN-001",
            "card1": 12345,
            "TransactionAmt": 99.99,
            "TransactionDT": 86400,
            "ProductCD": "W",
            "card4": "visa",
            "card6": "credit",
        }
        payload.update(overrides)
        return payload

    def test_valid_payload_parses_correctly(self):
        txn = TransactionInput(**self._valid_payload())
        assert txn.TransactionID == "TXN-001"
        assert txn.TransactionAmt == 99.99
        assert txn.card1 == 12345

    def test_vesta_features_defaults_to_339_zeros(self):
        txn = TransactionInput(**self._valid_payload())
        assert len(txn.vesta_features) == 339
        assert all(v == 0.0 for v in txn.vesta_features)

    def test_email_domains_default_to_unknown(self):
        txn = TransactionInput(**self._valid_payload())
        assert txn.P_emaildomain == "UNKNOWN"
        assert txn.R_emaildomain == "UNKNOWN"

    def test_c1_c2_d1_default_to_zero(self):
        txn = TransactionInput(**self._valid_payload())
        assert txn.C1 == 0.0
        assert txn.C2 == 0.0
        assert txn.D1 == 0.0

    def test_negative_transaction_amount_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            TransactionInput(**self._valid_payload(TransactionAmt=-1.0))
        errors = exc_info.value.errors()
        assert any("TransactionAmt" in str(e) for e in errors)

    def test_zero_transaction_amount_accepted(self):
        """Zero is a valid amount (ge=0.0)."""
        txn = TransactionInput(**self._valid_payload(TransactionAmt=0.0))
        assert txn.TransactionAmt == 0.0

    def test_missing_required_field_raises(self):
        payload = self._valid_payload()
        del payload["TransactionID"]
        with pytest.raises(ValidationError):
            TransactionInput(**payload)

    def test_missing_card1_raises(self):
        payload = self._valid_payload()
        del payload["card1"]
        with pytest.raises(ValidationError):
            TransactionInput(**payload)

    def test_custom_vesta_features_accepted(self):
        custom_vesta = [0.5] * 339
        txn = TransactionInput(**self._valid_payload(), vesta_features=custom_vesta)
        assert len(txn.vesta_features) == 339
        assert txn.vesta_features[0] == 0.5

    def test_optional_email_fields_accept_none_values(self):
        txn = TransactionInput(
            **self._valid_payload(P_emaildomain=None, R_emaildomain=None)
        )
        # Schema defines Optional[str] with a default — None should be accepted
        assert txn.P_emaildomain is None or txn.P_emaildomain == "UNKNOWN"

    def test_large_transaction_amount_accepted(self):
        txn = TransactionInput(**self._valid_payload(TransactionAmt=1_000_000.0))
        assert txn.TransactionAmt == 1_000_000.0

    def test_float_c1_c2_d1_accepted(self):
        txn = TransactionInput(**self._valid_payload(C1=3.5, C2=1.2, D1=45.7))
        assert txn.C1 == 3.5
        assert txn.C2 == 1.2
        assert txn.D1 == 45.7


# ===========================================================================
# FraudPredictionResponse Construction
# ===========================================================================

class TestFraudPredictionResponseSchema:

    def _valid_response(self, **overrides) -> dict:
        response = {
            "transaction_id": "TXN-001",
            "fraud_score": 0.25,
            "is_fraudulent": False,
            "processing_latency_ms": 12.5,
        }
        response.update(overrides)
        return response

    def test_valid_response_parses(self):
        resp = FraudPredictionResponse(**self._valid_response())
        assert resp.transaction_id == "TXN-001"
        assert resp.fraud_score == 0.25
        assert resp.is_fraudulent is False
        assert resp.processing_latency_ms == 12.5

    def test_fraud_score_zero(self):
        resp = FraudPredictionResponse(**self._valid_response(fraud_score=0.0))
        assert resp.fraud_score == 0.0

    def test_fraud_score_one(self):
        resp = FraudPredictionResponse(**self._valid_response(fraud_score=1.0))
        assert resp.fraud_score == 1.0

    def test_is_fraudulent_true(self):
        resp = FraudPredictionResponse(**self._valid_response(is_fraudulent=True))
        assert resp.is_fraudulent is True

    def test_missing_fraud_score_raises(self):
        payload = self._valid_response()
        del payload["fraud_score"]
        with pytest.raises(ValidationError):
            FraudPredictionResponse(**payload)

    def test_missing_transaction_id_raises(self):
        payload = self._valid_response()
        del payload["transaction_id"]
        with pytest.raises(ValidationError):
            FraudPredictionResponse(**payload)

    def test_response_serializable_to_dict(self):
        resp = FraudPredictionResponse(**self._valid_response())
        d = resp.model_dump()
        assert d["transaction_id"] == "TXN-001"
        assert d["is_fraudulent"] is False

    def test_json_serializable(self):
        """Response must be JSON-serializable (required for FastAPI returns)."""
        import json
        resp = FraudPredictionResponse(**self._valid_response())
        json_str = resp.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["fraud_score"] == 0.25
