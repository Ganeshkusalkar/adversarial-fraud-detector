import logging
import os
import pickle
import time
import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from ..schemas import TransactionRequest, PredictionResponse
from ..dependencies import get_onnx_session, ONNXInferenceSession

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/predict", tags=["Inference"])

# -----------------------------------------------------------------------
# Decision Threshold
# -----------------------------------------------------------------------
# Read from environment variable so staging / production can be tuned
# independently without redeployment. Matches the 0.38 threshold used
# during Focal Loss training evaluation for consistent recall behaviour.
DECISION_THRESHOLD = float(os.environ.get("FRAUD_DECISION_THRESHOLD", "0.38"))

# -----------------------------------------------------------------------
# Scaler (loaded once at startup, not per-request)
# -----------------------------------------------------------------------
_SCALER_PATH = os.environ.get("SCALER_PATH", "artifacts/production/scaler.pkl")
_scaler = None  # Lazy-loaded on first request

def _load_scaler():
    """Loads the fitted StandardScaler from the artifact store."""
    global _scaler
    if _scaler is not None:
        return _scaler
    if os.path.exists(_SCALER_PATH):
        with open(_SCALER_PATH, "rb") as f:
            _scaler = pickle.load(f)
        logger.info(f"StandardScaler loaded from: {_SCALER_PATH}")
    else:
        logger.warning(
            f"Scaler artifact not found at {_SCALER_PATH}. "
            "Feature vectors will be passed to ONNX un-scaled — "
            "run train_real_subset.py to generate the scaler artifact."
        )
    return _scaler


# -----------------------------------------------------------------------
# Request counters (in-process metrics — in production, use Prometheus)
# -----------------------------------------------------------------------
LATENCY_HISTOGRAM: list = []
TPS_COUNTER: int = 0


@router.post("", response_model=PredictionResponse)
def predict_transaction(
    request: TransactionRequest,
    session: ONNXInferenceSession = Depends(get_onnx_session),
) -> PredictionResponse:
    """
    Sub-50ms inference endpoint for real-time adversarial fraud detection.

    Accepts a transaction payload, constructs the feature vector,
    applies the fitted StandardScaler, and scores through the hardened
    ONNX GraphSAGE model. Uses the tuned 0.38 decision threshold
    (configurable via FRAUD_DECISION_THRESHOLD env variable) for
    high-recall fraud detection consistent with training evaluation.
    """
    global TPS_COUNTER
    start_time = time.perf_counter()
    TPS_COUNTER += 1

    try:
        # ── Feature Construction ──────────────────────────────────────
        features = request.features
        if features is None:
            # Construct the baseline 6-dimensional feature vector
            # from structured transaction attributes
            features = [
                float(request.amount),
                float(request.oldbalanceOrg) if request.oldbalanceOrg else 0.0,
                float(request.newbalanceOrig) if request.newbalanceOrig else 0.0,
                float(request.oldbalanceDest) if request.oldbalanceDest else 0.0,
                float(request.newbalanceDest) if request.newbalanceDest else 0.0,
                float(request.step) if request.step else 0.0,
            ]

        # ── Feature Scaling ───────────────────────────────────────────
        # Apply the same StandardScaler fitted during training so that
        # raw transaction amounts don't dominate the GNN embeddings.
        node_features = np.array([features], dtype=np.float32)  # shape: [1, n_features]

        scaler = _load_scaler()
        if scaler is not None:
            try:
                node_features = scaler.transform(node_features).astype(np.float32)
            except Exception as scaler_err:
                logger.warning(
                    f"Scaler transform failed (shape mismatch?): {scaler_err}. "
                    "Proceeding with un-scaled features."
                )

        # ── ONNX Inference ────────────────────────────────────────────
        # Single-node self-loop edge index: the node is its own neighbor
        # (required for the SAGEConv to operate without graph context).
        edge_index = np.array([[0], [0]], dtype=np.int64)

        risk_score: float = session.predict(node_features, edge_index)

        # ── Decision at Tuned Threshold ───────────────────────────────
        # Using 0.38 (vs naive 0.5) to match recall-optimized training evaluation.
        # This trades ~5% precision for ~20% recall improvement on fraud class.
        flagged: bool = risk_score >= DECISION_THRESHOLD

        # ── Latency Tracking ──────────────────────────────────────────
        latency_ms = (time.perf_counter() - start_time) * 1_000.0
        LATENCY_HISTOGRAM.append(latency_ms)
        if len(LATENCY_HISTOGRAM) > 1_000:
            LATENCY_HISTOGRAM.pop(0)

        logger.info(
            "Transaction scored.",
            extra={
                "extra_fields": {
                    "txId": request.txId,
                    "risk_score": round(risk_score, 4),
                    "flagged": flagged,
                    "threshold": DECISION_THRESHOLD,
                    "latency_ms": round(latency_ms, 2),
                }
            },
        )

        return PredictionResponse(
            txId=request.txId,
            risk_score=risk_score,
            flagged=flagged,
            inference_latency_ms=round(latency_ms, 2),
        )

    except Exception as exc:
        logger.error(f"Inference pipeline failure: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal GNN Inference Failure")
