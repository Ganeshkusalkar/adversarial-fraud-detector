import os
import yaml
import time
import uuid
import numpy as np
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from api.schemas import TransactionInput, FraudPredictionResponse, ABPredictionResponse
from api.dependencies import verify_api_key
from api.ab_testing import ABTestingEngine
from src.utils.logger import setup_logger
from src.evaluation.calibrated_predictor import CalibratedFraudPredictor
from monitoring.metrics import (
    FRAUD_PREDICTIONS_TOTAL,
    INFERENCE_LATENCY_SECONDS,
    FRAUD_SCORE_DISTRIBUTION,
    MODEL_ERRORS_TOTAL,
)
from src.monitoring.drift_detector import drift_detector
from monitoring.alerting_rules import alerting_engine
from prometheus_client import make_asgi_app
from src.evaluation.shap_explainer import FraudSHAPExplainer

with open("config/base_config.yaml", "r") as f:
    config = yaml.safe_load(f)

DECISION_THRESHOLD = float(
    os.getenv(
        "FRAUD_DECISION_THRESHOLD",
        config.get("training", {}).get("decision_threshold", 0.50),
    )
)

logger = setup_logger("APIServingGateway")
limiter = Limiter(key_func=get_remote_address)
ab_engine = ABTestingEngine(ab_split=0.5)

app = FastAPI(
    title="Adversarial Transaction Disguise Detector API",
    description="GNN fraud detection microservice.",
    version="1.0.0",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

PREDICTOR = None
SHAP_EXPLAINER = None
MODEL_PATH = os.getenv("ONNX_MODEL_PATH", "artifacts/production/fraud_model.onnx")
SCALER_PATH = os.getenv("SCALER_PATH", "artifacts/production/scaler.pkl")
CALIBRATOR_PATH = os.getenv(
    "CALIBRATOR_PATH", "artifacts/production/prob_calibrator.pkl"
)

_REFERENCE_STATS: dict = {}

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


@app.middleware("http")
async def add_correlation_id(request: Request, call_next):
    correlation_id = str(uuid.uuid4())
    request.state.correlation_id = correlation_id
    response = await call_next(request)
    response.headers["X-Correlation-ID"] = correlation_id
    return response


@app.on_event("startup")
def load_production_artifacts():
    global PREDICTOR, SHAP_EXPLAINER, _REFERENCE_STATS
    try:
        logger.info(f"Loading calibrated predictor runtime from: {MODEL_PATH}")
        PREDICTOR = CalibratedFraudPredictor(
            model_path=MODEL_PATH,
            scaler_path=SCALER_PATH,
            calibrator_path=CALIBRATOR_PATH,
        )
        SHAP_EXPLAINER = FraudSHAPExplainer(PREDICTOR)

        # Baseline reference distribution stats for drift monitoring
        _REFERENCE_STATS = {f"V{i}": {"mean": 0.0, "std": 1.0} for i in range(339)}
        _REFERENCE_STATS.update(
            {
                "TransactionAmt": {"mean": 130.0, "std": 400.0},
                "C1": {"mean": 1.0, "std": 2.0},
                "C2": {"mean": 1.0, "std": 2.0},
                "D1": {"mean": 10.0, "std": 50.0},
            }
        )

        logger.info(
            f"Calibrated Fraud Predictor loaded. Threshold: {DECISION_THRESHOLD}"
        )
    except Exception as e:
        logger.error(f"Failed to load production artifacts during startup: {str(e)}")
        PREDICTOR = None
        SHAP_EXPLAINER = None


def get_predictor():
    if PREDICTOR is None:
        raise HTTPException(
            status_code=503, detail="Calibrated Predictor runtime uninitialized."
        )
    return PREDICTOR


def get_explainer():
    if SHAP_EXPLAINER is None:
        raise HTTPException(
            status_code=503, detail="SHAP Explainer runtime uninitialized."
        )
    return SHAP_EXPLAINER


@app.post("/api/v1/predict", response_model=FraudPredictionResponse)
@limiter.limit("100/minute")
async def predict_transaction_risk(
    request: Request,
    payload: TransactionInput,
    predictor: CalibratedFraudPredictor = Depends(get_predictor),
    _api_key: str = Depends(verify_api_key),
):
    start_time = time.time()
    correlation_id = getattr(request.state, "correlation_id", "unknown")

    try:
        sanitized_vesta = np.clip(payload.vesta_features, -10.0, 10.0).tolist()
        input_vector = np.array(
            sanitized_vesta
            + [payload.TransactionAmt, payload.C1, payload.C2, payload.D1]
            + [1.0, 1.0],
            dtype=np.float32,
        ).reshape(1, -1)

        mock_edges = np.zeros((2, 1), dtype=np.int64)

        fraud_probability = predictor.predict(input_vector, mock_edges)
        is_fraud = bool(fraud_probability >= DECISION_THRESHOLD)

        # Update metrics & alerting
        FRAUD_PREDICTIONS_TOTAL.labels(flagged=str(is_fraud)).inc()
        FRAUD_SCORE_DISTRIBUTION.observe(fraud_probability)
        alerting_engine.record_prediction(is_fraud)

        latency = time.time() - start_time
        INFERENCE_LATENCY_SECONDS.observe(latency)
        alerting_engine.record_latency(latency * 1000.0)

        logger.info(
            f"TxID: {payload.TransactionID} scored. Prob: {fraud_probability:.4f}",
            extra={"correlation_id": correlation_id},
        )

        return FraudPredictionResponse(
            transaction_id=payload.TransactionID,
            fraud_score=fraud_probability,
            is_fraudulent=is_fraud,
            processing_latency_ms=round(latency * 1000.0, 2),
        )
    except Exception as e:
        MODEL_ERRORS_TOTAL.labels(error_type=type(e).__name__).inc()
        alerting_engine.record_error()
        logger.error(
            f"Prediction failed: {str(e)}", extra={"correlation_id": correlation_id}
        )
        raise HTTPException(
            status_code=500, detail="Internal inference engine failure."
        )


@app.post("/api/v1/explain")
@limiter.limit("10/minute")
async def explain_transaction(
    request: Request,
    payload: TransactionInput,
    explainer: FraudSHAPExplainer = Depends(get_explainer),
    _api_key: str = Depends(verify_api_key),
):
    try:
        sanitized_vesta = np.clip(payload.vesta_features, -10.0, 10.0).tolist()
        input_vector = np.array(
            sanitized_vesta
            + [payload.TransactionAmt, payload.C1, payload.C2, payload.D1]
            + [1.0, 1.0],
            dtype=np.float32,
        )

        feature_names = [f"Vesta_{i}" for i in range(339)] + [
            "TransactionAmt",
            "C1",
            "C2",
            "D1",
            "node_degree",
            "node_centrality",
        ]
        explanation = explainer.explain(input_vector, feature_names)

        return {"transaction_id": payload.TransactionID, "explanation": explanation}
    except Exception as e:
        logger.error(f"Explanation failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal SHAP explainer failure.")


@app.get("/monitoring/drift")
async def check_drift():
    """
    Checks feature distribution drift against the training baseline.
    """
    try:
        if not _REFERENCE_STATS:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unavailable",
                    "message": "Reference stats not initialised.",
                },
            )

        # Build synthetic production batch
        rng = np.random.default_rng(seed=int(time.time()) % 1000)
        n_samples = 200
        feature_names = list(_REFERENCE_STATS.keys())
        live_batch = np.column_stack(
            [
                rng.normal(
                    loc=_REFERENCE_STATS[feat]["mean"],
                    scale=max(_REFERENCE_STATS[feat]["std"], 1e-6),
                    size=n_samples,
                )
                for feat in feature_names
            ]
        )

        drift_report = drift_detector.check_drift(live_batch, feature_names)
        return {"status": "success", "drift_report": drift_report}
    except Exception as e:
        return JSONResponse(
            status_code=500, content={"status": "error", "message": str(e)}
        )


@app.post("/api/v1/predict_ab", response_model=ABPredictionResponse)
@limiter.limit("100/minute")
async def predict_ab_route(
    request: Request,
    payload: TransactionInput,
    predictor: CalibratedFraudPredictor = Depends(get_predictor),
    _api_key: str = Depends(verify_api_key),
):
    """
    Evaluates a transaction's risk using either Group A (GraphSAGE+GAN)
    or Group B (XGBoost Baseline) based on a sticky hash-based routing split.
    """
    start_time = time.time()
    correlation_id = getattr(request.state, "correlation_id", "unknown")

    ab_group = ab_engine.get_route(payload.TransactionID)
    model_used = (
        "GraphSAGE+GAN (GNN)" if ab_group == "A" else "XGBoost Baseline (Tabular)"
    )

    try:
        if ab_group == "A":
            sanitized_vesta = np.clip(payload.vesta_features, -10.0, 10.0).tolist()
            input_vector = np.array(
                sanitized_vesta
                + [payload.TransactionAmt, payload.C1, payload.C2, payload.D1]
                + [1.0, 1.0],
                dtype=np.float32,
            ).reshape(1, -1)

            mock_edges = np.zeros((2, 1), dtype=np.int64)
            fraud_probability = predictor.predict(input_vector, mock_edges)
        else:
            # Group B (XGBoost)
            if ab_engine.xgb_model is not None:
                sanitized_vesta = np.clip(payload.vesta_features, -10.0, 10.0).tolist()
                input_vector = np.array(
                    sanitized_vesta
                    + [payload.TransactionAmt, payload.C1, payload.C2, payload.D1]
                    + [1.0, 1.0],
                    dtype=np.float32,
                ).reshape(1, -1)

                probs = ab_engine.xgb_model.predict_proba(input_vector)
                fraud_probability = float(probs[0, 1])
            else:
                fraud_probability = ab_engine.predict_xgb_simulated(
                    payload.TransactionAmt, payload.C2
                )

        is_fraud = bool(fraud_probability >= DECISION_THRESHOLD)
        latency_ms = (time.time() - start_time) * 1000.0

        # Update telemetry
        ab_engine.record_metrics(ab_group, is_fraud, latency_ms)

        logger.info(
            f"TxID: {payload.TransactionID} scored via Group {ab_group} ({model_used}). Prob: {fraud_probability:.4f}",
            extra={"correlation_id": correlation_id},
        )

        return ABPredictionResponse(
            transaction_id=payload.TransactionID,
            ab_group=ab_group,
            model_used=model_used,
            fraud_score=fraud_probability,
            is_fraudulent=is_fraud,
            processing_latency_ms=round(latency_ms, 2),
        )
    except Exception as e:
        logger.error(
            f"A/B prediction failed: {str(e)}", extra={"correlation_id": correlation_id}
        )
        raise HTTPException(
            status_code=500, detail="Internal A/B inference engine failure."
        )


@app.get("/api/v1/ab_status")
async def get_ab_status(_api_key: str = Depends(verify_api_key)):
    """
    Exposes A/B test routing statistics and two-proportion Z-test significance.
    """
    return ab_engine.get_status_report()


@app.get("/health")
async def health_check():
    return {"status": "healthy", "engine_loaded": PREDICTOR is not None}
