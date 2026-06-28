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
from api.schemas import TransactionInput, FraudPredictionResponse
from src.utils.logger import setup_logger
from src.evaluation.calibrated_predictor import CalibratedFraudPredictor
from monitoring.metrics import FRAUD_PREDICTIONS_TOTAL, INFERENCE_LATENCY_SECONDS, FRAUD_SCORE_DISTRIBUTION, MODEL_ERRORS_TOTAL
from src.monitoring.drift_detector import drift_detector
from monitoring.alerting_rules import alerting_engine

with open("config/base_config.yaml", "r") as f:
    config = yaml.safe_load(f)

DECISION_THRESHOLD = float(os.getenv("FRAUD_DECISION_THRESHOLD", config.get("training", {}).get("decision_threshold", 0.50)))

logger = setup_logger("APIServingGateway")

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="Adversarial Transaction Disguise Detector API",
    description="FAANG-standard high-throughput inductive GNN fraud detection microservice micro-engine.",
    version="1.0.0"
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

PREDICTOR = None
MODEL_PATH = os.getenv("ONNX_MODEL_PATH", "artifacts/production/fraud_model.onnx")
SCALER_PATH = os.getenv("SCALER_PATH", "artifacts/production/scaler.pkl")
CALIBRATOR_PATH = os.getenv("CALIBRATOR_PATH", "artifacts/production/prob_calibrator.pkl")

# Metrics Endpoint
from prometheus_client import make_asgi_app
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

@app.middleware("http")
async def add_correlation_id(request: Request, call_next):
    correlation_id = str(uuid.uuid4())
    request.state.correlation_id = correlation_id
    response = await call_next(request)
    response.headers["X-Correlation-ID"] = correlation_id
    return response

from src.evaluation.shap_explainer import FraudSHAPExplainer

@app.on_event("startup")
def load_production_artifacts():
    global PREDICTOR, SHAP_EXPLAINER
    try:
        logger.info(f"Loading calibrated execution runtime graph configurations from target path: {MODEL_PATH}")
        PREDICTOR = CalibratedFraudPredictor(
            model_path=MODEL_PATH,
            scaler_path=SCALER_PATH,
            calibrator_path=CALIBRATOR_PATH
        )
        # Initialize SHAP explainer
        SHAP_EXPLAINER = FraudSHAPExplainer(PREDICTOR)
        
        logger.info("Calibrated Fraud Predictor runtime established in memory.")
        logger.info(f"Configured DECISION_THRESHOLD: {DECISION_THRESHOLD}")
    except Exception as e:
        logger.error(f"Critical System Failure during loading phase: {str(e)}")
        PREDICTOR = None
        SHAP_EXPLAINER = None

def get_predictor():
    if PREDICTOR is None:
        raise HTTPException(status_code=503, detail="Calibrated Predictor runtime uninitialized.")
    return PREDICTOR

def get_explainer():
    if SHAP_EXPLAINER is None:
        raise HTTPException(status_code=503, detail="SHAP Explainer runtime uninitialized.")
    return SHAP_EXPLAINER

@app.post("/api/v1/predict", response_model=FraudPredictionResponse)
@limiter.limit("100/minute")
async def predict_transaction_risk(request: Request, payload: TransactionInput, predictor: CalibratedFraudPredictor = Depends(get_predictor)):
    start_time = time.time()
    correlation_id = getattr(request.state, "correlation_id", "unknown")
    
    try:
        # Input Sanitization
        sanitized_vesta = np.clip(payload.vesta_features, -10.0, 10.0).tolist()
        
        # Step 1: Process and vector-align the unstructured payload input properties
        input_vector = np.array(
            sanitized_vesta + [payload.TransactionAmt, payload.C1, payload.C2, payload.D1] + [1.0, 1.0], 
            dtype=np.float32
        ).reshape(1, -1) 
        
        # In production environments, mock graph connectivity structures are passed to simulate identity mapping paths
        mock_edges = np.zeros((2, 1), dtype=np.int64) 
        
        # Step 2: Execute inference and calibration layer
        fraud_probability = predictor.predict(input_vector, mock_edges)
        is_fraud = bool(fraud_probability >= DECISION_THRESHOLD)
        
        # Step 3: Metrics & Alerting Update
        FRAUD_PREDICTIONS_TOTAL.labels(flagged=str(is_fraud)).inc()
        FRAUD_SCORE_DISTRIBUTION.observe(fraud_probability)
        alerting_engine.record_prediction(is_fraud)
        
        latency = (time.time() - start_time)
        INFERENCE_LATENCY_SECONDS.observe(latency)
        alerting_engine.record_latency(latency * 1000.0)
        
        logger.info(f"TxID: {payload.TransactionID} scored securely in {latency*1000.0:.2f}ms. Calibrated Prob: {fraud_probability:.4f}", extra={"correlation_id": correlation_id})
        
        return FraudPredictionResponse(
            transaction_id=payload.TransactionID,
            fraud_score=fraud_probability,
            is_fraudulent=is_fraud, 
            processing_latency_ms=round(latency * 1000.0, 2)
        )
    except Exception as e:
        MODEL_ERRORS_TOTAL.labels(error_type=type(e).__name__).inc()
        alerting_engine.record_error()
        logger.error(f"Prediction failed: {str(e)}", extra={"correlation_id": correlation_id})
        raise HTTPException(status_code=500, detail="Internal inference engine failure.")

@app.post("/api/v1/explain")
@limiter.limit("10/minute")
async def explain_transaction(request: Request, payload: TransactionInput, explainer: FraudSHAPExplainer = Depends(get_explainer)):
    try:
        sanitized_vesta = np.clip(payload.vesta_features, -10.0, 10.0).tolist()
        input_vector = np.array(
            sanitized_vesta + [payload.TransactionAmt, payload.C1, payload.C2, payload.D1] + [1.0, 1.0], 
            dtype=np.float32
        )
        
        feature_names = [f"Vesta_{i}" for i in range(339)] + ["TransactionAmt", "C1", "C2", "D1", "node_degree", "node_centrality"]
        explanation = explainer.explain(input_vector, feature_names)
        
        return {
            "transaction_id": payload.TransactionID,
            "explanation": explanation
        }
    except Exception as e:
        logger.error(f"Explanation failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal SHAP explainer failure.")

@app.get("/monitoring/drift")
async def check_drift():
    try:
        # Just use some random mock batch for API test
        mock_live_batch = np.random.randn(100, 345)
        feature_names = [f"V{i}" for i in range(339)] + ["TransactionAmt", "C1", "C2", "D1", "node_degree", "node_centrality"]
        drift_report = drift_detector.check_drift(mock_live_batch, feature_names)
        return {"status": "success", "drift_report": drift_report}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.get("/health")
async def health_check():
    """
    Exposes an infrastructure monitoring probe interface.
    """
    return {"status": "healthy", "engine_loaded": PREDICTOR is not None}