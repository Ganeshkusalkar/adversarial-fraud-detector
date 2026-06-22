import os
import yaml
import time
import numpy as np
import uuid
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from api.schemas import TransactionInput, FraudPredictionResponse
from src.utils.logger import setup_logger, correlation_id_var
from src.evaluation.calibrated_predictor import CalibratedFraudPredictor
from monitoring.metrics import metrics_router, fraud_predictions_total, inference_latency_seconds, fraud_score_distribution, model_errors_total
from monitoring.alerting_rules import alerter

with open("config/base_config.yaml", "r") as f:
    config = yaml.safe_load(f)
DECISION_THRESHOLD = config.get("training", {}).get("decision_threshold", 0.50)

logger = setup_logger("APIServingGateway")

app = FastAPI(
    title="Adversarial Transaction Disguise Detector API",
    description="FAANG-standard high-throughput inductive GNN fraud detection microservice micro-engine.",
    version="1.0.0"
)

app.include_router(metrics_router)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Validation error: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={"error_code": "VALIDATION_ERROR", "message": "Invalid transaction payload", "details": exc.errors()}
    )

@app.middleware("http")
async def add_correlation_id(request: Request, call_next):
    correlation_id = str(uuid.uuid4())
    correlation_id_var.set(correlation_id)
    response = await call_next(request)
    response.headers["X-Correlation-ID"] = correlation_id
    return response

PREDICTOR = None
MODEL_PATH = os.environ.get("ONNX_MODEL_PATH", "artifacts/production/fraud_model.onnx")
SCALER_PATH = os.environ.get("SCALER_PATH", "artifacts/production/scaler.pkl")
CALIBRATOR_PATH = os.environ.get("CALIBRATOR_PATH", "artifacts/production/prob_calibrator.pkl")
MAX_REQ = os.environ.get("MAX_REQUESTS_PER_MINUTE", "100")

@app.on_event("startup")
def load_production_artifacts():
    global PREDICTOR
    try:
        # Check model registry version
        active_version = "unknown"
        active_version_file = "artifacts/registry/active_version.json"
        if os.path.exists(active_version_file):
            import json
            with open(active_version_file, "r") as f:
                active_version = json.load(f).get("current_version", "unknown")
                
        logger.info(f"Loading calibrated execution runtime graph configurations from target path: {MODEL_PATH}")
        logger.info(f"Active Model Registry Version: v{active_version}")
        PREDICTOR = CalibratedFraudPredictor(
            model_path=MODEL_PATH,
            scaler_path=SCALER_PATH,
            calibrator_path=CALIBRATOR_PATH
        )
        logger.info("Calibrated Fraud Predictor runtime established in memory.")
        logger.info(f"Configured DECISION_THRESHOLD: {DECISION_THRESHOLD}")
    except Exception as e:
        logger.error(f"Critical System Failure during loading phase: {str(e)}")
        PREDICTOR = None

def get_predictor():
    if PREDICTOR is None:
        raise HTTPException(status_code=503, detail="Calibrated Predictor runtime uninitialized.")
    return PREDICTOR

from src.monitoring.drift_detector import drift_detector

DRIFT_CHECK_INTERVAL = int(os.environ.get("DRIFT_CHECK_INTERVAL", "1000"))
drift_counter = 0

@app.get("/monitoring/drift")
async def get_drift_metrics():
    """
    Exposes latest Population Stability Index (PSI) drift metrics.
    """
    return {"status": "success", "psi": drift_detector.latest_psi}

@app.post("/api/v1/predict", response_model=FraudPredictionResponse)
@limiter.limit(f"{MAX_REQ}/minute")
async def predict_transaction_risk(request: Request, payload: TransactionInput, predictor: CalibratedFraudPredictor = Depends(get_predictor)):
    global drift_counter
    start_time = time.time()
    
    try:
        raw_vector = np.array(
            payload.vesta_features + [payload.TransactionAmt, payload.C1, payload.C2, payload.D1] + [1.0, 1.0], 
            dtype=np.float32
        ).reshape(1, -1) 

        input_vector = np.clip(raw_vector, -10.0, 10.0)
        
        # Record features for drift
        drift_detector.record_features(input_vector)
        drift_counter += 1
        if drift_counter >= DRIFT_CHECK_INTERVAL:
            drift_detector.calculate_drift()
            drift_counter = 0
            
        try:
            if hasattr(predictor.scaler, 'mean_') and hasattr(predictor.scaler, 'scale_'):
                z_scores = (raw_vector - predictor.scaler.mean_) / predictor.scaler.scale_
                if np.any(np.abs(z_scores) > 5.0):
                    logger.warning("Data Drift Alert: One or more features exceed 5 standard deviations from training mean.")
        except Exception as e:
            logger.warning(f"Could not compute z-scores for input sanitization: {e}")
        
        mock_edges = np.zeros((2, 1), dtype=np.int64) 
        
        fraud_probability = predictor.predict(input_vector, mock_edges)
        flagged = fraud_probability >= DECISION_THRESHOLD
        
        latency = (time.time() - start_time) * 1000.0 
        logger.info(f"TxID: {payload.TransactionID} scored securely in {latency:.2f}ms. Calibrated Prob: {fraud_probability:.4f}", extra={"transaction_id": payload.TransactionID, "fraud_score": fraud_probability, "flagged": flagged, "latency_ms": latency, "threshold_used": DECISION_THRESHOLD})
        
        # Prometheus Metrics
        fraud_predictions_total.labels(flagged=str(flagged).lower()).inc()
        inference_latency_seconds.observe(latency / 1000.0)
        fraud_score_distribution.observe(fraud_probability)
        
        # Alerting Rules
        alerter.record_prediction(flagged, latency)
        
        return FraudPredictionResponse(
            transaction_id=payload.TransactionID,
            fraud_score=fraud_probability,
            is_fraudulent=flagged, 
            processing_latency_ms=round(latency, 2)
        )
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        model_errors_total.labels(error_type=type(e).__name__).inc()
        alerter.record_error()
        raise HTTPException(status_code=500, detail="Internal inference engine failure.")

@app.get("/health")
async def health_check():
    """
    Exposes an infrastructure monitoring probe interface.
    """
    return {"status": "healthy", "engine_loaded": PREDICTOR is not None}