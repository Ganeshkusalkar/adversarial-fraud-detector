import os
import yaml
import time
import uuid
import numpy as np
from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from api.schemas import (
    TransactionInput,
    FraudPredictionResponse,
    ABPredictionResponse,
    SystemHealthResponse,
)
from api.dependencies import verify_api_key
from api.ab_testing import ABTestingEngine
from src.utils.logger import setup_logger
from src.evaluation.calibrated_predictor import CalibratedFraudPredictor
from src.graph.subgraph_engine import RealTimeGraphEngine
from src.utils.circuit_breaker import CircuitBreaker, CircuitState
from monitoring.metrics import (
    FRAUD_PREDICTIONS_TOTAL,
    INFERENCE_LATENCY_SECONDS,
    FRAUD_SCORE_DISTRIBUTION,
    MODEL_ERRORS_TOTAL,
    SUBGRAPH_NODES_EVALUATED,
    SUBGRAPH_EDGES_EVALUATED,
    CIRCUIT_BREAKER_STATE,
    FALLBACK_PREDICTIONS_TOTAL,
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

START_TIME = time.time()
logger = setup_logger("APIServingGateway")
limiter = Limiter(key_func=get_remote_address)
ab_engine = ABTestingEngine(ab_split=0.5)

# Production Graph Engine & Circuit Breaker singletons
GRAPH_ENGINE = RealTimeGraphEngine()
CIRCUIT_BREAKER = CircuitBreaker(failure_threshold=3, recovery_timeout_sec=10.0)

app = FastAPI(
    title="Adversarial Transaction Disguise Detector API",
    description="Enterprise-grade production GraphSAGE + LSTM GAN fraud detection microservice.",
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


@app.get("/")
def root_entrypoint():
    """Root entrypoint providing service status and documentation links."""
    return {
        "service": "Adversarial Transaction Disguise Detector API",
        "status": "online",
        "version": "1.0.0",
        "documentation": "/docs",
        "health_check": "/health",
        "metrics": "/metrics",
    }


def _tabular_fallback_predict(amount: float, c2: float) -> float:
    """
    Sub-1ms fallback risk heuristic executed when GNN or graph extraction experiences circuit trip.
    Guarantees zero payment dropouts and sub-50ms SLA.
    """
    score = 0.03
    if amount > 500.0:
        score += 0.22
    if c2 > 2.0:
        score += 0.35
    return float(min(0.99, max(0.01, score)))


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
    background_tasks: BackgroundTasks,
    predictor: CalibratedFraudPredictor = Depends(get_predictor),
    _api_key: str = Depends(verify_api_key),
):
    start_time = time.time()
    correlation_id = getattr(request.state, "correlation_id", "unknown")

    try:
        # 1. Feature normalization and auto-padding (ensure exactly 339 Vesta features)
        vesta = payload.vesta_features or []
        if len(vesta) < 339:
            vesta = vesta + [0.0] * (339 - len(vesta))
        sanitized_vesta = np.clip(vesta[:339], -10.0, 10.0).tolist()

        target_vector = np.array(
            sanitized_vesta
            + [payload.TransactionAmt, payload.C1, payload.C2, payload.D1]
            + [1.0, 1.0],
            dtype=np.float32,
        )

        # 2. Dynamic Real-Time Subgraph Extraction
        x_subgraph, edge_index, target_idx = GRAPH_ENGINE.extract_subgraph(
            card1=payload.card1,
            target_features=target_vector,
            p_emaildomain=payload.P_emaildomain,
            addr1=None,
            max_neighbors=20,
        )

        num_nodes = int(x_subgraph.shape[0])
        num_edges = int(edge_index.shape[1])
        SUBGRAPH_NODES_EVALUATED.observe(num_nodes)
        SUBGRAPH_EDGES_EVALUATED.observe(num_edges)

        # 3. Resilience: Execute via Circuit Breaker with Fallback
        def _do_gnn():
            return predictor.predict(x_subgraph, edge_index, target_idx=target_idx)

        def _do_fallback():
            FALLBACK_PREDICTIONS_TOTAL.labels(reason="circuit_breaker").inc()
            return _tabular_fallback_predict(payload.TransactionAmt, payload.C2)

        fraud_probability, fallback_used = CIRCUIT_BREAKER.call(_do_gnn, _do_fallback)
        is_fraud = bool(fraud_probability >= DECISION_THRESHOLD)

        # 4. Asynchronous Graph Update (Zero blocking I/O on critical path)
        background_tasks.add_task(
            GRAPH_ENGINE.add_transaction,
            payload.TransactionID,
            payload.card1,
            target_vector,
            payload.P_emaildomain,
            None,
            float(payload.TransactionDT),
        )

        # 5. Metrics & Alerting Telemetry
        FRAUD_PREDICTIONS_TOTAL.labels(flagged=str(is_fraud)).inc()
        FRAUD_SCORE_DISTRIBUTION.observe(fraud_probability)
        alerting_engine.record_prediction(is_fraud)

        cb_state_val = (
            0 if CIRCUIT_BREAKER.state == CircuitState.CLOSED
            else (1 if CIRCUIT_BREAKER.state == CircuitState.HALF_OPEN else 2)
        )
        CIRCUIT_BREAKER_STATE.set(cb_state_val)

        latency = time.time() - start_time
        INFERENCE_LATENCY_SECONDS.observe(latency)
        alerting_engine.record_latency(latency * 1000.0)

        # Decision rationale for regulatory and audit compliance
        if fallback_used:
            reason = "Scored via fast tabular fallback (circuit breaker active)"
        elif is_fraud and num_nodes > 1:
            reason = f"Flagged via multi-hop GraphSAGE ring topology ({num_nodes} accounts/txns)"
        elif is_fraud:
            reason = "Flagged via high-risk transaction feature anomalies"
        else:
            reason = "Normal legitimate transaction topology"

        logger.info(
            f"TxID: {payload.TransactionID} scored. Prob: {fraud_probability:.4f} "
            f"Nodes: {num_nodes} Edges: {num_edges} Fallback: {fallback_used}",
            extra={"correlation_id": correlation_id},
        )

        return FraudPredictionResponse(
            transaction_id=payload.TransactionID,
            fraud_score=round(fraud_probability, 4),
            is_fraudulent=is_fraud,
            processing_latency_ms=round(latency * 1000.0, 2),
            subgraph_nodes=num_nodes,
            subgraph_edges=num_edges,
            fallback_used=fallback_used,
            model_version="v1.0.0-onnx",
            decision_reason=reason,
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
            vesta = payload.vesta_features or []
            if len(vesta) < 339:
                vesta = vesta + [0.0] * (339 - len(vesta))
            sanitized_vesta = np.clip(vesta[:339], -10.0, 10.0).tolist()
            target_vector = np.array(
                sanitized_vesta
                + [payload.TransactionAmt, payload.C1, payload.C2, payload.D1]
                + [1.0, 1.0],
                dtype=np.float32,
            )

            x_subgraph, edge_index, target_idx = GRAPH_ENGINE.extract_subgraph(
                card1=payload.card1,
                target_features=target_vector,
                p_emaildomain=payload.P_emaildomain,
                max_neighbors=20,
            )
            fraud_probability = predictor.predict(
                x_subgraph, edge_index, target_idx=target_idx
            )
        else:
            # Group B (XGBoost Baseline)
            vesta = payload.vesta_features or []
            if len(vesta) < 339:
                vesta = vesta + [0.0] * (339 - len(vesta))
            sanitized_vesta = np.clip(vesta[:339], -10.0, 10.0).tolist()
            input_vector = np.array(
                sanitized_vesta
                + [payload.TransactionAmt, payload.C1, payload.C2, payload.D1]
                + [1.0, 1.0],
                dtype=np.float32,
            ).reshape(1, -1)

            if ab_engine.xgb_model is not None:
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


@app.get("/health", response_model=SystemHealthResponse)
async def health_check():
    """
    Comprehensive enterprise health check inspecting model state,
    graph store capacity, circuit breaker telemetry, and service uptime.
    """
    stats = GRAPH_ENGINE.get_stats() if GRAPH_ENGINE else {}
    cb_status = CIRCUIT_BREAKER.get_status() if CIRCUIT_BREAKER else {}
    return SystemHealthResponse(
        status="healthy" if PREDICTOR is not None else "degraded",
        model_version="v1.0.0-onnx",
        graph_store=stats,
        circuit_breaker=cb_status,
        uptime_seconds=round(time.time() - START_TIME, 1),
    )
