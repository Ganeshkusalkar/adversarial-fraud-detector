import logging
import os
from pathlib import Path
from typing import Generator, Optional

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# API Key Authentication
# ---------------------------------------------------------------------------

_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def _get_valid_keys() -> set:
    """
    Reads the comma-separated list of valid API keys from the API_KEYS env var.
    Falls back to a development key if the env var is not set, but logs a warning.
    """
    raw = os.getenv("API_KEYS", "")
    if not raw:
        logger.warning(
            "API_KEYS env var is not set. "
            "All API key checks will FAIL. "
            "Set API_KEYS=<your-key> in .env or environment."
        )
        return set()
    keys = {k.strip() for k in raw.split(",") if k.strip()}
    logger.info(f"API key authentication active: {len(keys)} key(s) configured.")
    return keys


def verify_api_key(api_key: Optional[str] = Security(_API_KEY_HEADER)) -> str:
    """
    FastAPI dependency that validates the X-API-Key request header.

    Usage:
        @app.post("/protected")
        async def route(key: str = Depends(verify_api_key)):
            ...

    Raises:
        HTTPException(403): when the key is missing or not in the allowed set.
    """
    valid_keys = _get_valid_keys()
    if not api_key or api_key not in valid_keys:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing API key. Provide a valid key in the X-API-Key header.",
        )
    return api_key


# Fallback for ONNX Runtime
try:
    import onnxruntime as ort

    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False
    logger.warning(
        "onnxruntime is not installed. Running dependencies in simulated environment."
    )


class ONNXInferenceSession:
    """
    Wrapper around ONNX runtime InferenceSession.
    Ensures safe initialization and execution of performance-vetted GNN models.
    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.session = None
        self._load_session()

    def _load_session(self):
        if not HAS_ONNX:
            logger.warning(
                "ONNX runtime unavailable. Model predictions will be simulated."
            )
            return

        if not Path(self.model_path).exists():
            logger.error(
                f"ONNX model weight not found at {self.model_path}. Please compile PyTorch checkpoints first."
            )
            return

        try:
            # Load with CPU Execution Provider as default fallback
            self.session = ort.InferenceSession(
                self.model_path, providers=["CPUExecutionProvider"]
            )
            logger.info(f"Loaded ONNX model session from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load ONNX model session: {e}")

    def predict(self, node_features, edge_index):
        """
        Executes real-time inference on ONNX session.
        """
        if self.session is None:
            # Simulated inference logic: simple deterministic hashing/randomness for mock testing
            import numpy as np

            # Generate dummy probability based on node features
            avg_val = float(np.mean(node_features)) if len(node_features) > 0 else 0.1
            prob = min(max(abs(avg_val), 0.0), 1.0)
            return prob

        # Execute ONNX runtime inference
        inputs = {"node_features": node_features, "edge_index": edge_index}
        outputs = self.session.run(None, inputs)
        # Assuming model outputs logits as first output
        logits = outputs[0]
        # Softmax computation
        import numpy as np

        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        return float(probs[0, 1])  # Class 1 (fraud) probability


# Singleton pattern for ONNX Session Injection
_inference_session = None


def get_onnx_session() -> ONNXInferenceSession:
    """
    Dependency injection helper to obtain the active ONNX inference session.
    """
    global _inference_session
    if _inference_session is None:
        model_path = os.getenv(
            "ONNX_MODEL_PATH", "artifacts/production/discriminator.onnx"
        )
        _inference_session = ONNXInferenceSession(model_path)
    return _inference_session
