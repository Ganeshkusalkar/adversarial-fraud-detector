import logging
import os
from pathlib import Path
from typing import Generator, Optional

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

logger = logging.getLogger(__name__)

_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def _get_valid_keys() -> set:
    """
    Reads the comma-separated list of valid API keys from the API_KEYS env var.
    """
    raw = os.getenv("API_KEYS", "")
    if not raw:
        logger.warning(
            "API_KEYS env var is not set. All API key checks will fail."
        )
        return set()
    keys = {k.strip() for k in raw.split(",") if k.strip()}
    logger.info(f"API key authentication active: {len(keys)} key(s) configured.")
    return keys


def verify_api_key(api_key: Optional[str] = Security(_API_KEY_HEADER)) -> str:
    """
    Validates the X-API-Key request header.
    """
    valid_keys = _get_valid_keys()
    if not api_key or api_key not in valid_keys:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing API key. Provide a valid key in the X-API-Key header.",
        )
    return api_key


try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False
    logger.warning("onnxruntime is not installed. Running in simulated mode.")


class ONNXInferenceSession:
    """
    Wrapper around ONNX runtime InferenceSession.
    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.session = None
        self._load_session()

    def _load_session(self):
        if not HAS_ONNX:
            logger.warning("ONNX runtime unavailable. Predictions will be simulated.")
            return

        if not Path(self.model_path).exists():
            logger.error(f"ONNX model not found at {self.model_path}.")
            return

        try:
            self.session = ort.InferenceSession(
                self.model_path, providers=["CPUExecutionProvider"]
            )
            logger.info(f"Loaded ONNX model session from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load ONNX model session: {e}")

    def predict(self, node_features, edge_index):
        if self.session is None:
            # Simulated inference logic: simple deterministic hashing for mock testing
            import numpy as np
            avg_val = float(np.mean(node_features)) if len(node_features) > 0 else 0.1
            prob = min(max(abs(avg_val), 0.0), 1.0)
            return prob

        inputs = {"node_features": node_features, "edge_index": edge_index}
        outputs = self.session.run(None, inputs)
        logits = outputs[0]
        
        import numpy as np
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        return float(probs[0, 1])


# Singleton for ONNX Session
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
