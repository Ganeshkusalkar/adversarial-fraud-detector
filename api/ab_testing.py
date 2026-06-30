import os
import hashlib
import time
import math
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

# Try to import xgboost for Group B baseline
try:
    import xgboost as xgb

    HAS_XGB = True
except ImportError:
    HAS_XGB = False


def normal_cdf(z: float) -> float:
    """
    Cumulative distribution function for standard normal distribution.
    Uses approximation math.erf to remain dependency-free.
    """
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def calculate_z_test(n_a: int, x_a: int, n_b: int, x_b: int) -> Dict[str, Any]:
    """
    Computes a two-proportion Z-test for statistical significance.
    """
    if n_a < 10 or n_b < 10:
        return {
            "z_score": 0.0,
            "p_value": 1.0,
            "significant": False,
            "message": "Insufficient data (minimum 10 requests per group required).",
        }

    p_a = x_a / n_a
    p_b = x_b / n_b

    # Pooled proportion
    p_pooled = (x_a + x_b) / (n_a + n_b)

    if p_pooled <= 0.0 or p_pooled >= 1.0:
        return {
            "z_score": 0.0,
            "p_value": 1.0,
            "significant": False,
            "message": "Zero variance in proportion groups.",
        }

    # Standard error
    se = math.sqrt(p_pooled * (1.0 - p_pooled) * (1.0 / n_a + 1.0 / n_b))

    if se == 0.0:
        return {
            "z_score": 0.0,
            "p_value": 1.0,
            "significant": False,
            "message": "Division by zero in standard error.",
        }

    z = (p_a - p_b) / se
    p_value = 2.0 * (1.0 - normal_cdf(abs(z)))

    return {
        "z_score": round(z, 4),
        "p_value": round(p_value, 4),
        "significant": p_value < 0.05,
        "message": (
            "Significant (p < 0.05)"
            if p_value < 0.05
            else "Not statistically significant."
        ),
    }


class ABTestingEngine:
    """
    Manages client routing and registers telemetry metrics for A/B split tests.
    Sticky-routes transactions by hashing the Transaction ID.
    """

    def __init__(self, ab_split: float = 0.5):
        self.ab_split = ab_split
        self.xgb_model = None

        # Load production XGBoost baseline model if available
        if HAS_XGB:
            model_path = os.getenv(
                "XGB_MODEL_PATH", "artifacts/production/xgboost_model.json"
            )
            if os.path.exists(model_path):
                try:
                    self.xgb_model = xgb.XGBClassifier()
                    self.xgb_model.load_model(model_path)
                    logger.info(f"Loaded XGBoost baseline model from {model_path}")
                except Exception as e:
                    logger.error(f"Failed to load XGBoost baseline model: {e}")
            else:
                logger.warning(
                    "XGBoost model file not found. Running in simulated fallback mode."
                )

        # Performance accumulators
        self.requests_a = 0
        self.requests_b = 0
        self.fraud_a = 0
        self.fraud_b = 0
        self.latencies_a: List[float] = []
        self.latencies_b: List[float] = []

    def get_route(self, transaction_id: str) -> str:
        """
        Deterministic sticky routing based on transaction ID hashing.
        Returns "A" (GraphSAGE+GAN) or "B" (XGBoost Tabular Baseline).
        """
        hasher = hashlib.md5(str(transaction_id).encode("utf-8"))
        norm_val = int(hasher.hexdigest(), 16) % 10000 / 10000.0
        return "A" if norm_val < self.ab_split else "B"

    def predict_xgb_simulated(self, transaction_amt: float, c2: float) -> float:
        """
        Simulated fallback prediction for XGBoost baseline model.
        Tabular models are blind to graph structures and heavily rely on transaction amounts.
        """
        # Linear approximation for risk based on simple features
        risk = 0.05
        if transaction_amt > 2000.0:
            risk += 0.15
        if transaction_amt > 10000.0:
            risk += 0.25
        if c2 > 2.0:
            risk += 0.20
        return min(max(risk, 0.0), 1.0)

    def record_metrics(self, group: str, is_fraud: bool, latency_ms: float) -> None:
        """
        Updates live metrics arrays for real-time statistical evaluation.
        """
        if group == "A":
            self.requests_a += 1
            if is_fraud:
                self.fraud_a += 1
            self.latencies_a.append(latency_ms)
            if len(self.latencies_a) > 1000:
                self.latencies_a.pop(0)
        else:
            self.requests_b += 1
            if is_fraud:
                self.fraud_b += 1
            self.latencies_b.append(latency_ms)
            if len(self.latencies_b) > 1000:
                self.latencies_b.pop(0)

    def get_status_report(self) -> Dict[str, Any]:
        """
        Compiles structural performance metrics and Z-test evaluation report.
        """
        avg_lat_a = (
            sum(self.latencies_a) / len(self.latencies_a) if self.latencies_a else 0.0
        )
        avg_lat_b = (
            sum(self.latencies_b) / len(self.latencies_b) if self.latencies_b else 0.0
        )

        p_a = self.fraud_a / self.requests_a if self.requests_a > 0 else 0.0
        p_b = self.fraud_b / self.requests_b if self.requests_b > 0 else 0.0

        z_stats = calculate_z_test(
            self.requests_a, self.fraud_a, self.requests_b, self.fraud_b
        )

        return {
            "group_a": {
                "model_name": "GraphSAGE+GAN (GNN)",
                "total_requests": self.requests_a,
                "fraud_detected": self.fraud_a,
                "fraud_rate": round(p_a, 4),
                "avg_latency_ms": round(avg_lat_a, 2),
            },
            "group_b": {
                "model_name": "XGBoost Baseline (Tabular)",
                "total_requests": self.requests_b,
                "fraud_detected": self.fraud_b,
                "fraud_rate": round(p_b, 4),
                "avg_latency_ms": round(avg_lat_b, 2),
            },
            "z_test": z_stats,
            "timestamp": time.time(),
        }
