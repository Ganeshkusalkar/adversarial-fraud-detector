import time
from fastapi import APIRouter
from .predict import LATENCY_HISTOGRAM, TPS_COUNTER

router = APIRouter(prefix="/monitoring", tags=["Monitoring"])

START_TIME = time.time()

@router.get("/health")
def health_check():
    """
    Standard API gateway health probe endpoint.
    """
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "uptime_seconds": round(time.time() - START_TIME, 2)
    }

@router.get("/metrics")
def get_metrics():
    """
    Exposes metrics for Prometheus scraper/Streamlit health tracking dashboard.
    """
    avg_latency = float(sum(LATENCY_HISTOGRAM) / len(LATENCY_HISTOGRAM)) if LATENCY_HISTOGRAM else 0.0
    p95_latency = float(np.percentile(LATENCY_HISTOGRAM, 95)) if LATENCY_HISTOGRAM else 0.0
    
    # Simple uptime-based TPS estimation
    uptime = time.time() - START_TIME
    estimated_tps = TPS_COUNTER / max(uptime, 1.0)
    
    return {
        "active_tps": round(estimated_tps, 2),
        "total_requests": TPS_COUNTER,
        "latency": {
            "mean_ms": round(avg_latency, 2),
            "p95_ms": round(p95_latency, 2),
            "sla_violation_rate": round(sum(1 for x in LATENCY_HISTOGRAM if x > 50.0) / max(len(LATENCY_HISTOGRAM), 1) * 100.0, 2)
        }
    }

try:
    import numpy as np
except ImportError:
    class DummyNP:
        @staticmethod
        def percentile(a, q):
            if not a:
                return 0.0
            sorted_a = sorted(a)
            idx = int(len(sorted_a) * (q / 100.0))
            return sorted_a[min(idx, len(sorted_a) - 1)]
    np = DummyNP()
