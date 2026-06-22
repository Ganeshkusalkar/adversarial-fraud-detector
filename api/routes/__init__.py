"""
FastAPI Routes for prediction and service monitoring.
"""

from .predict import router as predict_router
from .monitoring import router as monitoring_router

__all__ = ["predict_router", "monitoring_router"]
