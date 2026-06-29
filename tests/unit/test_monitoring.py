import pytest
import numpy as np
import pandas as pd
from src.monitoring.drift_detection import DataDriftDetector


def test_drift_detector_no_drift():
    """Test that the drift detector does not flag drift when distributions match."""
    np.random.seed(42)
    ref_data = pd.DataFrame({"feat1": np.random.normal(0, 1, 1000)})
    prod_data = pd.DataFrame({"feat1": np.random.normal(0, 1, 500)})

    detector = DataDriftDetector(ref_data, psi_threshold=0.2)
    results = detector.check_drift(prod_data)

    assert not results["drift_detected"]
    assert results["features"]["feat1"]["psi_value"] <= 0.2


def test_drift_detector_with_drift():
    """Test that the drift detector flags drift when distributions diverge significantly."""
    np.random.seed(42)
    ref_data = pd.DataFrame({"feat1": np.random.normal(0, 1, 1000)})
    # Prod data shifted mean and variance
    prod_data = pd.DataFrame({"feat1": np.random.normal(5, 2, 500)})

    detector = DataDriftDetector(ref_data, psi_threshold=0.2)
    results = detector.check_drift(prod_data)

    assert results["drift_detected"]
    assert results["features"]["feat1"]["psi_value"] > 0.2
    assert results["features"]["feat1"]["is_drifting"]
