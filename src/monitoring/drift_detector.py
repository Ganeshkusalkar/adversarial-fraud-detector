import json
import logging
import numpy as np
import os
from typing import Dict, List

logger = logging.getLogger("DriftDetector")

class DriftDetector:
    def __init__(self, reference_path="artifacts/production/reference_stats.json"):
        self.reference_path = reference_path
        self.reference_stats = {}
        self.load_reference()
        
    def load_reference(self):
        if os.path.exists(self.reference_path):
            with open(self.reference_path, "r") as f:
                self.reference_stats = json.load(f)
        else:
            logger.warning("Reference stats not found. Drift detection disabled.")
            
    def _calculate_psi(self, expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
        """Calculate the Population Stability Index (PSI) between two distributions."""
        if len(expected) == 0 or len(actual) == 0:
            return 0.0
            
        breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
        # Ensure breakpoints are unique
        breakpoints = np.unique(breakpoints)
        if len(breakpoints) < 2:
            return 0.0

        expected_percents = np.histogram(expected, breakpoints)[0] / len(expected)
        actual_percents = np.histogram(actual, breakpoints)[0] / len(actual)
        
        # Avoid division by zero
        expected_percents = np.maximum(expected_percents, 0.0001)
        actual_percents = np.maximum(actual_percents, 0.0001)
        
        psi = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
        return float(psi)

    def check_drift(self, live_batch: np.ndarray, feature_names: List[str]) -> Dict[str, float]:
        """Compute PSI for top features."""
        if not self.reference_stats:
            return {}
            
        drift_report = {}
        
        # Only check features we have reference stats for
        for i, feat_name in enumerate(feature_names):
            if feat_name in self.reference_stats:
                expected_dist = np.array(self.reference_stats[feat_name]['distribution'])
                actual_dist = live_batch[:, i]
                
                psi = self._calculate_psi(expected_dist, actual_dist)
                drift_report[feat_name] = round(psi, 4)
                
                if psi > 0.25:
                    logger.critical(f"Severe Data Drift Detected on {feat_name}! PSI: {psi:.4f}")
                elif psi > 0.1:
                    logger.warning(f"Moderate Data Drift on {feat_name}. PSI: {psi:.4f}")
                    
        return drift_report

drift_detector = DriftDetector()
