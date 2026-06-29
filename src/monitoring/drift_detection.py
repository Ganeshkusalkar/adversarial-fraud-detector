import numpy as np
import pandas as pd


def calculate_psi(expected, actual, buckets=10):
    """
    Calculate the Population Stability Index (PSI) between two distributions.
    """

    def scale_range(input, min, max):
        input += -(np.min(input))
        input /= np.max(input) / (max - min)
        input += min
        return input

    expected_percents, expected_bins = np.histogram(expected, bins=buckets)
    actual_percents, _ = np.histogram(actual, bins=expected_bins)

    # Avoid zero division
    expected_percents = np.maximum(expected_percents, 1)
    actual_percents = np.maximum(actual_percents, 1)

    expected_fractions = expected_percents / sum(expected_percents)
    actual_fractions = actual_percents / sum(actual_percents)

    psi_value = np.sum(
        (expected_fractions - actual_fractions)
        * np.log(expected_fractions / actual_fractions)
    )
    return psi_value


class DataDriftDetector:
    """
    Detects feature distribution drift in production data compared to training data.
    Uses Population Stability Index (PSI) for continuous features.
    """

    def __init__(self, reference_data: pd.DataFrame, psi_threshold: float = 0.2):
        """
        Args:
            reference_data: DataFrame containing the training data baseline.
            psi_threshold: Threshold above which drift is flagged (e.g., > 0.2 means significant change).
        """
        self.reference_data = reference_data
        self.threshold = psi_threshold
        self.numerical_cols = reference_data.select_dtypes(include=[np.number]).columns

    def check_drift(self, production_data: pd.DataFrame) -> dict:
        """
        Checks for drift in the provided production data batch using PSI.

        Args:
            production_data: DataFrame containing new transactions.

        Returns:
            Dictionary with drift status and PSI values per feature.
        """
        drift_results = {"drift_detected": False, "features": {}}

        for col in self.numerical_cols:
            if col not in production_data.columns:
                continue

            # Perform PSI calculation
            psi_val = calculate_psi(
                self.reference_data[col].dropna().values,
                production_data[col].dropna().values,
            )

            is_drifting = psi_val > self.threshold
            if is_drifting:
                drift_results["drift_detected"] = True

            drift_results["features"][col] = {
                "psi_value": psi_val,
                "is_drifting": is_drifting,
            }

        return drift_results


if __name__ == "__main__":
    # Demo
    np.random.seed(42)
    # Reference data (normal transactions)
    ref_data = pd.DataFrame(
        {
            "amount": np.random.normal(100, 20, 1000),
            "velocity": np.random.normal(1, 0.2, 1000),
        }
    )

    # Production data (adversarial shift in amount)
    prod_data = pd.DataFrame(
        {
            "amount": np.random.normal(150, 30, 500),  # Shifted mean and variance
            "velocity": np.random.normal(1, 0.2, 500),  # Unchanged
        }
    )

    detector = DataDriftDetector(ref_data)
    results = detector.check_drift(prod_data)

    print(f"Drift Detected overall? {results['drift_detected']}")
    for feat, stats in results["features"].items():
        print(
            f"Feature: {feat}, PSI: {stats['psi_value']:.4f}, Drift: {stats['is_drifting']}"
        )
