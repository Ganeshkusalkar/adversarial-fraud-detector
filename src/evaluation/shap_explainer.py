import shap
import numpy as np
import logging
from src.evaluation.calibrated_predictor import CalibratedFraudPredictor

logger = logging.getLogger("SHAPExplainer")


class FraudSHAPExplainer:
    def __init__(
        self, predictor: CalibratedFraudPredictor, background_data: np.ndarray = None
    ):
        """
        Initializes the SHAP Explainer.
        Uses KernelExplainer since we are treating the GNN + Calibrator as a black box.

        Args:
            predictor: The CalibratedFraudPredictor instance used in production.
            background_data: A representative sample of background transactions (shape: N x 345).
        """
        self.predictor = predictor

        if background_data is None:
            # Generate a mock background dataset (zeroed or random) if none provided
            logger.warning(
                "No background data provided for SHAP. Using synthetic zero-baseline."
            )
            background_data = np.zeros((10, 345), dtype=np.float32)

        # Wrapping the predict function to handle batched inputs required by SHAP
        self.explainer = shap.KernelExplainer(self._predict_wrapper, background_data)

    def _predict_wrapper(self, x_batch: np.ndarray) -> np.ndarray:
        """
        Wrapper to feed batched data into the single-transaction predictor.
        In a real batched inference scenario, this would use a batched ONNX session.
        """
        mock_edges = np.zeros((2, 1), dtype=np.int64)
        predictions = []
        for x in x_batch:
            # Reshape to 1 x 345
            x_input = x.reshape(1, -1).astype(np.float32)
            prob = self.predictor.predict(x_input, mock_edges)
            predictions.append(prob)
        return np.array(predictions)

    def explain(self, x_instance: np.ndarray, feature_names: list = None) -> dict:
        """
        Generates SHAP values for a single transaction instance.
        """
        if x_instance.ndim == 1:
            x_instance = x_instance.reshape(1, -1)

        shap_values = self.explainer.shap_values(x_instance, nsamples=100)

        # If output is a list (e.g. multi-class), grab the first element
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        shap_values = shap_values.flatten()

        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(len(shap_values))]

        # Pair feature names with their absolute SHAP values and sort
        importance = [
            {"feature": name, "shap_value": float(val), "abs_impact": float(abs(val))}
            for name, val in zip(feature_names, shap_values)
        ]

        # Sort by absolute impact descending
        importance.sort(key=lambda x: x["abs_impact"], reverse=True)

        return {
            "top_features": importance[:10],  # Return top 10 contributors
            "base_value": float(self.explainer.expected_value),
        }
