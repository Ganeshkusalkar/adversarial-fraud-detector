import onnxruntime as ort
import pickle
import numpy as np


class CalibratedFraudPredictor:
    """
    Production inference wrapper that loads the ONNX model, the feature scaler,
    and the probability calibrator (Isotonic Regression/Platt Scaling) to yield
    true, trustworthy class probabilities.
    """

    def __init__(self, model_path: str, scaler_path: str, calibrator_path: str):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.calibrator_path = calibrator_path
        self._load_artifacts()

    def _load_artifacts(self):
        # Load ONNX Session
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 3
        self.session = ort.InferenceSession(
            self.model_path,
            providers=["CPUExecutionProvider"],
            sess_options=sess_options,
        )
        self.input_name = self.session.get_inputs()[0].name
        self.edge_name = self.session.get_inputs()[1].name

        # Load Scaler
        with open(self.scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

        # Load Calibrator
        with open(self.calibrator_path, "rb") as f:
            self.calibrator = pickle.load(f)

    def predict(self, input_vector: np.ndarray, edge_index: np.ndarray) -> float:
        """
        Runs the full inference pipeline and returns a calibrated probability.
        """
        # 1. Scale
        scaled_vector = self.scaler.transform(input_vector).astype(np.float32)

        # 2. ONNX Inference
        onnx_inputs = {self.input_name: scaled_vector, self.edge_name: edge_index}
        raw_logits = self.session.run(None, onnx_inputs)[0]

        # 3. Softmax to get uncalibrated probability
        exp_logits = np.exp(raw_logits - np.max(raw_logits))
        probabilities = exp_logits / exp_logits.sum(axis=-1)
        raw_prob = float(probabilities[0][1])

        # 4. Calibrate!
        calibrated_prob_array = self.calibrator.predict_proba(np.array([[raw_prob]]))
        calibrated_prob = float(calibrated_prob_array[0][1])

        return calibrated_prob
