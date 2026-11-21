import sys
import pickle
import numpy as np
import onnxruntime as ort
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression


class PlattCalibrator:
    """Platt scaling probability calibrator."""

    def __init__(self):
        self.lr = LogisticRegression(class_weight="balanced")

    def fit(self, X, y):
        self.lr.fit(X.reshape(-1, 1), y)
        return self

    def predict_proba(self, X):
        return self.lr.predict_proba(X.reshape(-1, 1))


class IsoCalibrator:
    """Isotonic regression probability calibrator."""

    def __init__(self):
        self.iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")

    def fit(self, X, y):
        self.iso.fit(X.reshape(-1), y)
        return self

    def predict_proba(self, X):
        probs = self.iso.predict(X.reshape(-1))
        return np.column_stack((1 - probs, probs))


# Register classes in __main__ at import time so unpickling always finds them
for mod_name in ["__main__", "src.evaluation.calibrate_probabilities"]:
    if mod_name in sys.modules:
        setattr(sys.modules[mod_name], "PlattCalibrator", PlattCalibrator)
        setattr(sys.modules[mod_name], "IsoCalibrator", IsoCalibrator)


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

        # Patch __main__ and calibrate_probabilities before unpickling
        main_mod = sys.modules.get("__main__")
        if main_mod:
            setattr(main_mod, "PlattCalibrator", PlattCalibrator)
            setattr(main_mod, "IsoCalibrator", IsoCalibrator)

        # Load Calibrator
        with open(self.calibrator_path, "rb") as f:
            self.calibrator = pickle.load(f)

    def predict(
        self,
        input_vector: np.ndarray,
        edge_index: np.ndarray,
        target_idx: int = 0,
    ) -> float:
        """
        Runs the full inference pipeline and returns a calibrated probability for target_idx.
        Supports multi-node localized subgraphs.
        """
        # 1. Scale node features
        scaled_vector = self.scaler.transform(input_vector).astype(np.float32)

        # 2. ONNX Inference
        onnx_inputs = {self.input_name: scaled_vector, self.edge_name: edge_index}
        raw_logits = self.session.run(None, onnx_inputs)[0]

        # 3. Numerically stable softmax to get uncalibrated probability
        shift_logits = raw_logits - np.max(raw_logits, axis=-1, keepdims=True)
        exp_logits = np.exp(shift_logits)
        probabilities = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        raw_prob = float(probabilities[target_idx][1])

        # 4. Calibrate probability
        calibrated_prob_array = self.calibrator.predict_proba(np.array([[raw_prob]]))
        calibrated_prob = float(calibrated_prob_array[0][1])

        return calibrated_prob
