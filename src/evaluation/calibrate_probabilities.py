import os
import yaml
import onnxruntime as ort
import pickle
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    brier_score_loss,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)
from src.pipelines.ieee_pipeline import IEEECISPipeline
from src.graph.graph_builder import TransactionGraphBuilder

from sklearn.base import BaseEstimator, ClassifierMixin


class DummyEstimator(BaseEstimator, ClassifierMixin):
    """A dummy estimator that just returns the input probabilities for CalibratedClassifierCV."""

    _estimator_type = "classifier"

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        return np.column_stack((1 - X, X))

    def predict(self, X):
        return (X >= 0.5).astype(int)


from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression


class PlattCalibrator:
    def __init__(self):
        self.lr = LogisticRegression(class_weight="balanced")

    def fit(self, X, y):
        self.lr.fit(X.reshape(-1, 1), y)
        return self

    def predict_proba(self, X):
        return self.lr.predict_proba(X.reshape(-1, 1))


class IsoCalibrator:
    def __init__(self):
        self.iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")

    def fit(self, X, y):
        self.iso.fit(X.reshape(-1), y)
        return self

    def predict_proba(self, X):
        probs = self.iso.predict(X.reshape(-1))
        return np.column_stack((1 - probs, probs))


def calibrate():
    with open("config/base_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    tx_path = config["data"]["ieee"]["transaction_path"]
    id_path = config["data"]["ieee"]["identity_path"]

    # Load Validation Set (rows 100,001 - 150,000)
    pipeline_val = IEEECISPipeline(config)

    def load_val():
        df_txn = pd.read_csv(tx_path, skiprows=range(1, 100001), nrows=50000)
        df_id = pd.read_csv(id_path)
        return df_txn, df_id

    pipeline_val.load_raw = load_val
    with open("artifacts/production/label_encoders.pkl", "rb") as f:
        pipeline_val.label_encoders = pickle.load(f)
    processed_df_val = pipeline_val.run_pipeline(fit=False)
    builder_val = TransactionGraphBuilder(config)
    graph_data_val = builder_val.build_inductive_graph(processed_df_val)

    with open("artifacts/production/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    x_scaled_val = scaler.transform(graph_data_val.x.numpy()).astype(np.float32)
    y_true_val = graph_data_val.y.numpy()

    # Load Test Set (rows 150,001 - 200,000)
    pipeline_test = IEEECISPipeline(config)

    def load_test():
        df_txn = pd.read_csv(tx_path, skiprows=range(1, 150001), nrows=50000)
        df_id = pd.read_csv(id_path)
        return df_txn, df_id

    pipeline_test.load_raw = load_test
    with open("artifacts/production/label_encoders.pkl", "rb") as f:
        pipeline_test.label_encoders = pickle.load(f)
    processed_df_test = pipeline_test.run_pipeline(fit=False)
    builder_test = TransactionGraphBuilder(config)
    graph_data_test = builder_test.build_inductive_graph(processed_df_test)
    x_scaled_test = scaler.transform(graph_data_test.x.numpy()).astype(np.float32)
    y_true_test = graph_data_test.y.numpy()

    # Inference Session
    sess_options = ort.SessionOptions()
    sess_options.log_severity_level = 3
    session = ort.InferenceSession(
        "artifacts/production/fraud_model.onnx", sess_options=sess_options
    )
    input_name = session.get_inputs()[0].name
    edge_name = session.get_inputs()[1].name

    def get_raw_probs(x_scaled, y_true):
        batch_size = 256
        fraud_probs = []
        n_nodes = len(y_true)
        for start in range(0, n_nodes, batch_size):
            end = min(start + batch_size, n_nodes)
            batch_x = x_scaled[start:end]
            batch_len = end - start
            local_edges = np.array(
                [list(range(batch_len)), list(range(batch_len))], dtype=np.int64
            )
            raw_logits = session.run(
                None, {input_name: batch_x, edge_name: local_edges}
            )[0]
            logits_s = raw_logits - raw_logits.max(axis=-1, keepdims=True)
            exp_l = np.exp(logits_s)
            probs = exp_l / exp_l.sum(axis=-1, keepdims=True)
            fraud_probs.extend(probs[:, 1].tolist())
        return np.array(fraud_probs)

    print("Running ONNX inference on validation set...")
    val_probs = get_raw_probs(x_scaled_val, y_true_val)
    print("Running ONNX inference on test set...")
    test_probs = get_raw_probs(x_scaled_test, y_true_test)

    print("Fitting Calibrators...")

    # 1. Platt Scaling (Sigmoid)
    cal_sigmoid = PlattCalibrator()
    cal_sigmoid.fit(val_probs, y_true_val)

    # 2. Isotonic Regression
    cal_isotonic = IsoCalibrator()
    cal_isotonic.fit(val_probs, y_true_val)

    def evaluate_calibrator(name, probs, y_true):
        brier = brier_score_loss(y_true, probs)
        auc = roc_auc_score(y_true, probs)
        preds = (probs >= 0.50).astype(int)
        prec = precision_score(y_true, preds, zero_division=0)
        rec = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)
        print(f"--- {name} ---")
        print(f"Brier Score: {brier:.4f}")
        print(f"AUC-ROC    : {auc:.4f}")
        print(f"Precision  : {prec:.4f}")
        print(f"Recall     : {rec:.4f}")
        print(f"F1-Score   : {f1:.4f}")
        return brier, f1, rec

    # Uncalibrated baseline (using threshold 0.50 to see the impact of calibration)
    print("\n--- Uncalibrated (Threshold 0.50) ---")
    evaluate_calibrator("Uncalibrated", test_probs, y_true_test)

    sig_probs = cal_sigmoid.predict_proba(test_probs)[:, 1]
    iso_probs = cal_isotonic.predict_proba(test_probs)[:, 1]

    print("\n--- Calibrated Evaluation ---")
    brier_sig, f1_sig, rec_sig = evaluate_calibrator(
        "Platt Scaling (Sigmoid)", sig_probs, y_true_test
    )
    brier_iso, f1_iso, rec_iso = evaluate_calibrator(
        "Isotonic Regression", iso_probs, y_true_test
    )

    # Selection Logic
    calibrators = [
        {
            "name": "Platt Scaling",
            "calibrator": cal_sigmoid,
            "f1": f1_sig,
            "rec": rec_sig,
        },
        {
            "name": "Isotonic Regression",
            "calibrator": cal_isotonic,
            "f1": f1_iso,
            "rec": rec_iso,
        },
    ]

    valid_cals = [c for c in calibrators if c["rec"] >= 0.75]
    if valid_cals:
        best_cal = max(valid_cals, key=lambda x: x["f1"])
    else:
        best_cal = max(calibrators, key=lambda x: x["rec"])

    print(
        f"\n{best_cal['name']} won! Saving to artifacts/production/prob_calibrator.pkl"
    )
    best_calibrator = best_cal["calibrator"]

    with open("artifacts/production/prob_calibrator.pkl", "wb") as f:
        pickle.dump(best_calibrator, f)

    # Update config to 0.50
    config["training"]["decision_threshold"] = 0.50
    with open("config/base_config.yaml", "w") as f:
        yaml.dump(config, f)
    print("Updated base_config.yaml decision_threshold to 0.50")


if __name__ == "__main__":
    calibrate()
