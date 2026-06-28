import yaml
import onnxruntime as ort
import pickle
import numpy as np
import pandas as pd
from src.evaluation.metrics import FraudEvaluationEngine
from src.pipelines.ieee_pipeline import IEEECISPipeline
from src.graph.graph_builder import TransactionGraphBuilder
import os


def calibrate():
    with open("config/base_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    old_threshold = config["training"].get("decision_threshold", 0.38)

    # Load Validation Set (e.g. 50,000 rows skipping the first 100,000)
    tx_path = config["data"]["ieee"]["transaction_path"]
    id_path = config["data"]["ieee"]["identity_path"]

    pipeline = IEEECISPipeline(config)

    def load_val():
        # Skip first 100000 rows (used for train/test), load next 50000
        df_txn = pd.read_csv(tx_path, skiprows=range(1, 100001), nrows=50000)
        df_id = pd.read_csv(id_path)
        return df_txn, df_id

    pipeline.load_raw = load_val
    with open("artifacts/production/label_encoders.pkl", "rb") as f:
        pipeline.label_encoders = pickle.load(f)

    processed_df = pipeline.run_pipeline(fit=False)
    builder = TransactionGraphBuilder(config)
    graph_data = builder.build_inductive_graph(processed_df)

    with open("artifacts/production/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    x_scaled = scaler.transform(graph_data.x.numpy()).astype(np.float32)
    y_true = graph_data.y.numpy()

    sess_options = ort.SessionOptions()
    sess_options.log_severity_level = 3
    session = ort.InferenceSession(
        "artifacts/production/fraud_model.onnx", sess_options=sess_options
    )

    input_name = session.get_inputs()[0].name
    edge_name = session.get_inputs()[1].name

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
        raw_logits = session.run(None, {input_name: batch_x, edge_name: local_edges})[0]
        logits_s = raw_logits - raw_logits.max(axis=-1, keepdims=True)
        exp_l = np.exp(logits_s)
        probs = exp_l / exp_l.sum(axis=-1, keepdims=True)
        fraud_probs.extend(probs[:, 1].tolist())

    y_prob = np.array(fraud_probs)

    engine = FraudEvaluationEngine(config)

    # Evaluate at old threshold for comparison
    old_metrics = engine.compute_standard_metrics(y_true, y_prob)
    old_f1 = old_metrics["F1_Score_Fraud_Class"]

    # Find new optimal threshold
    new_threshold = engine.find_threshold_with_constraints(
        y_true, y_prob, min_recall=0.85, min_precision=0.40
    )

    engine.decision_threshold = new_threshold
    new_metrics = engine.compute_standard_metrics(y_true, y_prob)
    new_f1 = new_metrics["F1_Score_Fraud_Class"]
    new_recall = new_metrics["Recall_Sensitivity"]

    print("=" * 60)
    print(f"Calibration Summary:")
    print(f"  Old Threshold: {old_threshold:.4f} -> New Threshold: {new_threshold:.4f}")
    print(f"  Old F1: {old_f1:.4f} -> New F1: {new_f1:.4f}")
    print(f"  Recall at new threshold: {new_recall:.4f}")
    print("=" * 60)

    # Save back to config
    config["training"]["decision_threshold"] = float(new_threshold)
    with open("config/base_config.yaml", "w") as f:
        yaml.dump(config, f)
    print("Updated config/base_config.yaml with new threshold.")


if __name__ == "__main__":
    calibrate()
