import pickle
import pandas as pd
import numpy as np
import onnxruntime as ort
import yaml
from src.pipelines.ieee_pipeline import IEEECISPipeline
from src.graph.graph_builder import TransactionGraphBuilder
from src.evaluation.metrics import FraudEvaluationEngine

def evaluate_set(name, skiprows, nrows, config, scaler, session, input_name, edge_name, decision_threshold):
    print(f"\n{'='*60}")
    print(f"  EVALUATING: {name}")
    print(f"{'='*60}")
    
    tx_path = config["data"]["ieee"]["transaction_path"]
    id_path = config["data"]["ieee"]["identity_path"]
    
    pipeline = IEEECISPipeline(config)
    def load_raw():
        # if skiprows > 0, we must keep header
        if skiprows > 0:
            df_txn = pd.read_csv(tx_path, skiprows=range(1, skiprows+1), nrows=nrows)
        else:
            df_txn = pd.read_csv(tx_path, nrows=nrows)
        df_id = pd.read_csv(id_path)
        return df_txn, df_id
        
    pipeline.load_raw = load_raw
    processed_df = pipeline.run_pipeline()
    
    builder = TransactionGraphBuilder(config)
    graph_data = builder.build_inductive_graph(processed_df)
    
    x_scaled = scaler.transform(graph_data.x.numpy()).astype(np.float32)
    y_true = graph_data.y.numpy()
    
    n_nodes = len(y_true)
    fraud_probs = []
    batch_size = 256
    
    for start in range(0, n_nodes, batch_size):
        end = min(start + batch_size, n_nodes)
        batch_x = x_scaled[start:end]
        batch_len = end - start
        local_edges = np.array([list(range(batch_len)), list(range(batch_len))], dtype=np.int64)
        raw_logits = session.run(None, {input_name: batch_x, edge_name: local_edges})[0]
        logits_s = raw_logits - raw_logits.max(axis=-1, keepdims=True)
        exp_l = np.exp(logits_s)
        probs = exp_l / exp_l.sum(axis=-1, keepdims=True)
        fraud_probs.extend(probs[:, 1].tolist())
        
    y_prob = np.array(fraud_probs)
    
    eval_engine = FraudEvaluationEngine(config)
    metrics = eval_engine.compute_standard_metrics(y_true, y_prob)
    
    y_pred = (y_prob >= decision_threshold).astype(int)
    accuracy = float(np.mean(y_pred == y_true))
    
    print(f"  Threshold used       : {decision_threshold}")
    print(f"  Overall Accuracy     : {accuracy * 100:.2f}%")
    for k, v in metrics.items():
        if 'Precision_at' in k:
            print(f"  {k:<35}: {v*100:.2f}%")
        else:
            print(f"  {k:<35}: {v:.4f}")

if __name__ == "__main__":
    with open("config/base_config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    decision_threshold = config["training"].get("decision_threshold", 0.38)
    
    with open("artifacts/production/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
        
    sess_options = ort.SessionOptions()
    sess_options.log_severity_level = 3
    session = ort.InferenceSession("artifacts/production/fraud_model.onnx", sess_options=sess_options)
    
    input_name = session.get_inputs()[0].name
    edge_name = session.get_inputs()[1].name
    
    # Evaluate Training Set (first 200,000 rows)
    evaluate_set("TRAINING SET (In-Sample)", 0, 200000, config, scaler, session, input_name, edge_name, decision_threshold)
    
    # Evaluate Testing Set (next 50,000 rows, skipping the first 200000)
    evaluate_set("TESTING SET (Out-of-Sample)", 200000, 50000, config, scaler, session, input_name, edge_name, decision_threshold)
