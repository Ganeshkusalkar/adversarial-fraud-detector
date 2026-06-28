"""
src/main.py
===========
Central pipeline entry point for training and evaluating the 
Adversarial Transaction Disguise Detector model across multiple datasets.
"""

import argparse
import os
import pickle
import yaml
import numpy as np
import pandas as pd
import onnxruntime as ort
import torch
from sklearn.preprocessing import StandardScaler

from src.pipelines.ieee_pipeline import IEEECISPipeline
from src.pipelines.paysim_pipeline import PaySimPipeline
from src.pipelines.elliptic_pipeline import EllipticPipeline
from src.graph.graph_builder import TransactionGraphBuilder
from src.models.discriminator_gnn import FraudGNN
from src.models.generator_lstm import FraudTransactionGenerator
from src.training.engine import AdversarialTrainingEngine
from src.evaluation.metrics import FraudEvaluationEngine
from src.utils.logger import setup_logger

logger = setup_logger("MainPipeline")

# Directories and paths
ARTIFACTS_DIR = "artifacts/production"
CHECKPOINT_DIR = "artifacts/checkpoints"
ONNX_PATH = os.path.join(ARTIFACTS_DIR, "fraud_model.onnx")
SCALER_PATH = os.path.join(ARTIFACTS_DIR, "scaler.pkl")
ENCODERS_PATH = os.path.join(ARTIFACTS_DIR, "label_encoders.pkl")


def load_config(config_path="config/base_config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train_ieee(config):
    logger.info("=" * 70)
    logger.info("  TRAINING ON IEEE-CIS DATASET (NODE FEATURES)")
    logger.info("=" * 70)

    tx_path = config["data"]["ieee"]["transaction_path"]
    id_path = config["data"]["ieee"]["identity_path"]
    decision_threshold = config["training"].get("decision_threshold", 0.38)
    total_epochs = config["training"].get("epochs", 30)
    batch_size = config["training"].get("batch_size", 64)

    pipeline = IEEECISPipeline(config)
    train_rows = config["data"]["ieee"].get("train_rows", 50_000)

    def load_raw_subset():
        logger.info(f"Loading transaction subset from: {tx_path} ({train_rows:,} rows)")
        df_txn = pd.read_csv(tx_path, nrows=train_rows)
        df_id = pd.read_csv(id_path)
        return df_txn, df_id

    pipeline.load_raw = load_raw_subset
    processed_df = pipeline.run_pipeline()

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    with open(ENCODERS_PATH, "wb") as f:
        pickle.dump(pipeline.label_encoders, f)

    builder = TransactionGraphBuilder(config)
    graph_data = builder.build_inductive_graph(processed_df)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(graph_data.x.numpy())
    graph_data.x = torch.tensor(x_scaled, dtype=torch.float32)

    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    y_np = graph_data.y.numpy()
    n_positive = int(y_np.sum())
    n_negative = int(len(y_np) - n_positive)
    if n_positive == 0:
        n_positive = max(1, int(n_negative * 0.035))
        
    # NEW: Compute per-node sample weights for WeightedRandomSampler
    node_weights = np.ones(len(y_np))
    # Legitimate nodes get 1.0, fraud nodes get (n_negative / n_positive)
    node_weights[y_np == 1] = float(n_negative / n_positive)

    gnn_config = config["model"]["gnn"]
    model = FraudGNN(
        in_channels=gnn_config["in_channels"],
        hidden_channels=gnn_config["hidden_channels"],
        out_channels=gnn_config["out_channels"],
        dropout=gnn_config["dropout"],
        decision_threshold=decision_threshold,
    )

    gen_config = config["model"]["generator"]
    generator = FraudTransactionGenerator(
        noise_dim=gen_config["noise_dim"],
        hidden_dim=gen_config["hidden_dim"],
        sequence_length=gen_config["sequence_length"],
        feature_dim=gen_config["feature_dim"],
        num_layers=gen_config["num_layers"],
    )

    engine = AdversarialTrainingEngine(
        gnn_model=model,
        gen_model=generator,
        device="cpu",
        n_negative=n_negative,
        n_positive=n_positive,
        checkpoint_dir=CHECKPOINT_DIR,
        total_epochs=total_epochs,
        sample_weights=node_weights,
    )

    engine.run_training_orchestration(
        real_graph_data=graph_data,
        total_epochs=total_epochs,
        batch_size=batch_size,
    )

    # Export ONNX
    model.eval()
    dummy_x = torch.randn(1, gnn_config["in_channels"])
    dummy_edge = torch.zeros((2, 1), dtype=torch.long)

    torch.onnx.export(
        model,
        (dummy_x, dummy_edge),
        ONNX_PATH,
        input_names=["x", "edge_index"],
        output_names=["logits"],
        dynamic_axes={
            "x": {0: "num_nodes"},
            "edge_index": {1: "num_edges"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    logger.info(f"[OK] ONNX model exported successfully to: {ONNX_PATH}")


def train_paysim(config):
    logger.info("=" * 70)
    logger.info("  TRAINING ON PAYSIM DATASET (TEMPORAL SEQUENCES)")
    logger.info("=" * 70)
    try:
        pipeline = PaySimPipeline(config)
        sequences = pipeline.run_pipeline()
        logger.info(f"Successfully processed {len(sequences)} PaySim sequences.")
        logger.info("Note: Sequence training logic for LSTM integration would run here.")
    except Exception as e:
        logger.error(f"PaySim pipeline failed (missing data?): {e}")


def train_elliptic(config):
    logger.info("=" * 70)
    logger.info("  TRAINING ON ELLIPTIC DATASET (NATIVE GRAPH)")
    logger.info("=" * 70)
    try:
        pipeline = EllipticPipeline(config)
        graph_data = pipeline.run_pipeline()
        logger.info(f"Successfully built Elliptic graph: {graph_data.num_nodes} nodes, {graph_data.num_edges} edges.")
        logger.info("Note: Elliptic graph training loop would run here.")
    except Exception as e:
        logger.error(f"Elliptic pipeline failed (missing data?): {e}")


def evaluate_ieee(config):
    print("=" * 60)
    print("  EVALUATION PIPELINE - REAL IEEE-CIS Data")
    print("=" * 60)

    tx_path = config["data"]["ieee"]["transaction_path"]
    id_path = config["data"]["ieee"]["identity_path"]
    decision_threshold = config["training"].get("decision_threshold", 0.5375)
    train_rows = config["data"]["ieee"].get("train_rows", 200_000)

    def evaluate_subset(name, skiprows, nrows):
        print(f"\n{'='*60}")
        print(f"  EVALUATING: {name}")
        print(f"{'='*60}")
        pipeline = IEEECISPipeline(config)
        
        def load_raw_subset():
            if skiprows > 0:
                df_txn = pd.read_csv(tx_path, skiprows=range(1, skiprows+1), nrows=nrows)
            else:
                df_txn = pd.read_csv(tx_path, nrows=nrows)
            df_id = pd.read_csv(id_path)
            return df_txn, df_id

        pipeline.load_raw = load_raw_subset
        with open(ENCODERS_PATH, "rb") as f:
            pipeline.label_encoders = pickle.load(f)
            
        processed_df = pipeline.run_pipeline(fit=False)
        builder = TransactionGraphBuilder(config)
        graph_data = builder.build_inductive_graph(processed_df)
        y_true = graph_data.y.numpy()

        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        x_scaled = scaler.transform(graph_data.x.numpy()).astype(np.float32)

        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 3
        session = ort.InferenceSession(ONNX_PATH, sess_options=sess_options)

        input_name = session.get_inputs()[0].name
        edge_name  = session.get_inputs()[1].name

        n_nodes = len(y_true)
        fraud_probs = []
        batch_size = 256
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

        y_prob = np.array(fraud_probs)

        eval_engine = FraudEvaluationEngine(config)
        metrics = eval_engine.compute_standard_metrics(y_true, y_prob)

        y_pred = (y_prob >= decision_threshold).astype(int)
        accuracy = float(np.mean(y_pred == y_true))

        print("=" * 60)
        print(f"  RESULTS FOR: {name}")
        print("=" * 60)
        print(f"  Threshold used       : {decision_threshold}")
        print(f"  Overall Accuracy     : {accuracy * 100:.2f}%")
        for k, v in metrics.items():
            if "Precision_at" in k:
                print(f"  {k:<35}: {v*100:.2f}%")
            else:
                print(f"  {k:<35}: {v:.4f}")
        print("=" * 60)

    # Evaluate Train (In-Sample)
    evaluate_subset("TRAINING SET (In-Sample)", 0, train_rows)
    
    # Evaluate Test (Out-of-Sample)
    # The user asked for "rows 100,001 to 150,000", which corresponds to skiprows=100000, nrows=50000
    evaluate_subset("TESTING SET (Out-of-Sample)", 100000, 50000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adversarial Fraud Detector Pipeline")
    parser.add_argument("action", choices=["train", "evaluate"], help="Action to perform")
    parser.add_argument("--dataset", choices=["ieee", "paysim", "elliptic", "all"], default="ieee", help="Target dataset")
    
    args = parser.parse_args()
    config = load_config()

    if args.action == "train":
        if args.dataset in ["ieee", "all"]:
            train_ieee(config)
        if args.dataset in ["paysim", "all"]:
            train_paysim(config)
        if args.dataset in ["elliptic", "all"]:
            train_elliptic(config)
            
    elif args.action == "evaluate":
        if args.dataset in ["ieee", "all"]:
            evaluate_ieee(config)
        if args.dataset in ["paysim", "elliptic"]:
            logger.warning(f"Evaluation for {args.dataset} is not fully integrated. Please evaluate IEEE-CIS.")
